// ldls_driver: the shared C++ driver.
//
//   explain    --doc D.json                                       sequence / unit layout and what each sequence sees
//   ask        --model M --doc D.json --node ID --question Q      prefill (structured or flat) and answer one question
//   run-tests  --model M --doc D.json --doc-mod-b B.json --doc-mod-a A.json
//              --target ID --modify ID --evict ID --question Q --out R.json
//                                                                 test 1 (change B, ask A), test 2 (evict C), controls
//   batched    --model M --prompt P1 --prompt P2 [--shared-prefix S --suffix X1 --suffix X2] --out R.json
//                                                                 test 3 (batched vs separate runs)
//
// Common options: --threads N (8), --ctx N (8192), --n-predict N (48), --verbose

#include "engine.hpp"
#include "structured_doc.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------------------------
// argument parsing

struct Args {
    std::string cmd;
    std::map<std::string, std::vector<std::string>> kv;
    std::set<std::string> flags;

    bool has(const std::string & k) const { return kv.count(k) || flags.count(k); }
    std::string get(const std::string & k, const std::string & def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second.back();
    }
    std::vector<std::string> all(const std::string & k) const {
        auto it = kv.find(k);
        return it == kv.end() ? std::vector<std::string>{} : it->second;
    }
    int geti(const std::string & k, int def) const { return has(k) ? std::stoi(get(k)) : def; }
    std::string req(const std::string & k) const {
        if (!kv.count(k)) {
            throw std::runtime_error("missing required option --" + k);
        }
        return get(k);
    }
};

static Args parse_args(int argc, char ** argv) {
    static const std::set<std::string> boolean = {"verbose", "flat", "trace-generation"};
    Args a;
    if (argc < 2) {
        throw std::runtime_error("usage: ldls_driver <explain|ask|run-tests|batched> [options]");
    }
    a.cmd = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0) {
            throw std::runtime_error("unexpected argument " + k);
        }
        k = k.substr(2);
        if (boolean.count(k)) {
            a.flags.insert(k);
        } else {
            if (i + 1 >= argc) {
                throw std::runtime_error("option --" + k + " needs a value");
            }
            a.kv[k].push_back(argv[++i]);
        }
    }
    return a;
}

static EngineParams engine_params(const Args & a, int n_seq_max) {
    EngineParams p;
    p.model_path = a.req("model");
    p.n_ctx      = a.geti("ctx", 8192);
    p.n_batch    = p.n_ctx;
    p.n_threads  = a.geti("threads", 8);
    p.n_seq_max  = std::max(1, n_seq_max);
    p.verbose    = a.has("verbose");
    p.trace_generation = a.has("trace-generation");
    return p;
}

// ---------------------------------------------------------------------------------------------
// helpers

static json answer_json(const Answer & a) {
    json j;
    j["seq"]             = a.seq;
    j["text"]            = a.text;
    j["tokens"]          = a.tokens;
    j["n_question"]      = a.n_question;
    j["t_question_ms"]   = a.t_question_ms;
    j["t_generate_ms"]   = a.t_generate_ms;
    j["gen_tokens_per_s"] = a.tokens.empty() ? 0.0 : 1000.0 * a.tokens.size() / std::max(a.t_generate_ms, 1e-9);
    if (!a.generation_trace.empty()) {
        j["generation_trace"] = json::array();
        for (const auto & s : a.generation_trace) {
            j["generation_trace"].push_back({{"token", s.token}, {"runner_up", s.runner_up},
                {"logit", s.logit}, {"runner_up_logit", s.runner_up_logit},
                {"margin", s.logit - s.runner_up_logit}, {"end_of_generation", s.end_of_generation}});
        }
    }
    return j;
}

static std::vector<TokenEntry> flat_entries(const std::vector<llama_token> & toks) {
    std::vector<TokenEntry> out;
    for (size_t i = 0; i < toks.size(); ++i) {
        out.push_back({toks[i], (llama_pos) i, {0}, -1});
    }
    return out;
}

static int resolve_node(const StructuredDoc & d, const std::string & id) {
    const int k = d.find(id);
    if (k < 0) {
        throw std::runtime_error("unknown node id " + id + " in " + d.path);
    }
    return k;
}

// ---------------------------------------------------------------------------------------------
// incremental update: bring a cache that holds `oldd` to the state of `newd`

struct PatchReport {
    std::vector<llama_seq_id> changed_seqs;
    std::vector<llama_pos>    first_diff;      // per changed seq
    int    n_redecoded = 0;                    // token entries decoded again (shared tokens count once)
    double t_rm_ms = 0.0;
    double t_decode_ms = 0.0;
};

static PatchReport patch_cache(Engine & eng, const StructuredDoc & oldd, const StructuredDoc & newd) {
    if (oldd.n_seq != newd.n_seq) {
        throw std::runtime_error("patch: the two document versions use a different number of sequences");
    }
    const auto so = oldd.seq_streams();
    const auto sn = newd.seq_streams();

    PatchReport r;
    std::vector<llama_pos> cut(oldd.n_seq, -1);
    for (int s = 0; s < oldd.n_seq; ++s) {
        size_t p = 0;
        while (p < so[s].size() && p < sn[s].size() && so[s][p] == sn[s][p]) {
            ++p;
        }
        if (p == so[s].size() && p == sn[s].size()) {
            continue;
        }
        cut[s] = (llama_pos) p;
        r.changed_seqs.push_back(s);
        r.first_diff.push_back((llama_pos) p);
    }

    // drop everything from the first differing position onwards, in the changed sequences only
    double t0 = now_ms();
    for (auto s : r.changed_seqs) {
        if (!eng.seq_rm(s, cut[s], -1)) {
            throw std::runtime_error("patch: llama_memory_seq_rm failed");
        }
    }
    r.t_rm_ms = now_ms() - t0;

    // re-decode the new tokens of those sequences; a token shared by several changed sequences
    // is decoded once with all of them (they share the same prefix, so their cut points agree)
    std::vector<TokenEntry> todo;
    for (const auto & e : newd.structured_entries()) {
        TokenEntry x = e;
        x.seqs.clear();
        for (auto s : e.seqs) {
            if (cut[s] >= 0 && e.pos >= cut[s]) {
                x.seqs.push_back(s);
            }
        }
        if (!x.seqs.empty()) {
            todo.push_back(x);
        }
    }
    r.n_redecoded = (int) todo.size();
    r.t_decode_ms = eng.decode(todo);
    return r;
}

static json patch_json(const PatchReport & r) {
    json j;
    j["changed_seqs"]  = r.changed_seqs;
    j["first_diff_pos"] = r.first_diff;
    j["n_redecoded"]   = r.n_redecoded;
    j["t_rm_ms"]       = r.t_rm_ms;
    j["t_decode_ms"]   = r.t_decode_ms;
    return j;
}

// ---------------------------------------------------------------------------------------------
// eviction of a subtree by position range

struct EvictReport {
    struct Range { llama_seq_id seq; llama_pos p0, p1; };
    std::vector<Range> ranges;
    bool range_exact = true;   // each range contains only tokens of the subtree
    int  expected_freed = 0;   // cells whose every sequence is covered by a range
    int  subtree_tokens = 0;
    double t_ms = 0.0;
};

static EvictReport evict_subtree(Engine & eng, const StructuredDoc & d, int node) {
    EvictReport r;
    std::set<int> sub;
    for (int k : d.subtree(node)) {
        sub.insert(k);
        r.subtree_tokens += (int) d.nodes[k].tokens.size();
    }

    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> range; // seq -> [min, max]
    for (int k : sub) {
        const auto & n = d.nodes[k];
        for (size_t i = 0; i < n.tokens.size(); ++i) {
            for (auto s : n.seq_ids) {
                auto it = range.find(s);
                if (it == range.end()) {
                    range[s] = {n.positions[i], n.positions[i]};
                } else {
                    it->second.first  = std::min(it->second.first,  n.positions[i]);
                    it->second.second = std::max(it->second.second, n.positions[i]);
                }
            }
        }
    }

    const auto entries = d.structured_entries();
    for (const auto & e : entries) {
        bool all_covered = true;
        for (auto s : e.seqs) {
            auto it = range.find(s);
            const bool in = it != range.end() && e.pos >= it->second.first && e.pos <= it->second.second;
            if (in && !sub.count(e.node)) {
                r.range_exact = false; // a foreign token sits inside the position range
            }
            all_covered = all_covered && in;
        }
        if (all_covered) {
            ++r.expected_freed;
        }
    }

    const double t0 = now_ms();
    for (const auto & [s, mm] : range) {
        if (!eng.seq_rm(s, mm.first, mm.second + 1)) {
            throw std::runtime_error("evict: llama_memory_seq_rm failed");
        }
        r.ranges.push_back({s, mm.first, mm.second + 1});
    }
    r.t_ms = now_ms() - t0;
    return r;
}

// ---------------------------------------------------------------------------------------------
// explain

static int cmd_explain(const Args & a) {
    const StructuredDoc d = StructuredDoc::load(a.req("doc"));
    d.validate(0);

    printf("document %s (%s): %d nodes, %d document tokens + %zu prefix tokens, %d sequences\n\n",
           d.document_id.c_str(), d.domain.c_str(), (int) d.nodes.size(), d.n_doc_tokens(), d.prefix_tokens.size(), d.n_seq);

    printf("units (positions are [start, end) inside every sequence of the unit):\n");
    for (const auto & u : d.units) {
        std::string seqs;
        for (auto s : u.seq_ids) {
            seqs += (seqs.empty() ? "" : ",") + std::to_string(s);
        }
        std::string label;
        for (const auto & nid : u.nodes) {
            const auto & n = d.nodes[d.find(nid)];
            if (!n.title.empty()) {
                label = n.title;
                break;
            }
        }
        printf("  %-5s %-12s parent=%-5s pos=[%4d,%4d) tokens=%4d seqs={%s}  %s\n", u.id.c_str(), u.kind.c_str(),
               u.parent_unit.empty() ? "-" : u.parent_unit.c_str(), u.pos_start, u.pos_end, u.pos_end - u.pos_start,
               seqs.c_str(), label.c_str());
    }

    const auto st = d.seq_streams();
    const int flat = (int) d.flat_tokens().size();
    printf("\nwhat each sequence attends to (prefix + units on its root-to-leaf path):\n");
    for (int s = 0; s < d.n_seq; ++s) {
        std::vector<std::string> path;
        for (const auto & u : d.units) {
            if (std::find(u.seq_ids.begin(), u.seq_ids.end(), s) != u.seq_ids.end() && u.pos_end > u.pos_start) {
                path.push_back(u.id);
            }
        }
        std::string p;
        for (auto & x : path) {
            p += (p.empty() ? "" : " > ") + x;
        }
        printf("  seq %2d: %4zu of %4d tokens (%5.1f%%)  %s\n", s, st[s].size(), flat, 100.0 * st[s].size() / flat, p.c_str());
    }
    return 0;
}

// ---------------------------------------------------------------------------------------------
// ask

static int cmd_ask(const Args & a) {
    const StructuredDoc d = StructuredDoc::load(a.req("doc"));
    d.validate(0);
    Engine eng(engine_params(a, d.n_seq));
    const int n_predict = a.geti("n-predict", 48);
    const bool flat = a.has("flat");
    const auto q = eng.tokenize(d.question_text(a.req("question")), true);

    double t_prefill;
    llama_seq_id seq = 0;
    if (flat) {
        t_prefill = eng.decode(flat_entries(d.flat_tokens()));
    } else {
        t_prefill = eng.decode(d.structured_entries());
        seq = a.has("seq") ? std::stoi(a.get("seq")) : d.target_seq(resolve_node(d, a.req("node")));
    }
    const int cells = eng.used_cells();
    const Answer ans = eng.ask(seq, q, n_predict);

    printf("mode=%s seq=%d visible=%d cells=%d prefill=%.1f ms\n", flat ? "flat" : "structured", seq,
           eng.seq_pos_max(seq) + 1, cells, t_prefill);
    printf("Q: %s\nA: %s\n", a.get("question").c_str(), ans.text.c_str());
    return 0;
}

// ---------------------------------------------------------------------------------------------
// run-tests: tests 1 and 2 of the assignment plus controls, on one document

static int cmd_run_tests(const Args & a) {
    const StructuredDoc d0 = StructuredDoc::load(a.req("doc"));
    const StructuredDoc dB = StructuredDoc::load(a.req("doc-mod-b"));
    const StructuredDoc dA = StructuredDoc::load(a.req("doc-mod-a"));
    const int n_seq_max = std::max({d0.n_seq, dB.n_seq, dA.n_seq});

    Engine eng(engine_params(a, n_seq_max));
    for (const auto * d : {&d0, &dB, &dA}) {
        d->validate(eng.params().n_seq_max);
    }
    const int n_predict = a.geti("n-predict", 48);

    const int target = resolve_node(d0, a.req("target"));
    const int modify = resolve_node(d0, a.req("modify"));
    const int evict  = resolve_node(d0, a.req("evict"));
    const llama_seq_id seq = d0.target_seq(target);
    const std::string question = a.req("question");
    const auto q = eng.tokenize(d0.question_text(question), true);

    json out;
    out["document_id"] = d0.document_id;
    out["question"]    = question;
    out["engine"] = {
        {"model", eng.params().model_path}, {"n_ctx", eng.n_ctx()}, {"n_threads", eng.params().n_threads},
        {"n_seq_max", eng.params().n_seq_max}, {"kv_unified", eng.params().kv_unified},
        {"kv_bytes_per_cell", eng.kv_bytes_per_cell()}, {"n_predict", n_predict},
    };
    out["nodes"] = {
        {"target", {{"id", d0.nodes[target].id}, {"path", d0.nodes[target].path}}},
        {"modify", {{"id", d0.nodes[modify].id}, {"path", d0.nodes[modify].path}}},
        {"evict",  {{"id", d0.nodes[evict].id},  {"path", d0.nodes[evict].path}}},
    };
    const auto streams = d0.seq_streams();
    out["doc"] = {
        {"n_seq", d0.n_seq}, {"n_doc_tokens", d0.n_doc_tokens()}, {"n_prefix_tokens", d0.prefix_tokens.size()},
        {"flat_tokens", d0.flat_tokens().size()}, {"target_seq", seq}, {"target_seq_tokens", streams[seq].size()},
    };
    json timings;
    timings["model_load_ms"] = eng.load_ms();

    // --- prefill timing: flat vs structured, repeated (median is reported by the report script) ----
    {
        const int reps = std::max(1, a.geti("repeat", 3));
        std::vector<double> tf, ts;
        for (int r = 0; r < reps; ++r) {
            eng.clear();
            tf.push_back(eng.decode(flat_entries(d0.flat_tokens())));
            eng.clear();
            ts.push_back(eng.decode(d0.structured_entries()));
        }
        timings["prefill_flat_ms_runs"] = tf;
        timings["prefill_structured_ms_runs"] = ts;
    }

    // --- flat (full attention) baseline and negative control -----------------------------------
    eng.clear();
    timings["prefill_flat_ms"] = eng.decode(flat_entries(d0.flat_tokens()));
    const int cells_flat = eng.used_cells();
    const Answer flat0 = eng.ask(0, q, n_predict);
    eng.clear();
    eng.decode(flat_entries(dB.flat_tokens()));
    const Answer flatB = eng.ask(0, q, n_predict);
    out["flat"] = {
        {"cells", cells_flat},
        {"original", answer_json(flat0)},
        {"modified_b", answer_json(flatB)},
        {"answer_changed_by_b", flat0.tokens != flatB.tokens},
    };

    // --- structured baseline -------------------------------------------------------------------
    eng.clear();
    timings["prefill_structured_ms"] = eng.decode(d0.structured_entries());
    const int cells0 = eng.used_cells();
    const Answer base = eng.ask(seq, q, n_predict);
    out["structured"] = {
        {"cells", cells0},
        {"cells_expected", (int) d0.structured_entries().size()},
        {"target_seq_cells", eng.used_cells(seq)},
        {"answer", answer_json(base)},
    };
    timings["ask_structured_ms"] = base.t_question_ms + base.t_generate_ms;
    timings["ask_flat_ms"]       = flat0.t_question_ms + flat0.t_generate_ms;

    // --- test 1: change subtree B, ask about subtree A -----------------------------------------
    {
        const PatchReport pr = patch_cache(eng, d0, dB);
        const bool untouched = std::find(pr.changed_seqs.begin(), pr.changed_seqs.end(), seq) == pr.changed_seqs.end();
        const int cells_patched = eng.used_cells();
        const Answer ans = eng.ask(seq, q, n_predict);

        // reference: the modified document prefilled from scratch (detects stale cache state)
        eng.clear();
        timings["full_reprefill_b_ms"] = eng.decode(dB.structured_entries());
        const int cells_fresh = eng.used_cells();
        const Answer fresh = eng.ask(seq, q, n_predict);

        const bool same = ans.tokens == base.tokens;
        const bool same_as_fresh = ans.tokens == fresh.tokens;
        json t;
        t["patch"]                    = patch_json(pr);
        t["target_seq_untouched"]     = untouched;
        t["answer"]                   = answer_json(ans);
        t["tokens_equal_to_baseline"] = same;
        t["max_abs_logit_diff"]       = max_abs_diff(base.logits[0], ans.logits[0]);
        t["fresh_prefill_answer"]     = answer_json(fresh);
        t["patched_equals_fresh"]     = same_as_fresh;
        t["patched_vs_fresh_logit_diff"] = max_abs_diff(fresh.logits[0], ans.logits[0]);
        t["cells_patched"]            = cells_patched;
        t["cells_fresh"]              = cells_fresh;
        t["pass"] = untouched && same && same_as_fresh && cells_patched == cells_fresh;
        out["test1_modify_b"] = t;
        timings["patch_b_ms"] = pr.t_rm_ms + pr.t_decode_ms;
    }

    // --- positive control: change subtree A itself; the answer is expected to change -----------
    {
        eng.clear();
        eng.decode(d0.structured_entries());
        const PatchReport pr = patch_cache(eng, d0, dA);
        const bool touched = std::find(pr.changed_seqs.begin(), pr.changed_seqs.end(), seq) != pr.changed_seqs.end();
        const Answer ans = eng.ask(seq, q, n_predict);
        out["control_modify_a"] = {
            {"patch", patch_json(pr)},
            {"target_seq_changed", touched},
            {"answer", answer_json(ans)},
            {"answer_changed", ans.tokens != base.tokens},
        };
    }

    // --- test 2: evict an unrelated subtree by position range -----------------------------------
    {
        eng.clear();
        eng.decode(d0.structured_entries());
        const int before = eng.used_cells();
        const size_t bytes_before = eng.state_bytes();
        const EvictReport er = evict_subtree(eng, d0, evict);
        const int after = eng.used_cells();
        const size_t bytes_after = eng.state_bytes();
        const Answer ans = eng.ask(seq, q, n_predict);

        json ranges = json::array();
        for (const auto & r : er.ranges) {
            ranges.push_back({{"seq", r.seq}, {"p0", r.p0}, {"p1", r.p1}});
        }
        const bool target_hit = std::any_of(er.ranges.begin(), er.ranges.end(), [&](const auto & r) { return r.seq == seq; });
        const bool same = ans.tokens == base.tokens;
        json t;
        t["ranges"]          = ranges;
        t["range_exact"]     = er.range_exact;
        t["target_seq_in_ranges"] = target_hit;
        t["subtree_tokens"]  = er.subtree_tokens;
        t["cells_before"]    = before;
        t["cells_after"]     = after;
        t["cells_freed"]     = before - after;
        t["cells_expected_freed"] = er.expected_freed;
        t["kv_bytes_freed"]  = (long long) (before - after) * (long long) eng.kv_bytes_per_cell();
        t["state_bytes_before"] = bytes_before;
        t["state_bytes_after"]  = bytes_after;
        t["t_evict_ms"]      = er.t_ms;
        t["answer"]          = answer_json(ans);
        t["tokens_equal_to_baseline"] = same;
        t["max_abs_logit_diff"] = max_abs_diff(base.logits[0], ans.logits[0]);
        t["pass"] = same && after < before && (before - after) == er.expected_freed && er.range_exact;
        out["test2_evict"] = t;
        timings["evict_ms"] = er.t_ms;

        // extension: keep only the target's root-to-leaf path, drop every other sequence
        const double t0 = now_ms();
        for (int s = 0; s < d0.n_seq; ++s) {
            if (s != seq) {
                eng.seq_rm(s, -1, -1);
            }
        }
        const double t_keep = now_ms() - t0;
        const int kept = eng.used_cells();
        const Answer ans2 = eng.ask(seq, q, n_predict);
        out["test2b_keep_only_target_path"] = {
            {"cells_after", kept},
            {"cells_expected", (int) streams[seq].size()},
            {"kv_bytes_kept", (long long) kept * (long long) eng.kv_bytes_per_cell()},
            {"fraction_of_flat_cache", (double) kept / cells_flat},
            {"t_ms", t_keep},
            {"answer", answer_json(ans2)},
            {"tokens_equal_to_baseline", ans2.tokens == base.tokens},
            {"pass", ans2.tokens == base.tokens && kept == (int) streams[seq].size()},
        };
    }

    out["timings_ms"] = timings;
    const std::string path = a.get("out");
    if (!path.empty()) {
        std::ofstream(path) << out.dump(2) << std::endl;
    }
    printf("%s: test1=%s test2=%s keep-path=%s | baseline: %s\n", d0.document_id.c_str(),
           out["test1_modify_b"]["pass"].get<bool>() ? "PASS" : "FAIL",
           out["test2_evict"]["pass"].get<bool>() ? "PASS" : "FAIL",
           out["test2b_keep_only_target_path"]["pass"].get<bool>() ? "PASS" : "FAIL", base.text.c_str());
    return 0;
}

// ---------------------------------------------------------------------------------------------
// batched: test 3, sequence isolation inside one batch

// Compares two greedy runs of the same sequences. Logit differences are measured only on the steps
// where both runs still had the same history (up to and including the first diverging step);
// at that step we also report the top-2 margin, to tell a numerical near-tie from a real difference.
static json compare_runs(const std::vector<Answer> & ref, const std::vector<Answer> & bat,
                         const char * ref_name = "separate", const char * bat_name = "batched") {
    json cases = json::array();
    bool all = true;
    for (size_t k = 0; k < ref.size(); ++k) {
        const auto & a = ref[k];
        const auto & b = bat[k];
        size_t div = 0;
        while (div < a.tokens.size() && div < b.tokens.size() && a.tokens[div] == b.tokens[div]) {
            ++div;
        }
        const bool eq = a.tokens == b.tokens;
        const size_t steps = std::min({div + 1, a.logits.size(), b.logits.size()});
        double md = 0.0;
        for (size_t i = 0; i < steps; ++i) {
            md = std::max(md, max_abs_diff(a.logits[i], b.logits[i]));
        }
        json c = {
            {"seq", b.seq},
            {ref_name, answer_json(a)},
            {bat_name, answer_json(b)},
            {"tokens_equal", eq},
            {"max_abs_logit_diff_before_divergence", md},
        };
        if (!eq && div < a.tokens.size() && div < b.tokens.size() && div < a.logits.size() && div < b.logits.size()) {
            const llama_token ta = a.tokens[div];
            const llama_token tb = b.tokens[div];
            c["first_divergence_step"] = div;
            c["margin_in_" + std::string(ref_name)] = a.logits[div][ta] - a.logits[div][tb];
            c["margin_in_" + std::string(bat_name)] = b.logits[div][tb] - b.logits[div][ta];
        }
        all = all && eq;
        cases.push_back(c);
    }
    return {{"sequences", cases}, {"pass", all}};
}

static int cmd_batched(const Args & a) {
    const auto prompts  = a.all("prompt");
    const auto suffixes = a.all("suffix");
    const std::string shared = a.get("shared-prefix");
    if (prompts.size() < 2) {
        throw std::runtime_error("batched: give at least two --prompt");
    }
    const int n_seq = (int) std::max(prompts.size(), suffixes.size());
    Engine eng(engine_params(a, n_seq));
    const int n_predict = a.geti("n-predict", 32);
    json out;
    out["engine"] = {{"n_threads", eng.params().n_threads}, {"kv_unified", eng.params().kv_unified}, {"n_predict", n_predict}};

    // case A: independent prompts, one sequence each (what examples/batched does with different prompts)
    {
        std::vector<std::vector<llama_token>> toks;
        std::vector<llama_seq_id> seqs;
        for (size_t k = 0; k < prompts.size(); ++k) {
            auto t = eng.tokenize(prompts[k], true);
            t.insert(t.begin(), eng.bos()); // <|begin_of_text|>
            toks.push_back(t);
            seqs.push_back((llama_seq_id) k);
        }
        std::vector<Answer> sep;
        for (size_t k = 0; k < toks.size(); ++k) {
            eng.clear();
            sep.push_back(eng.generate_parallel({toks[k]}, {0}, n_predict, true)[0]);
        }
        eng.clear();
        const auto bat = eng.generate_parallel(toks, seqs, n_predict, true);
        json c = compare_runs(sep, bat);
        for (size_t k = 0; k < prompts.size(); ++k) {
            c["sequences"][k]["prompt"] = prompts[k];
        }
        out["independent_prompts"] = c;

        // case A': isolation with identical batch shapes. Replace the LAST prompt by a different prompt
        // of the same token length and run the batch again: every other sequence must reproduce its
        // logits bit for bit, because nothing it can attend to has changed and the kernels see the
        // same shapes. Any difference would be leakage across sequences.
        const std::string perturb = a.get("perturb");
        if (!perturb.empty()) {
            auto alt = eng.tokenize(perturb, true);
            alt.insert(alt.begin(), eng.bos());
            if (alt.size() != toks.back().size()) {
                throw std::runtime_error("--perturb must have the same token count as the last --prompt (" +
                                         std::to_string(alt.size()) + " vs " + std::to_string(toks.back().size()) + ")");
            }
            auto toks2 = toks;
            toks2.back() = alt;
            eng.clear();
            const auto bat2 = eng.generate_parallel(toks2, seqs, n_predict, true);
            std::vector<Answer> keep_ref(bat.begin(), bat.end() - 1);
            std::vector<Answer> keep_new(bat2.begin(), bat2.end() - 1);
            json p = compare_runs(keep_ref, keep_new, "batched", "batched_other_neighbour");
            double md_all = 0.0;
            for (size_t k = 0; k < keep_ref.size(); ++k) {
                const size_t steps = std::min(keep_ref[k].logits.size(), keep_new[k].logits.size());
                for (size_t i = 0; i < steps; ++i) {
                    md_all = std::max(md_all, max_abs_diff(keep_ref[k].logits[i], keep_new[k].logits[i]));
                }
            }
            p["replaced_prompt"] = prompts.back();
            p["replacement"] = perturb;
            p["neighbour_outputs"] = {bat.back().text, bat2.back().text};
            p["max_abs_logit_diff_all_steps"] = md_all;
            p["pass"] = p["pass"].get<bool>() && md_all == 0.0;
            out["isolation_same_shape"] = p;
        }
    }

    // case B: one shared prefix stored once with seq ids {0..n-1}, then a different suffix per sequence
    if (!shared.empty() && suffixes.size() >= 2) {
        auto pre = eng.tokenize(shared, true);
        std::vector<std::vector<llama_token>> suf;
        std::vector<llama_seq_id> seqs;
        for (size_t k = 0; k < suffixes.size(); ++k) {
            suf.push_back(eng.tokenize(suffixes[k], true));
            seqs.push_back((llama_seq_id) k);
        }
        std::vector<Answer> sep;
        int cells_separate = 0;
        for (size_t k = 0; k < suf.size(); ++k) {
            eng.clear();
            std::vector<llama_token> full = pre;
            full.insert(full.end(), suf[k].begin(), suf[k].end());
            sep.push_back(eng.generate_parallel({full}, {0}, n_predict, true)[0]);
            cells_separate += eng.used_cells();
        }
        eng.clear();
        std::vector<TokenEntry> pe;
        for (size_t i = 0; i < pre.size(); ++i) {
            pe.push_back({pre[i], (llama_pos) i, seqs, -1});
        }
        eng.decode(pe);
        const auto bat = eng.generate_parallel(suf, seqs, n_predict, true);
        json c = compare_runs(sep, bat);
        c["prefix_tokens"]   = pre.size();
        c["cells_separate"]  = cells_separate;
        c["cells_batched"]   = eng.used_cells();
        for (size_t k = 0; k < suffixes.size(); ++k) {
            c["sequences"][k]["suffix"] = suffixes[k];
        }
        out["shared_prefix"] = c;
    }

    const std::string path = a.get("out");
    if (!path.empty()) {
        std::ofstream(path) << out.dump(2) << std::endl;
    }
    for (const auto & key : {"independent_prompts", "isolation_same_shape", "shared_prefix"}) {
        if (!out.contains(key)) {
            continue;
        }
        printf("%s: %s\n", key, out[key]["pass"].get<bool>() ? "PASS" : "FAIL");
        for (const auto & s : out[key]["sequences"]) {
            for (const auto & [name, v] : s.items()) {
                if (v.is_object() && v.contains("text")) {
                    printf("  seq %d %-24s: %s\n", s["seq"].get<int>(), name.c_str(), v["text"].get<std::string>().c_str());
                }
            }
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    try {
        const Args a = parse_args(argc, argv);
        if (a.cmd == "explain")   return cmd_explain(a);
        if (a.cmd == "ask")       return cmd_ask(a);
        if (a.cmd == "run-tests") return cmd_run_tests(a);
        if (a.cmd == "batched")   return cmd_batched(a);
        throw std::runtime_error("unknown command " + a.cmd);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
