#include "structured_doc.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>

using json = nlohmann::json;

static constexpr int LDLS_LLAMA_MAX_SEQ = 256; // LLAMA_MAX_SEQ in src/llama-cparams.h of the pinned llama.cpp

template <typename T>
static std::vector<T> vec(const json & j, const char * key) {
    std::vector<T> out;
    if (j.contains(key) && !j.at(key).is_null()) {
        for (const auto & v : j.at(key)) {
            out.push_back(v.get<T>());
        }
    }
    return out;
}

StructuredDoc StructuredDoc::load(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    json j = json::parse(f);

    StructuredDoc d;
    d.path        = path;
    d.document_id = j.value("document_id", "");
    d.domain      = j.value("domain", "");

    const json & layout = j.at("layout");
    d.n_seq = layout.at("n_seq").get<int>();
    for (const auto & u : layout.at("units")) {
        DocUnit du;
        du.id          = u.at("id").get<std::string>();
        du.kind        = u.value("kind", "");
        du.parent_unit = u.at("parent_unit").is_null() ? "" : u.at("parent_unit").get<std::string>();
        du.seq_ids     = vec<llama_seq_id>(u, "seq_ids");
        du.pos_start   = u.at("pos_start").get<llama_pos>();
        du.pos_end     = u.at("pos_end").get<llama_pos>();
        du.nodes       = vec<std::string>(u, "nodes");
        d.units.push_back(du);
    }

    const json & prompt = j.at("prompt");
    d.prefix_tokens     = vec<llama_token>(prompt.at("prefix"), "tokens");
    d.prefix_positions  = vec<llama_pos>(prompt.at("prefix"), "positions");
    d.prefix_seq_ids    = vec<llama_seq_id>(prompt.at("prefix"), "seq_ids");
    d.question_template = prompt.at("question_template").get<std::string>();

    for (const auto & n : j.at("nodes")) {
        DocNode dn;
        dn.id        = n.at("id").get<std::string>();
        dn.node_type = n.at("node_type").get<std::string>();
        dn.parent    = n.at("parent").is_null() ? "" : n.at("parent").get<std::string>();
        dn.children  = vec<std::string>(n, "children");
        dn.unit      = n.value("unit", "");
        dn.title     = n.value("title", "");
        dn.path      = n.value("path", "");
        dn.depth     = n.value("depth", 0);
        dn.tokens    = vec<llama_token>(n, "tokens");
        dn.positions = vec<llama_pos>(n, "positions");
        dn.seq_ids   = vec<llama_seq_id>(n, "seq_ids");
        d.node_index[dn.id] = (int) d.nodes.size();
        d.nodes.push_back(std::move(dn));
    }
    return d;
}

void StructuredDoc::validate(int n_seq_max_engine) const {
    auto fail = [&](const std::string & msg) { throw std::runtime_error(path + ": " + msg); };

    if (n_seq < 1 || n_seq > LDLS_LLAMA_MAX_SEQ) {
        fail("n_seq out of range: " + std::to_string(n_seq));
    }
    if (n_seq_max_engine > 0 && n_seq > n_seq_max_engine) {
        fail("document needs " + std::to_string(n_seq) + " sequences but the context allows " + std::to_string(n_seq_max_engine));
    }
    if (prefix_tokens.size() != prefix_positions.size()) {
        fail("prefix tokens/positions length mismatch");
    }
    for (size_t i = 0; i < prefix_positions.size(); ++i) {
        if (prefix_positions[i] != (llama_pos) i) {
            fail("prefix positions must be 0..P-1");
        }
    }
    if ((int) prefix_seq_ids.size() != n_seq) {
        fail("prefix must belong to every sequence");
    }
    if (question_template.find("{question}") == std::string::npos) {
        fail("question_template lacks {question}");
    }

    int n_roots = 0;
    for (const auto & n : nodes) {
        if (n.tokens.size() != n.positions.size()) {
            fail("node " + n.id + ": tokens/positions length mismatch");
        }
        for (size_t i = 1; i < n.positions.size(); ++i) {
            if (n.positions[i] != n.positions[i - 1] + 1) {
                fail("node " + n.id + ": positions are not contiguous");
            }
        }
        if (!n.tokens.empty() && n.seq_ids.empty()) {
            fail("node " + n.id + ": has tokens but no seq_ids");
        }
        for (size_t i = 0; i < n.seq_ids.size(); ++i) {
            if (n.seq_ids[i] < 0 || n.seq_ids[i] >= n_seq) {
                fail("node " + n.id + ": seq id out of range");
            }
            if (i > 0 && n.seq_ids[i] <= n.seq_ids[i - 1]) {
                fail("node " + n.id + ": seq_ids must be sorted and unique");
            }
        }
        if (n.parent.empty()) {
            ++n_roots;
        } else {
            const int p = find(n.parent);
            if (p < 0) {
                fail("node " + n.id + ": unknown parent " + n.parent);
            }
            const auto & pc = nodes[p].children;
            if (std::find(pc.begin(), pc.end(), n.id) == pc.end()) {
                fail("node " + n.id + ": parent does not list it as a child");
            }
        }
    }
    if (n_roots != 1) {
        fail("expected exactly one root node, found " + std::to_string(n_roots));
    }

    // Per sequence: every position 0..len-1 holds exactly one token, and the sequence sets of
    // the tokens never grow along the stream. The second rule is what makes llama.cpp's mask
    // (which looks only at seq_id[i][0]) give every member sequence of a token the same view.
    std::vector<std::map<llama_pos, std::vector<llama_seq_id>>> sets(n_seq);
    for (size_t i = 0; i < prefix_tokens.size(); ++i) {
        for (auto s : prefix_seq_ids) {
            sets[s][prefix_positions[i]] = prefix_seq_ids;
        }
    }
    for (const auto & n : nodes) {
        for (size_t i = 0; i < n.tokens.size(); ++i) {
            for (auto s : n.seq_ids) {
                if (!sets[s].emplace(n.positions[i], n.seq_ids).second) {
                    fail("sequence " + std::to_string(s) + " has two tokens at position " + std::to_string(n.positions[i]));
                }
            }
        }
    }
    for (int s = 0; s < n_seq; ++s) {
        llama_pos expect = 0;
        const std::vector<llama_seq_id> * prev = nullptr;
        for (const auto & [p, set] : sets[s]) {
            if (p != expect++) {
                fail("sequence " + std::to_string(s) + " has a gap before position " + std::to_string(p));
            }
            if (prev && !std::includes(prev->begin(), prev->end(), set.begin(), set.end())) {
                fail("sequence " + std::to_string(s) + ": sequence set grows at position " + std::to_string(p));
            }
            prev = &set;
        }
    }
}

std::vector<TokenEntry> StructuredDoc::structured_entries() const {
    std::vector<TokenEntry> out;
    for (size_t i = 0; i < prefix_tokens.size(); ++i) {
        out.push_back({prefix_tokens[i], prefix_positions[i], prefix_seq_ids, -1});
    }
    for (size_t k = 0; k < nodes.size(); ++k) {
        const auto & n = nodes[k];
        for (size_t i = 0; i < n.tokens.size(); ++i) {
            out.push_back({n.tokens[i], n.positions[i], n.seq_ids, (int) k});
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const TokenEntry & a, const TokenEntry & b) {
        if (a.pos != b.pos) {
            return a.pos < b.pos;
        }
        return a.seqs.front() < b.seqs.front();
    });
    return out;
}

std::vector<std::vector<llama_token>> StructuredDoc::seq_streams() const {
    std::vector<std::vector<llama_token>> st(n_seq);
    for (const auto & e : structured_entries()) {
        for (auto s : e.seqs) {
            if ((llama_pos) st[s].size() != e.pos) {
                throw std::runtime_error("seq_streams: non-contiguous stream (run validate first)");
            }
            st[s].push_back(e.tok);
        }
    }
    return st;
}

std::vector<llama_token> StructuredDoc::flat_tokens() const {
    std::vector<llama_token> out = prefix_tokens;
    for (const auto & n : nodes) {
        out.insert(out.end(), n.tokens.begin(), n.tokens.end());
    }
    return out;
}

int StructuredDoc::n_doc_tokens() const {
    int n = 0;
    for (const auto & node : nodes) {
        n += (int) node.tokens.size();
    }
    return n;
}

int StructuredDoc::find(const std::string & node_id) const {
    auto it = node_index.find(node_id);
    return it == node_index.end() ? -1 : it->second;
}

std::vector<int> StructuredDoc::subtree(int node) const {
    std::vector<int> out;
    std::vector<int> stack = {node};
    while (!stack.empty()) {
        int k = stack.back();
        stack.pop_back();
        out.push_back(k);
        const auto & ch = nodes[k].children;
        for (auto it = ch.rbegin(); it != ch.rend(); ++it) {
            stack.push_back(find(*it));
        }
    }
    return out;
}

llama_seq_id StructuredDoc::target_seq(int node) const {
    // a node without own tokens (e.g. an empty root) inherits the first sequence below it
    for (int k : subtree(node)) {
        if (!nodes[k].seq_ids.empty()) {
            return nodes[k].seq_ids.front();
        }
    }
    return 0;
}

std::string StructuredDoc::question_text(const std::string & question) const {
    std::string out = question_template;
    out.replace(out.find("{question}"), std::string("{question}").size(), question);
    return out;
}
