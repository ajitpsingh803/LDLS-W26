#include "engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static bool g_verbose = false;

static void log_cb(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) {
        fputs(text, stderr);
    }
}

llama_token argmax(const float * logits, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (logits[i] > logits[best]) {
            best = i;
        }
    }
    return best;
}

double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, (double) std::fabs(a[i] - b[i]));
    }
    return m;
}

Engine::Engine(const EngineParams & p) : p_(p) {
    g_verbose = p.verbose;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();

    const double t0 = now_ms();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0; // CPU only
    model_ = llama_model_load_from_file(p.model_path.c_str(), mp);
    if (!model_) {
        throw std::runtime_error("failed to load model " + p.model_path);
    }
    vocab_   = llama_model_get_vocab(model_);
    n_vocab_ = llama_vocab_n_tokens(vocab_);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = p.n_ctx;
    cp.n_batch         = p.n_batch;
    cp.n_ubatch        = p.n_ubatch;
    cp.n_seq_max       = p.n_seq_max;
    cp.n_threads       = p.n_threads;
    cp.n_threads_batch = p.n_threads;
    cp.kv_unified      = p.kv_unified;
    cp.no_perf         = false;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        throw std::runtime_error("failed to create llama_context");
    }
    mem_ = llama_get_memory(ctx_);
    t_load_ms_ = now_ms() - t0;

    // K and V for every layer, F16 (the default cache type): n_layer * 2 * n_head_kv * head_dim * 2 bytes
    const int head_dim = llama_model_n_embd(model_) / llama_model_n_head(model_);
    kv_bytes_per_cell_ = (size_t) llama_model_n_layer(model_) * 2 * llama_model_n_head_kv(model_) * head_dim * 2;

    batch_ = llama_batch_init(p.n_batch, 0, p.n_seq_max);
}

Engine::~Engine() {
    llama_batch_free(batch_);
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_model_free(model_);
    }
    llama_backend_free();
}

int Engine::n_ctx() const {
    return (int) llama_n_ctx(ctx_);
}

std::vector<llama_token> Engine::tokenize(const std::string & text, bool parse_special) const {
    std::vector<llama_token> out(text.size() + 8);
    int n = llama_tokenize(vocab_, text.data(), (int32_t) text.size(), out.data(), (int32_t) out.size(), false, parse_special);
    if (n < 0) {
        out.resize(-n);
        n = llama_tokenize(vocab_, text.data(), (int32_t) text.size(), out.data(), (int32_t) out.size(), false, parse_special);
    }
    out.resize(n);
    return out;
}

std::string Engine::detokenize(const std::vector<llama_token> & toks) const {
    std::string out;
    char buf[256];
    for (auto t : toks) {
        int n = llama_token_to_piece(vocab_, t, buf, sizeof(buf), 0, false);
        if (n > 0) {
            out.append(buf, n);
        }
    }
    return out;
}

void Engine::clear() {
    llama_memory_clear(mem_, true);
}

void Engine::fill_batch(const std::vector<TokenEntry> & entries, size_t off, size_t n, const std::vector<int> & outputs) {
    batch_.n_tokens = (int32_t) n;
    for (size_t i = 0; i < n; ++i) {
        const auto & e = entries[off + i];
        if ((int) e.seqs.size() > p_.n_seq_max) {
            throw std::runtime_error("token belongs to more sequences than n_seq_max");
        }
        batch_.token[i]    = e.tok;
        batch_.pos[i]      = e.pos;
        batch_.n_seq_id[i] = (int32_t) e.seqs.size();
        for (size_t k = 0; k < e.seqs.size(); ++k) {
            batch_.seq_id[i][k] = e.seqs[k];
        }
        batch_.logits[i] = 0;
    }
    for (int o : outputs) {
        batch_.logits[o] = 1;
    }
}

int Engine::run_batch() {
    const int ret = llama_decode(ctx_, batch_);
    if (ret != 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(ret));
    }
    return ret;
}

double Engine::decode(const std::vector<TokenEntry> & entries) {
    const double t0 = now_ms();
    for (size_t off = 0; off < entries.size(); off += p_.n_batch) {
        const size_t n = std::min(entries.size() - off, (size_t) p_.n_batch);
        fill_batch(entries, off, n, {});
        run_batch();
    }
    llama_synchronize(ctx_);
    return now_ms() - t0;
}

Answer Engine::ask(llama_seq_id seq, const std::vector<llama_token> & question, int n_predict, bool cleanup, bool keep_all_logits) {
    Answer a;
    a.seq        = seq;
    a.q_start    = seq_pos_max(seq) + 1;
    a.n_question = (int) question.size();

    std::vector<TokenEntry> q;
    for (size_t i = 0; i < question.size(); ++i) {
        q.push_back({question[i], a.q_start + (llama_pos) i, {seq}, -2});
    }

    double t0 = now_ms();
    fill_batch(q, 0, q.size(), {(int) q.size() - 1});
    run_batch();
    a.t_question_ms = now_ms() - t0;

    t0 = now_ms();
    llama_pos pos = a.q_start + (llama_pos) question.size();
    for (int i = 0; i < n_predict; ++i) {
        const float * logits = llama_get_logits_ith(ctx_, -1);
        if (i == 0 || keep_all_logits) {
            a.logits.emplace_back(logits, logits + n_vocab_);
        }
        const llama_token t = argmax(logits, n_vocab_);
        if (p_.trace_generation) {
            int runner = t == 0 ? 1 : 0;
            for (int k = 0; k < n_vocab_; ++k) {
                if (k != t && logits[k] > logits[runner]) runner = k;
            }
            a.generation_trace.push_back({t, runner, logits[t], logits[runner],
                                         llama_vocab_is_eog(vocab_, t)});
        }
        if (llama_vocab_is_eog(vocab_, t)) {
            break;
        }
        a.tokens.push_back(t);
        if (i + 1 == n_predict) {
            break;
        }
        std::vector<TokenEntry> next = {{t, pos++, {seq}, -2}};
        fill_batch(next, 0, 1, {0});
        run_batch();
    }
    a.t_generate_ms = now_ms() - t0;
    a.text = detokenize(a.tokens);

    if (cleanup) {
        seq_rm(seq, a.q_start, -1);
    }
    return a;
}

std::vector<Answer> Engine::generate_parallel(const std::vector<std::vector<llama_token>> & prompts,
                                              const std::vector<llama_seq_id> & seqs, int n_predict, bool keep_all_logits) {
    const size_t n = prompts.size();
    std::vector<Answer> out(n);
    std::vector<TokenEntry> entries;
    std::vector<int> outputs;
    std::vector<int> i_batch(n);
    std::vector<llama_pos> pos(n);
    for (size_t k = 0; k < n; ++k) {
        out[k].seq        = seqs[k];
        out[k].q_start    = seq_pos_max(seqs[k]) + 1;
        out[k].n_question = (int) prompts[k].size();
        for (size_t i = 0; i < prompts[k].size(); ++i) {
            entries.push_back({prompts[k][i], out[k].q_start + (llama_pos) i, {seqs[k]}, -2});
        }
        i_batch[k] = (int) entries.size() - 1;
        outputs.push_back(i_batch[k]);
        pos[k] = out[k].q_start + (llama_pos) prompts[k].size();
    }

    double t0 = now_ms();
    fill_batch(entries, 0, entries.size(), outputs);
    run_batch();
    const double t_prompt = now_ms() - t0;

    t0 = now_ms();
    for (int step = 0; step < n_predict; ++step) {
        std::vector<TokenEntry> next;
        std::vector<int> next_out;
        for (size_t k = 0; k < n; ++k) {
            if (i_batch[k] < 0) {
                continue;
            }
            const float * logits = llama_get_logits_ith(ctx_, i_batch[k]);
            if (step == 0 || keep_all_logits) {
                out[k].logits.emplace_back(logits, logits + n_vocab_);
            }
            const llama_token t = argmax(logits, n_vocab_);
            if (llama_vocab_is_eog(vocab_, t)) {
                i_batch[k] = -1;
                continue;
            }
            out[k].tokens.push_back(t);
            if (step + 1 == n_predict) {
                i_batch[k] = -1;
                continue;
            }
            i_batch[k] = (int) next.size();
            next_out.push_back(i_batch[k]);
            next.push_back({t, pos[k]++, {seqs[k]}, -2});
        }
        if (next.empty()) {
            break;
        }
        fill_batch(next, 0, next.size(), next_out);
        run_batch();
    }
    const double t_gen = now_ms() - t0;
    for (auto & a : out) {
        a.text          = detokenize(a.tokens);
        a.t_question_ms = t_prompt;
        a.t_generate_ms = t_gen;
    }
    return out;
}

bool Engine::seq_rm(llama_seq_id s, llama_pos p0, llama_pos p1) {
    return llama_memory_seq_rm(mem_, s, p0, p1);
}

llama_pos Engine::seq_pos_max(llama_seq_id s) const {
    return llama_memory_seq_pos_max(mem_, s);
}

size_t Engine::state_bytes(llama_seq_id seq) const {
    return llama_state_seq_get_size(ctx_, seq);
}

int Engine::used_cells(llama_seq_id seq) const {
    // Layout written by llama_context::state_seq_get_data() + llama_kv_cache::state_write():
    //   uint32 io_magic (0xaf143cd8), int32 seq_id, uint32 n_stream, then per stream uint32 cell_count, ...
    static constexpr uint32_t io_magic = 0xaf143cd8;
    const size_t size = llama_state_seq_get_size(ctx_, seq);
    if (size < 4 * sizeof(uint32_t)) {
        return 0;
    }
    std::vector<uint8_t> buf(size);
    llama_state_seq_get_data(ctx_, buf.data(), size, seq);
    uint32_t magic = 0;
    uint32_t n_stream = 0;
    uint32_t cell_count = 0;
    std::memcpy(&magic, buf.data(), sizeof(uint32_t));
    if (magic != io_magic) {
        throw std::runtime_error("unexpected llama.cpp state header; update Engine::used_cells for this llama.cpp version");
    }
    std::memcpy(&n_stream, buf.data() + 8, sizeof(uint32_t));
    std::memcpy(&cell_count, buf.data() + 12, sizeof(uint32_t));
    if (n_stream != 1) {
        return -1; // only the unified cache layout (one stream) is parsed here
    }
    return (int) cell_count;
}
