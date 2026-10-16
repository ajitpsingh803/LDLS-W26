// Thin wrapper around the llama.h C API: batched decode with multi-sequence tokens,
// greedy question answering inside one sequence, KV-cache edits and measurements.
#pragma once

#include "structured_doc.hpp"

#include "llama.h"

#include <string>
#include <vector>

struct EngineParams {
    std::string model_path;
    int  n_ctx      = 8192;
    int  n_seq_max  = 64;
    int  n_threads  = 8;
    int  n_batch    = 8192;   // one llama_decode call can take the whole document
    int  n_ubatch   = 512;
    bool kv_unified = true;   // required: tokens shared by several sequences live in one cell
    bool verbose    = false;
    bool trace_generation = false; // optional top-two evidence, excluded from timing runs
};

struct GenerationStep {
    int token, runner_up;
    float logit, runner_up_logit;
    bool end_of_generation;
};

struct Answer {
    llama_seq_id seq = 0;
    llama_pos    q_start = 0;                  // first position of the question in `seq`
    int          n_question = 0;
    std::vector<llama_token> tokens;           // generated tokens (EOG excluded)
    std::string  text;
    std::vector<GenerationStep> generation_trace;
    std::vector<std::vector<float>> logits;    // logits per generated step (only the first unless all requested)
    double t_question_ms = 0.0;                // decode of the question tokens
    double t_generate_ms = 0.0;                // token-by-token generation
};

class Engine {
public:
    explicit Engine(const EngineParams & p);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    std::vector<llama_token> tokenize(const std::string & text, bool parse_special) const;
    std::string detokenize(const std::vector<llama_token> & toks) const;

    // Empty the whole KV cache (all sequences).
    void clear();

    // Decode a list of tokens (any mix of positions / sequence sets) without producing logits.
    // Returns wall time in ms; throws on any llama_decode error.
    double decode(const std::vector<TokenEntry> & entries);

    // Append `question` to sequence `seq` (at pos_max(seq)+1) and greedily generate up to n_predict
    // tokens inside that sequence only. When `cleanup` is set, the question and the answer are
    // removed from the cache afterwards so the next question starts from the same state.
    Answer ask(llama_seq_id seq, const std::vector<llama_token> & question, int n_predict,
               bool cleanup = true, bool keep_all_logits = false);

    // Run several prompts at once, one sequence each, sampling greedily per sequence (batched.cpp style).
    // prompts[i] is decoded into seqs[i] starting at that sequence's pos_max+1.
    std::vector<Answer> generate_parallel(const std::vector<std::vector<llama_token>> & prompts,
                                          const std::vector<llama_seq_id> & seqs, int n_predict,
                                          bool keep_all_logits);

    bool      seq_rm(llama_seq_id s, llama_pos p0, llama_pos p1);
    llama_pos seq_pos_max(llama_seq_id s) const;

    // Number of occupied KV cells, read from llama.cpp's own serialized state header
    // (seq = -1: whole cache, otherwise the cells holding `seq`).
    int    used_cells(llama_seq_id seq = -1) const;
    size_t state_bytes(llama_seq_id seq = -1) const;

    const EngineParams & params() const { return p_; }
    int    n_vocab() const { return n_vocab_; }
    llama_token bos() const { return llama_vocab_bos(vocab_); }
    size_t kv_bytes_per_cell() const { return kv_bytes_per_cell_; }
    double load_ms() const { return t_load_ms_; }
    int    n_ctx() const;

private:
    EngineParams p_;
    llama_model *       model_ = nullptr;
    llama_context *     ctx_   = nullptr;
    const llama_vocab * vocab_ = nullptr;
    llama_memory_t      mem_   = nullptr;
    llama_batch         batch_ {};
    int    n_vocab_ = 0;
    size_t kv_bytes_per_cell_ = 0;
    double t_load_ms_ = 0.0;

    void fill_batch(const std::vector<TokenEntry> & entries, size_t off, size_t n, const std::vector<int> & outputs);
    int  run_batch();
};

// Greedy choice; ties go to the lowest token id, so it is deterministic.
llama_token argmax(const float * logits, int n);
double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b);
