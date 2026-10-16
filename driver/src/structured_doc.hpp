// In-memory form of the shared JSON schema (schema/structured_doc.schema.json).
#pragma once

#include "llama.h"

#include <map>
#include <string>
#include <vector>

struct DocNode {
    std::string id;
    std::string node_type;
    std::string parent;                 // "" for the root
    std::vector<std::string> children;
    std::string unit;
    std::string title;                  // heading text / short label, may be empty
    std::string path;                   // "Title > Section > Subsection"
    int depth = 0;

    std::vector<llama_token>  tokens;
    std::vector<llama_pos>    positions; // one per token, contiguous
    std::vector<llama_seq_id> seq_ids;   // shared by every token of the node
};

struct DocUnit {
    std::string id;
    std::string kind;                    // root | subtree | merged_group
    std::string parent_unit;
    std::vector<llama_seq_id> seq_ids;
    llama_pos pos_start = 0;
    llama_pos pos_end   = 0;             // exclusive
    std::vector<std::string> nodes;
};

// One token of a batch: the token, its position, and every sequence it belongs to.
struct TokenEntry {
    llama_token tok;
    llama_pos   pos;
    std::vector<llama_seq_id> seqs;
    int node;                            // index into StructuredDoc::nodes, -1 for the prompt prefix
};

struct StructuredDoc {
    std::string path;
    std::string document_id;
    std::string domain;
    int n_seq = 0;

    std::vector<llama_token>  prefix_tokens;
    std::vector<llama_pos>    prefix_positions;
    std::vector<llama_seq_id> prefix_seq_ids;
    std::string question_template;       // contains "{question}"

    std::vector<DocNode> nodes;          // document (DFS) order
    std::vector<DocUnit> units;
    std::map<std::string, int> node_index;

    static StructuredDoc load(const std::string & path);

    // Throws std::runtime_error describing the first violated invariant.
    void validate(int n_seq_max_engine) const;

    // Every token with its structured positions / sequence ids, ordered by (pos, first seq).
    std::vector<TokenEntry> structured_entries() const;

    // Per-sequence token stream: stream[s][p] is the token sequence s holds at position p.
    std::vector<std::vector<llama_token>> seq_streams() const;

    // Prefix + all node tokens in document order, as one flat sequence (full attention baseline).
    std::vector<llama_token> flat_tokens() const;

    int n_doc_tokens() const;
    int find(const std::string & node_id) const;          // -1 if missing
    std::vector<int> subtree(int node) const;              // node + descendants
    llama_seq_id target_seq(int node) const;               // the sequence used to ask about a node
    std::string question_text(const std::string & question) const;
};
