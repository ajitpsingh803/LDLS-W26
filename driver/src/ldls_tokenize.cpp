// ldls_tokenize: tokenizes strings with the model's own (llama.cpp) tokenizer.
//
// Used by the Python parsers so that the token ids written into the JSON documents are
// exactly the ids the C++ driver would produce. Only the vocabulary is loaded.
//
// stdin : {"texts": ["...", ...], "parse_special": false}
// stdout: {"tokens": [[...], ...], "piece_bytes": [[...], ...], "vocab_size": N, "bos": id, "eos": id}
//
// piece_bytes[i][j] is the number of bytes of token j of text i once detokenized; the
// parser uses it to map tokens back to byte offsets of the source text.

#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool parse_special) {
    std::vector<llama_token> out(text.size() + 8);
    int n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), out.data(), (int32_t) out.size(),
                           /*add_special*/ false, parse_special);
    if (n < 0) {
        out.resize(-n);
        n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), out.data(), (int32_t) out.size(), false, parse_special);
    }
    out.resize(n);
    return out;
}

static std::string piece(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, /*special*/ true);
    if (n < 0) {
        std::string big(-n, '\0');
        n = llama_token_to_piece(vocab, t, big.data(), (int32_t) big.size(), 0, true);
        big.resize(n);
        return big;
    }
    return std::string(buf, n);
}

int main(int argc, char ** argv) {
    std::string model_path;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) {
            model_path = argv[++i];
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf < request.json > response.json\n", argv[0]);
        return 1;
    }

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load vocabulary from %s\n", model_path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string input((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    json req = json::parse(input);
    const bool parse_special = req.value("parse_special", false);

    json res;
    res["vocab_size"] = llama_vocab_n_tokens(vocab);
    res["bos"] = llama_vocab_bos(vocab);
    res["eos"] = llama_vocab_eos(vocab);
    res["eot"] = llama_vocab_eot(vocab);
    json all_tokens = json::array();
    json all_pieces = json::array();
    for (const auto & t : req.at("texts")) {
        const std::string text = t.get<std::string>();
        auto toks = tokenize(vocab, text, parse_special);
        json pb = json::array();
        for (auto id : toks) {
            pb.push_back(piece(vocab, id).size());
        }
        all_tokens.push_back(toks);
        all_pieces.push_back(pb);
    }
    res["tokens"] = all_tokens;
    res["piece_bytes"] = all_pieces;
    std::cout << res.dump() << std::endl;

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
