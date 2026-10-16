# Tree-Masked Attention for Long Structured Documents

*Technical report, Efficient Systems Group.*

## Abstract

Large language models usually read a structured document as one flat sequence of tokens.
We study **tree-masked attention**, where each token may only attend to its own subtree and
to the ancestors of that subtree. On question answering over structured documents, the
method keeps answer accuracy within one point of full attention while reducing the
key-value cache by up to 58%.

## 1. Introduction

Web pages, papers and configuration files are trees, but inference engines flatten them.
Flattening wastes memory, because a question usually concerns a single subtree. It also
couples unrelated parts of the document: an edit to one section changes the cached state
of every token that comes after it.

Our contributions are:

1. A mapping from document trees to sequence identifiers of an inference engine.
2. A merge rule that keeps the number of sequences below the engine limit.
3. An evaluation of accuracy, memory and latency on four document domains.

## 2. Related Work

Sparse attention patterns such as sliding windows and global tokens were proposed for long
inputs. Retrieval-augmented generation selects passages before prompting, but it does not
reuse the cache when a document changes. KV-cache compression methods quantize or
low-rank-approximate the cache entries of every token. Our method is complementary: it
decides *which* tokens are cached for a question, and can be combined with compression.

## 3. Method

### 3.1 Tree construction

A domain parser converts each document into a tree. Every node stores its own tokens, its
type, and a pointer to its parent. Subtrees with fewer than 40 tokens are merged into a
neighbouring subtree.

### 3.2 Sequence assignment

Every leaf subtree receives its own sequence identifier. A token belongs to all sequences
of the leaves below it, so a section heading is shared by all of its subsections. Positions
restart after the shared prefix, which means sibling subtrees occupy the same position
range without seeing each other.

## 4. Experimental Setup

### 4.1 Datasets

We evaluate on 1,240 documents drawn from four domains: Markdown manuals, HTML product
pages, LaTeX papers and JSON API responses. Each document has between 600 and 1,900
tokens and comes with three factual questions written by annotators.

### 4.2 Hardware and software

All runs use a single 32-core CPU server with 256 GB of memory. The inference engine is
compiled without GPU support, and every experiment uses greedy decoding with a fixed
thread count of 16.

## 5. Results

### 5.1 Accuracy

| Method | Exact match | F1 |
|---|---|---|
| Full attention | 71.4 | 80.2 |
| Tree-masked attention | 70.6 | 79.5 |
| Retrieval, top-3 passages | 64.9 | 74.8 |

Tree-masked attention loses 0.8 points of exact match compared with full attention.

### 5.2 Memory and latency

When only the subtree that answers the question is kept, the cache shrinks by 58% on
average. Prefill time drops by 31% because masked tokens of unrelated subtrees are never
re-encoded after an edit.

## 6. Limitations

The method assumes that the answer lives in one subtree. Questions that compare two
distant sections need both sections in a single sequence. Documents shaped like grids,
such as spreadsheets, only approximate a tree.

## 7. Conclusion

Tree-masked attention is a simple way to make inference engines aware of document
structure. It needs no training and no changes to model weights.
