# Structured document schema (v1.0)

The parsers write this format and the C++ driver reads it. It is the format used by this Markdown package; other parsers must emit it to use this driver.
The formal field types are in [structured_doc.schema.json](structured_doc.schema.json). The rules
that link fields together are checked by `python3 -m ldls.validate` and described below.

## The five required fields

| Field | Where | Meaning |
|---|---|---|
| `tokens` | every node | Token ids of the node's **own** text (children excluded), from the model's llama.cpp tokenizer |
| `positions` | every node | One position per token, contiguous. These are the `pos` values given to `llama_batch` |
| `seq_ids` | every node | The llama.cpp sequences this node's tokens belong to. Every token of the node shares them |
| `node_type` | every node | Domain-specific type: `heading`, `paragraph`, `list_item`, `code_block`, `table_row`, ... |
| `parent` | every node | Id of the parent node, or `null` for the root |

Other node fields: `id`, `children` (ordered), `depth`, `title`, `path`, `attrs` (for example the
heading level or code language), `unit`, `text` (the node's own source text), and `token_class`
(one character per token: `M` markup/syntax, `C` content, `W` whitespace).

Top level: `schema_version`, `document_id`, `domain`, `source` (path, sha256), `tokenizer`,
`prompt` (the chat prefix and the question template), `layout` (number of sequences, merge rules,
units), `nodes` (document order), and `stats` (precomputed statistics for the slide).

## A small real example

`schema/example/mini.md` (built with `--min-unit-tokens 4` so that tiny sections still split):

```
# Report            n1 heading    parent n0  positions 35-37  seq_ids [0,1]  unit u1   tokens [2, 8423, 271]
## Intro            n2 heading    parent n1  positions 38-40  seq_ids [0]    unit u2
This paper ...      n3 paragraph  parent n2  positions 41-46  seq_ids [0]    unit u2
## Methods          n4 heading    parent n1  positions 38-40  seq_ids [1]    unit u3
We use 12 samples.  n5 paragraph  parent n4  positions 41-46  seq_ids [1]    unit u3
```

Positions 0-34 hold the chat prefix in both sequences. "Intro" and "Methods" both use positions
38-46, but in different sequences, so they never see each other. "# Report" belongs to both
sequences. The full file is [example/mini.json](example/mini.json).

## Design decisions

These answer the five open questions of the plan.

1. **Positions are per sequence, not global.** The prefix takes `0..P-1`. The root unit starts at
   `P`. Each child unit starts right after its parent unit ends, so siblings share their start
   position. Editing B therefore never moves A's positions. With global positions, a longer B
   would shift every token after it, and RoPE would change A's attention scores even with a
   perfect mask.
2. **Sequences belong to leaf units, not to nodes.** A *unit* is a group of nodes laid out
   together: a section with its paragraphs, lists and small subsections. Every leaf unit owns
   exactly one sequence id. A token is visible from every leaf below its unit, so its `seq_ids`
   are the union of those leaves.
3. **Heading and child paragraphs** are in the same unit, in document order. The heading node
   holds the heading line; each paragraph or list is its own child node.
4. **No duplicated tokens.** A node stores only its own text. Concatenating `text` over `nodes` in
   order reproduces the source byte for byte; `source.sha256` is checked against it.
5. **Small subtrees are merged** (next section), so that the number of sequences stays small and
   no sequence is too short to be useful.

## Merge rules

Parameters (recorded in `layout.merge_rules`): `split_node_types` (Markdown: `heading`),
`min_unit_tokens` (default 40), `max_sequences` (default 32, hard limit 256 = `LLAMA_MAX_SEQ`).

1. Only a node of a split type can start a unit.
2. A split-type child whose **subtree** has at least `min_unit_tokens` tokens becomes a `subtree`
   unit, and the rules are applied again inside it.
3. Consecutive smaller split-type siblings are collected. If there are at least two and together
   they reach `min_unit_tokens`, they become one `merged_group` unit. Otherwise they are absorbed
   into the parent's unit.
4. Children of other types (paragraphs, lists, code, tables) always stay in the parent's unit,
   whatever their size.
5. If the leaf units would exceed `max_sequences`, `min_unit_tokens` is raised by 50% and the
   assignment is redone. The value that fits is stored as `effective_min_unit_tokens`.

A unit's own tokens always come before its child units. A small subsection that follows a large
one in the document is therefore placed before it inside the shared part. Such units are flagged
`"reordered": true`.

## Invariants (checked by the validator, required by llama.cpp)

* In every sequence, positions run `0..len-1` with exactly one token each. llama.cpp rejects a
  batch whose positions do not continue from the sequence's last cached position (`Y = X + 1`).
* Along every sequence, a token's set of sequences never grows. Shared tokens therefore always
  form a common prefix. llama.cpp builds the attention mask from `seq_id[i][0]` only (the first
  id of each token), so every member sequence gets the same view only if this holds.
* Tokens shared by several sequences are decoded while those sequences have equal cached position
  ranges ("coupled sequences"). This needs `kv_unified = true` in the context parameters.
* `n_seq <= max_sequences <= 256`; each leaf unit owns one sequence id; ids run `0..n_seq-1`.
* Document tokens stay under 2,000 (the week-1 limit). A larger document gets a warning.

## How the driver uses it

* **Prefill:** every token goes into one `llama_batch` with its `positions` and `seq_ids`, sorted
  by position.
* **Ask about node X:** the question template is appended to `seq_ids[0]` of X's unit, at
  `llama_memory_seq_pos_max(seq) + 1`. Afterwards it is removed again with `llama_memory_seq_rm`.
* **Evict subtree X:** for every sequence of the subtree, remove the subtree's position range
  `[min, max+1)` with `llama_memory_seq_rm(mem, seq, p0, p1)`. A cell is freed only when no
  sequence references it any more.
* **Change subtree B:** compare the old and new token stream of each sequence. Drop each changed
  sequence from its first differing position, and decode the new tokens. Sequences that did not
  change, including A's, are not touched.

## Adding a domain

Write `ldls/parsers/<domain>.py` with `parse(source_text) -> TreeNode`:

* fill `node_type`, `text`, and `byte_class` (`M`/`C`/`W` per UTF-8 byte of `text`);
* make sure the nodes' texts, concatenated in document order, reproduce the source;
* mark the node types that may become units with `splittable=True` (code: classes and functions;
  JSON: objects and arrays with keys; HTML: sections and divs, ...);
* register it in `ldls/parsers/__init__.py`.

Tokenization, layout, the merge rules, JSON output, validation and statistics are shared.
