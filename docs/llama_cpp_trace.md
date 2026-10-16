# Tracing llama.cpp: batches, sequence ids and the attention mask

All references are to llama.cpp commit `318a471` (pinned in `scripts/setup_llama_cpp.sh`).
Paths are relative to `third_party/llama.cpp/`.

## 1. Build and first run (CPU only)

```bash
scripts/setup_llama_cpp.sh      # clone at the pinned commit, cmake -DGGML_CUDA=OFF ..., build tools
scripts/download_model.sh       # models/Llama-3.2-1B-Instruct-Q4_K_M.gguf (0.8 GB)
third_party/llama.cpp/build/bin/llama-completion -m models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
    -p "The capital of France is" -n 16 --temp 0 -t 8 -no-cnv
```

**AMX pitfall reported in the original project.** The Xeon 8452Y reports AMX, so a default `-march=native`
build uses ggml's AMX matmul kernels. On this Linux 5.15 kernel they crash with `SIGILL` on
`tilezero` once a prompt is a few hundred tokens long. Short prompts happened to work, which hid
the problem in the first smoke test. gdb backtrace:
`tinygemm_kernel_amx<block_q8_K, block_q4_K>` → `ggml_backend_amx_mul_mat`. Both build scripts
therefore compile ggml-cpu without AMX, keeping AVX-512/VNNI and the repacked Q4_K kernels
(`CPU_REPACK` buffer).

The AMX observation above is retained from the supplied implementation. The current local rerun uses an E5-2620 v4 without AMX and does not reproduce that hardware-specific crash.

## 2. The parts of `include/llama.h` that matter

| What | Where | Notes |
|---|---|---|
| `struct llama_batch` | llama.h:264 | `token[i]`, `pos[i]`, `n_seq_id[i]`, `seq_id[i][..]`, `logits[i]` (output flag) |
| `llama_batch_init(n_tokens, embd, n_seq_max)` | llama.h:976 | `n_seq_max` = how many sequence ids **one token** may carry |
| `llama_decode(ctx, batch)` | llama.h:1006 | 0 ok, 1 no KV slot, -1 invalid batch (e.g. position rules) |
| `llama_context_params.n_seq_max` | llama.h:370 | number of distinct sequences, at most `LLAMA_MAX_SEQ` = 256 (src/llama-cparams.h:8) |
| `llama_context_params.kv_unified` | llama.h:415 | one KV buffer for all sequences. **Needed** as soon as a token belongs to more than one sequence |
| `llama_memory_seq_rm(mem, seq, p0, p1)` | llama.h:764 | removes `seq` from cells with position in `[p0, p1)`. `seq < 0` matches every sequence |
| `llama_memory_seq_cp / _keep / _add / _div` | llama.h:773-803 | copy, keep only, shift, divide positions |
| `llama_memory_seq_pos_min / _max` | llama.h:809-818 | the next token of a sequence must use `pos_max + 1` |
| `llama_state_seq_get_size / _get_data` | llama.h:885-894 | serialized cells; the driver reads the cell count from this header |
| `llama_get_logits_ith(ctx, i)` | llama.h:~1144 | logits of batch entry `i` (only entries with `logits[i] = 1`) |

This checkout also has a newer **extended batch API** (`llama_batch_ext_*`, llama.h:1014-1090) and
`llama_process()` (llama.h:1091), which the examples now use through `common_batch`. The classic
`llama_batch` is converted by `llama_batch_compat` (`llama_context::decode(const llama_batch&)`,
src/llama-context.cpp:4606). The driver uses the classic struct because it is what `llama.h`
documents and what the assignment asks us to read.

## 3. `examples/batched/batched.cpp`, step by step

1. Tokenize the prompt (`common_tokenize`, line 60). Size the context so that the prompt is
   stored once and every stream has room for its own continuation:
   `n_kv_req = n_prompt + (n_predict - n_prompt) * n_parallel` (line 62).
2. One sampler chain per stream (top-k, top-p, temperature, dist; lines 76-85).
3. Build the prompt batch. Every prompt token gets **all** sequence ids `{0..n_parallel-1}`
   (`seq_ids[i] = i`, line 124; `batch.add(tokens_list[i], i, seq_ids, false)`, line 129). The
   prompt is therefore stored **once** and shared by every stream. The commented-out
   `llama_kv_cache_seq_cp` loop below it is the older way of doing the same thing.
4. Only the last prompt token produces logits (`set_output`, line 149), then `llama_process(...DECODE)` (line 151).
5. Generation loop. For each live stream, sample from that stream's logits row
   (`llama_sampler_sample(..., i_batch[i])`, line 191), then add the new token with **one** sequence
   id `i` at position `n_cur` (`batch.add(new_token_id, n_cur, i, true)`, line 214). All streams'
   tokens go into one batch per step (line 227).

From this point on, the streams share the prompt cells but never see each other's generated
tokens. This is exactly what our document trees need: a shared ancestor prefix, then private branches.

**What we observed running it** (`scripts/run_foundation.sh`, logs in `results/foundation/`):

```
llama-batched -p "The capital of France is" -np 2 --temp 0          # default context
  split_equal: sequential split is not supported when there are coupled sequences in the
  input batch (you may need to use the -kvu flag)        -> llama_decode() failed
llama-batched ... -kvu                                    # unified KV cache
  sequence 0: The capital of France is Paris. The capital of France is Paris. ...
  sequence 1: The capital of France is Paris. The capital of France is Paris. ...
```

## 4. What happens inside `llama_decode`

```
llama_decode(ctx, batch)                                   src/llama.cpp  (C API)
 └ llama_context::decode(llama_batch)        → compat      src/llama-context.cpp:4606
   └ llama_context::decode(llama_batch_ext)                src/llama-context.cpp:1821
     ├ balloc->init(batch, vocab, ...)        validation    src/llama-batch.cpp  (section 5)
     ├ memory->init_batch(balloc, n_ubatch)                 src/llama-kv-cache.cpp:724
     │   split into ubatches: split_simple when there is one stream (kv_unified),
     │   split_equal otherwise, which refuses coupled sequences  (line 733)
     ├ prepare / find_slot: choose free cells               src/llama-kv-cache.cpp:771, 918
     ├ apply_ubatch: write pos + every seq id of the token  src/llama-kv-cache.cpp:1117
     │   into the cell (cells.seq_add, line 1177)
     └ build graph, set inputs, compute
        └ llm_graph_input_attn_kv::set_input                src/llama-graph.cpp:525
           └ llama_kv_cache::set_input_kq_mask              src/llama-kv-cache.cpp:1761
              └ set_input_kq_mask_impl                      src/llama-kv-cache.cpp:1572
```

A KV **cell** stores one token's K/V, its position and a bitset of sequence ids
(`src/llama-kv-cells.h`, `seq_set_t = std::bitset<LLAMA_MAX_SEQ>`). A token decoded with seq ids
`{0,1,2}` occupies **one** cell that is a member of three sequences.

## 5. Batch validation (`src/llama-batch.cpp`)

* **Coupled sequences** (lines 249-269): sequences that share a token in the batch. They must have
  identical cached position ranges at decode time (lines 428-440), and they need `kv_unified`
  (line 621, the error above).
* **Consecutive positions** (lines 395-425): if sequence `s` already has cells, its first new
  token must be at `pos_max(s) + 1` (`Y = X + 1`), and the positions inside the batch must have no
  gaps. Our first draft schema, with positions restarting at 0 for every node, would be rejected
  here.
* **Compatible sequence sets** (lines 459-489): along a sequence, the intersection of the tokens'
  sequence sets must stay non-empty, and positions must not decrease. Nested sets (`{0,1,2}`, then
  `{0,1}`, then `{0}`) are fine; that is the shape of a root-to-leaf path.

## 6. The mask (`set_input_kq_mask_impl`, src/llama-kv-cache.cpp:1572-1720)

For every token `i` of the ubatch (a row of the mask) and every cell `j` (a column):

```cpp
const llama_seq_id seq_id = ubatch->seq_id[i][0];        // line 1607: FIRST seq id only
...
if (cells.is_empty(j))            goto skip;              // line 1661
if (!cells.seq_has(j, seq_id))    goto skip;              // line 1666: other sequence -> -inf
p0 = cells.pos_get(j);
if (causal && p0 > p1)            goto skip;              // line 1683: future -> -inf
data[idst + j] = 0.0f;  ...  skip: data[idst + j] = -INFINITY;
```

So **token i may attend to cell j iff cell j is a member of seq_id[i][0] and pos_j ≤ pos_i.**
Points that matter for the project:

1. **Only the first sequence id of the query token counts.** A token that belongs to `{0,1,2}`
   attends as sequence 0. This is exact only if sequences 0, 1 and 2 have the same history up to
   that token, i.e. shared tokens form a common prefix. Our layout guarantees it, and
   `ldls/validate.py` checks it ("sequence sets never grow along a sequence").
2. **Membership is per cell.** A cell shared by `{0,1,2}` is visible to all three. After
   `llama_memory_seq_rm(mem, 2, ...)` it remains visible to 0 and 1.
3. **Masking is by sequence id plus position, not by cell order.** Sibling subtrees can reuse the
   same positions in different sequences. Removed cells become holes that the next tokens may fill.
4. **Row reuse** (lines 1620-1660): rows of the same sequence share most of the mask, so
   llama.cpp copies the previous row and only recomputes cells near the batch positions. This
   does not change the rule.
5. The mask spans `n_kv` cells, padded to 256 (`get_n_kv`, line 1282). Masked cells still occupy
   columns, but the CPU flash-attention kernel skips `-inf` entries
   (`ggml/src/ggml-cpu/ops.cpp:8968`, `if (mv == -INFINITY) continue;`). In our runs, decoding a
   question inside one sequence took 113 ms vs 202 ms over the whole document, and generation took
   25.3 vs 26.4 ms per token (medians over the 5 documents). Prefill cost is about the same in both
   modes (1.34 s vs 1.38 s): every token is still computed once, and the matmuls dominate.

## 7. Removing tokens (`llama_kv_cache::seq_rm`, src/llama-kv-cache.cpp:404)

For each cell whose position is in `[p0, p1)` and that holds `seq_id`, the id is cleared
(`llama_kv_cells::seq_rm`, src/llama-kv-cells.h:244). **A cell is freed only when its last sequence
id is removed** (`seq[i].none()`: position set to -1, cell erased from `used`). Consequences:

* Evicting a leaf subtree frees exactly the cells that only that subtree's sequences reference.
  Shared ancestor cells stay.
* `seq_rm(-1, p0, p1)` would hit **every** sequence in that position range, including siblings
  that reuse the same positions. Always remove per sequence.
* Freed memory is *capacity*, not process RSS: the KV buffer is allocated once for `n_ctx` cells.
  We report occupied cells read from llama.cpp's own state
  (`llama_state_seq_get_data` header: `io_magic`, `seq_id`, `n_stream`, `cell_count`;
  src/llama-context.cpp:3400-3430, src/llama-kv-cache.cpp:2073).

## 8. Experiments that confirm the trace

| Claim | Evidence (`results/`) |
|---|---|
| A sequence never sees another sequence's tokens | `batched_vs_separate.json` → `isolation_same_shape`: change seq 1's prompt, keep the batch shape: seq 0 logits identical to the bit over 32 steps |
| A shared prefix stored once behaves like two private copies | `shared_prefix`: separate and batched outputs identical, logit difference 0, 56 vs 72 cells |
| Batched ≠ separate is possible without leakage | `independent_prompts`: seq 0 flips at step 3 where the top-2 margin is 0.08 logits. Different batch shapes use different CPU kernels and move logits by up to 0.5 |
| Changing subtree B cannot affect A | `markdown/*.json` → `test1_modify_b`: only B's sequences re-decoded, A's first-step logits identical (Δ = 0) |
| Evicting C frees exactly C's cells | `test2_evict`: `cells_freed == cells_expected_freed` for all 5 documents |
