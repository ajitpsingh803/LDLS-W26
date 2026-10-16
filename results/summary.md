# Team Markdown: results (markdown)

Model `Llama-3.2-1B-Instruct-Q4_K_M.gguf`, CPU `Intel(R) Xeon(R) CPU E5-2620 v4 @ 2.10GHz`, 8 threads, llama.cpp `318a471`, greedy decoding, unified KV cache.

## Tree statistics

| Document | Tokens | Nodes | Depth | Units | Seqs | Section subtree tokens med (min-max) | Leaf unit tokens med (min-max) | Markup | Markup + whitespace | Avg. visible / flat |
|---|---|---|---|---|---|---|---|---|---|---|
| 01_research_paper | 790 | 38 | 5 | 15 | 10 | 77 (35-790) | 63.5 (49-116) | 6.1% | 14.6% | 20.0% |
| 02_software_readme | 547 | 36 | 4 | 10 | 6 | 62.5 (24-547) | 84 (40-129) | 16.6% | 24.3% | 27.6% |
| 03_lab_protocol | 565 | 41 | 5 | 10 | 7 | 78.5 (35-565) | 77 (40-92) | 12.6% | 20.7% | 30.9% |
| 04_meeting_notes | 504 | 43 | 5 | 9 | 6 | 48 (26-504) | 53.5 (40-107) | 9.5% | 19.4% | 38.7% |
| 05_city_guide | 464 | 31 | 4 | 10 | 6 | 48 (21-464) | 52 (46-78) | 9.5% | 16.2% | 31.4% |

## Tests

| Document | T1 change B, ask A | T2 evict C | Cells freed | Keep only A's path | Control: change A | Flat + B distractor |
|---|---|---|---|---|---|---|
| 01_research_paper | PASS (dlogit 0) | PASS (dlogit 0.45) | 95 of 825 (3.0 MiB) | PASS, 157 cells | PASS | answer flipped |
| 02_software_readme | PASS (dlogit 0) | PASS (dlogit 0.49) | 129 of 582 (4.0 MiB) | PASS, 178 cells | PASS | answer held |
| 03_lab_protocol | PASS (dlogit 0) | PASS (dlogit 0.5) | 92 of 600 (2.9 MiB) | PASS, 198 cells | PASS | answer held |
| 04_meeting_notes | PASS (dlogit 0) | PASS (dlogit 0.58) | 48 of 539 (1.5 MiB) | PASS, 275 cells | PASS | answer held |
| 05_city_guide | PASS (dlogit 0) | PASS (dlogit 0.69) | 55 of 499 (1.7 MiB) | FAIL, 166 cells | PASS | answer flipped |

Answers:

* **01_research_paper**: How many documents were used in the evaluation? structured `1,240`; after B edit `1,240`; after C eviction `1,240`; after A edit `2,750`; flat `1,240`; flat with B distractor `3,500 documents.`
* **02_software_readme**: What is the current stable release version of quillsync? structured `The current stable release version of quillsync is 2.4.1.`; after B edit `The current stable release version of quillsync is 2.4.1.`; after C eviction `The current stable release version of quillsync is 2.4.1.`; after A edit `The current stable release version of quillsync is 2.6.0.`; flat `The current stable release version of quillsync is 2.4.1, published on 3 March 2026.`; flat with B distractor `The current stable release is version 2.4.1, published on 3 March 2026.`
* **03_lab_protocol**: How long does the heat shock last? structured `45 seconds.`; after B edit `45 seconds.`; after C eviction `45 seconds.`; after A edit `30 seconds.`; flat `The heat shock lasts for exactly **45 seconds**.`; flat with B distractor `The heat shock lasts for exactly **45 seconds**.`
* **04_meeting_notes**: On which date is the cut-over window for the database migration? structured `Saturday 14 November, 02:00 to 04:00 UTC.`; after B edit `Saturday 14 November, 02:00 to 04:00 UTC.`; after C eviction `Saturday 14 November, 02:00 to 04:00 UTC.`; after A edit `Saturday 21 November, 02:00 to 04:00 UTC.`; flat `Saturday 14 November, 02:00 to 04:00 UTC.`; flat with B distractor `Saturday 14 November, 02:00 to 04:00 UTC.`
* **05_city_guide**: How much does a ticket to the Maritime Museum cost? structured `$12`; after B edit `$12`; after C eviction `$12`; after A edit `$15.`; flat `$12.`; flat with B distractor `The ticket to the Maritime Museum costs 20 euros.`

## Test 3: batched vs separate

* Independent prompts: 0/2 sequences token-identical.
  Sequence 0 diverges at step 3 on a near-tie (top-2 margin 0.161 separate vs 0.227 batched; max logit deviation before divergence 0.52).
  Sequence 1 diverges at step 24 on a near-tie (top-2 margin 0.038 separate vs 0.018 batched; max logit deviation before divergence 0.70).
* Same batch shape, other neighbour prompt: max logit difference 0.0 over 32 steps -> PASS (no cross-sequence effect observed in this test).
* Shared prefix stored once (seq ids {0,1}): PASS, 56 KV cells vs 72 for separate runs.

## CPU run times (median over documents, ms)

| Step | ms |
|---|---|
| Model load (once) | 1630.8 |
| Prefill, flat | 4919.1 |
| Prefill, structured | 4952.3 |
| Question decode, flat (sees whole doc) | 326.3 |
| Question decode, structured (sees A's path) | 249.3 |
| Generation per token, flat | 41.6 |
| Generation per token, structured | 38.6 |
| Re-prefill after editing B | 4999.4 |
| Patch only B's sequences | 483.8 |
| Evict subtree C (seq_rm) | 50 µs |

Section subtree tokens include the H1 heading, whose subtree is the whole document; leaf units are
the subtrees that own one sequence each. Markup = tokens whose non-whitespace bytes are all syntax.
