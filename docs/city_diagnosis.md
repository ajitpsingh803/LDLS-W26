# City guide diagnosis

The city question asks how much a Maritime Museum ticket costs. The normal answer is "$12". With only A's path left in the cache, the answer becomes "$12.". The price is still correct, but the extra full stop makes the exact-token test fail. The remaining cache size is correct at 166 cells. The first two generated tokens match. At step 2, the normal run prefers stopping over a full stop by 0.208 logits. In the trimmed-cache run, the full stop wins by 0.036.

The model is close to a tie at this step, so a small change in the logits is enough to change the output. Floating-point differences after the cache layout changes are a possible cause. We have not isolated the exact operation responsible. The main T1 and T2 tests still pass for this document. We left the extra test marked as failed because it requires identical tokens.

Evidence: `results/diagnostics/05_city_guide.json`. The diagnostic flag is optional and does not change sampling, prompts, sequence IDs or pass conditions. The ordinary timing run does not enable it.
