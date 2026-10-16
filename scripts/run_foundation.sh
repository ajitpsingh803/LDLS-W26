#!/usr/bin/env bash
# Common foundation (all teams): smoke test, the stock batched example, and test 3
# (two prompts in two sequences vs separate runs). Logs go to results/foundation/.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/third_party/llama.cpp/build/bin"
MODEL="${MODEL:-$REPO/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
THREADS="${THREADS:-8}"
OUT="$REPO/results/foundation"
mkdir -p "$OUT"

echo "== 1. smoke test: llama-completion, greedy"
"$BIN/llama-completion" -m "$MODEL" -p "The capital of France is" -n 16 --temp 0 -t "$THREADS" -no-cnv \
    > "$OUT/smoke_completion.log" 2>&1
grep -E "^The capital|prompt eval time|eval time" "$OUT/smoke_completion.log"

echo "== 2a. stock examples/batched, prompt shared by 2 sequences, default (non-unified) KV cache"
"$BIN/llama-batched" -m "$MODEL" -p "The capital of France is" -np 2 -n 32 --temp 0 -t "$THREADS" \
    > "$OUT/llama_batched_default.log" 2>&1 || echo "Non-unified example failed; see the diagnostic log."
grep -E "coupled|failed" "$OUT/llama_batched_default.log" || echo "(ran without error)"

echo "== 2b. stock examples/batched with -kvu (unified KV cache)"
"$BIN/llama-batched" -m "$MODEL" -p "The capital of France is" -np 2 -n 32 --temp 0 -t "$THREADS" -kvu \
    > "$OUT/llama_batched_kvu.log" 2>&1
grep -A2 -E "^sequence [0-9]:" "$OUT/llama_batched_kvu.log"

echo "== 3. test 3: batched vs separate (ldls_driver batched)"
python3 "$REPO/tests/run_batched_test.py" --threads "$THREADS" | tee "$OUT/test3_batched_vs_separate.log"
