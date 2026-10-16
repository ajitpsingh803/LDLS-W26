#!/usr/bin/env bash
# Build the shared driver (build/bin/ldls_driver) and the tokenizer helper (build/bin/ldls_tokenize).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
JOBS="${JOBS:-16}"
cmake -S "$REPO" -B "$REPO/build" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$REPO/build" -j "$JOBS" --target ldls_driver ldls_tokenize
ls -l "$REPO/build/bin/ldls_driver" "$REPO/build/bin/ldls_tokenize"
