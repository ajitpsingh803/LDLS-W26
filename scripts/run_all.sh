#!/usr/bin/env bash
# Full week-1 pipeline: setup, build, parse, validate, unit tests, foundation + tests 1-3, report.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
THREADS="${THREADS:-8}"
cd "$REPO"

bash scripts/setup_llama_cpp.sh
bash scripts/download_model.sh
bash scripts/build.sh

echo "== parse the 5 Markdown documents"
python3 -m ldls.build_doc --domain markdown data/markdown/*.md --out-dir build/docs
python3 -m ldls.validate build/docs/*.json --source-dir data/markdown --retokenize

echo "== unit tests"
python3 -m pytest

echo "== common foundation (smoke test, examples/batched, test 3)"
THREADS="$THREADS" bash scripts/run_foundation.sh

echo "== tests 1 and 2 on every document"
python3 tests/run_doc_tests.py --domain markdown --threads "$THREADS"

echo "== report"
python3 report/make_report.py --domain markdown
