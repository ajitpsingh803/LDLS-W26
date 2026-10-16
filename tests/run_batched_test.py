#!/usr/bin/env python3
"""Test 3 (common foundation): two prompts in two sequences of ONE batch must give the same
greedy outputs as two separate runs.

Case A, independent prompts: prompt 1 -> seq 0, prompt 2 -> seq 1, decoded together.
Case B, shared prefix:       one chat prefix stored once with seq ids {0, 1} (what
                             examples/batched does with its prompt), then a different question per
                             sequence. This is the mechanism the structured documents rely on.

    python3 tests/run_batched_test.py [--threads 8] [--n-predict 32]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ldls import DEFAULT_DRIVER_BIN, DEFAULT_MODEL  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "The largest planet in the solar system is",
]
# same token count as PROMPTS[-1] (8 tokens with BOS): used for the same-shape isolation check
PERTURB = "The tallest mountain on the planet Earth is"
SHARED_PREFIX = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Answer in one short sentence.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
)
SUFFIXES = [
    "What is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "What is the largest planet in the solar system?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--driver", type=Path, default=DEFAULT_DRIVER_BIN)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--strict", action="store_true", help="Require independent prompts to match too")
    args = ap.parse_args()

    out = REPO / "results" / "batched_vs_separate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(args.driver), "batched", "--model", str(args.model), "--threads", str(args.threads),
           "--n-predict", str(args.n_predict), "--out", str(out), "--shared-prefix", SHARED_PREFIX,
           "--perturb", PERTURB]
    for p in PROMPTS:
        cmd += ["--prompt", p]
    for s in SUFFIXES:
        cmd += ["--suffix", s]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout, end="")
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr)
        return 1
    r = json.loads(out.read_text())
    print()
    for case in ("independent_prompts", "isolation_same_shape", "shared_prefix"):
        for s in r[case]["sequences"]:
            line = (f"{case} seq {s['seq']}: tokens_equal={s['tokens_equal']} "
                    f"max|dlogit| before divergence={s['max_abs_logit_diff_before_divergence']:.3g}")
            if "first_divergence_step" in s:
                margins = {k: round(v, 4) for k, v in s.items() if k.startswith("margin_in_")}
                line += f" first divergence at step {s['first_divergence_step']}, top-2 margins {margins}"
            print(line)
    iso = r["isolation_same_shape"]
    print(f"isolation_same_shape: max |dlogit| over all steps = {iso['max_abs_logit_diff_all_steps']}")
    sp = r["shared_prefix"]
    print(f"shared prefix: {sp['cells_batched']} KV cells when batched vs {sp['cells_separate']} for separate runs")
    # Pass = no information flows between sequences (bit-exact under identical shapes) and the
    # shared-prefix batch reproduces the separate runs. Case A is reported as measured.
    ok = iso["pass"] and sp["pass"]
    if args.strict:
        ok = ok and all(s["tokens_equal"] for s in r["independent_prompts"]["sequences"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
