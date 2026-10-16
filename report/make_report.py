#!/usr/bin/env python3
"""Collect test results + tree statistics into results/summary.{json,md} for the one-slide report.
The editable report is supplied separately as report/slide.pptx.

    python3 report/make_report.py [--domain markdown] [--team "Team Markdown"]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent

# reference categorical palette, slots 1-3 (validated all-pairs, light mode); text never uses series colors
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
PASS, FAIL = "PASS", "FAIL"


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "CPU"


def llama_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO / "third_party" / "llama.cpp"), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "?"


def collect(domain: str) -> dict:
    rows = []
    for rp in sorted((REPO / "results" / domain).glob("*.json")):
        r = json.loads(rp.read_text())
        doc = json.loads((REPO / "build" / "test_docs" / domain / f"{rp.stem}.json").read_text())
        st, d, s = doc["stats"], r["driver"], r["summary"]
        t = d["timings_ms"]
        rows.append({
            "document": rp.stem,
            "stats": {
                "doc_tokens": st["n_doc_tokens"], "nodes": st["n_nodes"], "depth": st["tree_depth"],
                "units": st["n_units"], "sequences": st["n_seq"],
                "section_subtree_tokens": st["section_subtree_tokens"],
                "leaf_unit_tokens": st["leaf_unit_tokens"],
                "markup_share": st["markup_share"], "whitespace_share": st["whitespace_share"],
                "markup_or_whitespace_share": st["markup_or_whitespace_share"],
                "visible_fraction_mean": st["visible_fraction_mean"],
                "merged_small_subtrees": st["n_small_subtrees_merged"],
                "node_type_counts": st["node_type_counts"],
            },
            "tests": {
                "test1_change_b_ask_a": s["test1_pass"],
                "test1_logit_diff": d["test1_modify_b"]["max_abs_logit_diff"],
                "test1_changed_seqs": d["test1_modify_b"]["patch"]["changed_seqs"],
                "test1_redecoded_tokens": d["test1_modify_b"]["patch"]["n_redecoded"],
                "test2_evict_c": s["test2_pass"],
                "test2_cells_before": d["test2_evict"]["cells_before"],
                "test2_cells_freed": d["test2_evict"]["cells_freed"],
                "test2_kv_bytes_freed": d["test2_evict"]["kv_bytes_freed"],
                "test2_logit_diff": d["test2_evict"]["max_abs_logit_diff"],
                "keep_only_target_path": s["test2b_pass"],
                "keep_path_cells": d["test2b_keep_only_target_path"]["cells_after"],
                "control_a_changes_answer": s["control_a_pass"],
                "flat_flipped_by_distractor": not r["grading"]["flat_with_distractor_correct"],
            },
            "answers": s["answers"],
            "question": s["question"],
            "timings_ms": {
                "model_load": t["model_load_ms"],
                "prefill_flat": med(t["prefill_flat_ms_runs"]),
                "prefill_structured": med(t["prefill_structured_ms_runs"]),
                "question_flat": d["flat"]["original"]["t_question_ms"],
                "question_structured": d["structured"]["answer"]["t_question_ms"],
                "gen_per_token_flat": d["flat"]["original"]["t_generate_ms"] / max(1, len(d["flat"]["original"]["tokens"])),
                "gen_per_token_structured": d["structured"]["answer"]["t_generate_ms"] / max(1, len(d["structured"]["answer"]["tokens"])),
                "patch_b": t["patch_b_ms"],
                "full_reprefill_b": t["full_reprefill_b_ms"],
                "evict_c": t["evict_ms"],
            },
            "engine": d["engine"],
        })
    batched = json.loads((REPO / "results" / "batched_vs_separate.json").read_text())
    return {"domain": domain, "documents": rows, "batched_vs_separate": batched}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_markdown(summary: dict, path: Path, team: str) -> None:
    rows = summary["documents"]
    eng = rows[0]["engine"]
    b = summary["batched_vs_separate"]
    L = [f"# {team}: results ({summary['domain']})", "",
         f"Model `{Path(eng['model']).name}`, CPU `{cpu_name()}`, {eng['n_threads']} threads, "
         f"llama.cpp `{llama_commit()}`, greedy decoding, unified KV cache.", "",
         "## Tree statistics", "",
         "| Document | Tokens | Nodes | Depth | Units | Seqs | Section subtree tokens med (min-max) | Leaf unit tokens med (min-max) | Markup | Markup + whitespace | Avg. visible / flat |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["stats"]
        sec = s["section_subtree_tokens"]
        L.append(f"| {r['document']} | {s['doc_tokens']} | {s['nodes']} | {s['depth']} | {s['units']} | {s['sequences']} | "
                 f"{sec['median']:g} ({sec['min']}-{sec['max']}) | {s['leaf_unit_tokens']['median']:g} ({s['leaf_unit_tokens']['min']}-{s['leaf_unit_tokens']['max']}) | {pct(s['markup_share'])} | "
                 f"{pct(s['markup_or_whitespace_share'])} | {pct(s['visible_fraction_mean'])} |")
    L += ["", "## Tests", "",
          "| Document | T1 change B, ask A | T2 evict C | Cells freed | Keep only A's path | Control: change A | Flat + B distractor |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        t = r["tests"]
        L.append(f"| {r['document']} | {PASS if t['test1_change_b_ask_a'] else FAIL} (dlogit {t['test1_logit_diff']:.2g}) | "
                 f"{PASS if t['test2_evict_c'] else FAIL} (dlogit {t['test2_logit_diff']:.2g}) | "
                 f"{t['test2_cells_freed']} of {t['test2_cells_before']} ({t['test2_kv_bytes_freed'] / 2**20:.1f} MiB) | "
                 f"{PASS if t['keep_only_target_path'] else FAIL}, {t['keep_path_cells']} cells | "
                 f"{PASS if t['control_a_changes_answer'] else FAIL} | "
                 f"{'answer flipped' if t['flat_flipped_by_distractor'] else 'answer held'} |")
    L += ["", "Answers:", ""]
    for r in rows:
        a = r["answers"]
        L.append(f"* **{r['document']}**: {r['question']} structured `{a['structured']}`; after B edit "
                 f"`{a['structured_after_b_change']}`; after C eviction `{a['structured_after_c_eviction']}`; "
                 f"after A edit `{a['structured_after_a_change']}`; flat `{a['flat']}`; flat with B distractor "
                 f"`{a['flat_with_b_distractor']}`")
    ind, iso, sp = b["independent_prompts"], b["isolation_same_shape"], b["shared_prefix"]
    s0 = ind["sequences"][0]
    L += ["", "## Test 3: batched vs separate", "",
          f"* Independent prompts: {sum(s['tokens_equal'] for s in ind['sequences'])}/{len(ind['sequences'])} sequences token-identical."]
    for s in ind["sequences"]:
        if not s["tokens_equal"]:
            L.append(f"  Sequence {s['seq']} diverges at step {s['first_divergence_step']} on a near-tie "
                     f"(top-2 margin {s['margin_in_separate']:.3f} separate vs {s['margin_in_batched']:.3f} batched; "
                     f"max logit deviation before divergence {s['max_abs_logit_diff_before_divergence']:.2f}).")
    L += [f"* Same batch shape, other neighbour prompt: max logit difference {iso['max_abs_logit_diff_all_steps']} over "
          f"{len(s0['separate']['tokens'])} steps -> {PASS if iso['pass'] else FAIL} (no cross-sequence effect observed in this test).",
          f"* Shared prefix stored once (seq ids {{0,1}}): {PASS if sp['pass'] else FAIL}, "
          f"{sp['cells_batched']} KV cells vs {sp['cells_separate']} for separate runs.", "",
          "## CPU run times (median over documents, ms)", "",
          "| Step | ms |", "|---|---|"]
    for k, label in TIMING_LABELS:
        v = med([r['timings_ms'][k] for r in rows])
        L.append(f"| {label} | {v:.1f} |" if v >= 1 else f"| {label} | {v * 1000:.0f} µs |")
    L += ["", "Section subtree tokens include the H1 heading, whose subtree is the whole document; leaf units are",
          "the subtrees that own one sequence each. Markup = tokens whose non-whitespace bytes are all syntax."]
    path.write_text("\n".join(L) + "\n")


TIMING_LABELS = [
    ("model_load", "Model load (once)"),
    ("prefill_flat", "Prefill, flat"),
    ("prefill_structured", "Prefill, structured"),
    ("question_flat", "Question decode, flat (sees whole doc)"),
    ("question_structured", "Question decode, structured (sees A's path)"),
    ("gen_per_token_flat", "Generation per token, flat"),
    ("gen_per_token_structured", "Generation per token, structured"),
    ("full_reprefill_b", "Re-prefill after editing B"),
    ("patch_b", "Patch only B's sequences"),
    ("evict_c", "Evict subtree C (seq_rm)"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="markdown")
    ap.add_argument("--team", default="Team Markdown")
    args = ap.parse_args()
    summary = collect(args.domain)
    (REPO / "results" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_markdown(summary, REPO / "results" / "summary.md", args.team)
    print("wrote results/summary.json and results/summary.md; update the editable report slide if results changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
