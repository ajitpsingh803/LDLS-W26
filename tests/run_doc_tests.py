#!/usr/bin/env python3
"""Tests 1 and 2 (plus controls) for every document spec of a domain.

For each spec in tests/specs/<domain>/*.json:
  1. parse the original document and two edited versions (B edited, A edited) into the JSON schema
  2. check that each edit changed only nodes inside the intended subtree
  3. run `ldls_driver run-tests` (structured prefill, patch B / ask A, evict C, controls)
  4. grade the answers against the expected strings and write results/<domain>/<doc>.json

    python3 tests/run_doc_tests.py --domain markdown [--threads 8]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ldls import DEFAULT_DRIVER_BIN, DEFAULT_MODEL  # noqa: E402
from ldls.build_doc import build_document  # noqa: E402
from ldls.tokenizer import LlamaCppTokenizer  # noqa: E402
from ldls.validate import validate_doc  # noqa: E402


def normalize(s: str) -> str:
    return re.sub(r"[\s,]+", " ", s.lower()).strip()


def contains_any(answer: str, expected: list[str]) -> bool:
    a = normalize(answer).replace(" ", "")
    return any(normalize(e).replace(" ", "") in a for e in expected)


def find_heading(doc: dict, title: str) -> dict:
    hits = [n for n in doc["nodes"] if n["title"] == title]
    if len(hits) != 1:
        raise SystemExit(f"{doc['document_id']}: title {title!r} matches {len(hits)} nodes")
    return hits[0]


def subtree_ids(doc: dict, node_id: str) -> set[str]:
    by_id = {n["id"]: n for n in doc["nodes"]}
    out, stack = set(), [node_id]
    while stack:
        nid = stack.pop()
        out.add(nid)
        stack.extend(by_id[nid]["children"])
    return out


def edit(src: str, find: str, replace: str, where: str) -> str:
    count = src.count(find)
    if count != 1:
        raise SystemExit(f"{where}: find string occurs {count} times (must be exactly once)")
    return src.replace(find, replace)


def edit_confined_to(a: dict, b: dict, title: str) -> dict:
    """The edit may add/remove/change nodes inside the subtree `title`; everything else must be identical."""
    def split(doc: dict) -> tuple[list, list]:
        sub = subtree_ids(doc, find_heading(doc, title)["id"])
        inside = [(n["node_type"], n["text"]) for n in doc["nodes"] if n["id"] in sub]
        outside = [(n["node_type"], n["text"]) for n in doc["nodes"] if n["id"] not in sub]
        return inside, outside

    ina, outa = split(a)
    inb, outb = split(b)
    return {
        "subtree": title,
        "nodes_before": len(ina),
        "nodes_after": len(inb),
        "subtree_changed": ina != inb,
        "outside_unchanged": outa == outb,
    }


def run_spec(spec_path: Path, tok: LlamaCppTokenizer, out_dir: Path, work_dir: Path, args) -> dict:
    spec = json.loads(spec_path.read_text())
    src_path = REPO / spec["document"]
    src = src_path.read_text(encoding="utf-8")
    doc_id = src_path.stem
    domain = spec["domain"]

    def build(text: str, suffix: str) -> tuple[dict, Path]:
        d = build_document(text, domain, tok, doc_id, spec["document"], args.min_unit_tokens, args.max_sequences)
        validate_doc(d, text)
        p = work_dir / f"{doc_id}{suffix}.json"
        p.write_text(json.dumps(d, ensure_ascii=False))
        return d, p

    d0, p0 = build(src, "")
    target = find_heading(d0, spec["target"]["title"])
    node_b = find_heading(d0, spec["modify_b"]["title"])
    node_c = find_heading(d0, spec["evict_c"]["title"])

    src_b = edit(src, spec["modify_b"]["find"], spec["modify_b"]["replace"], f"{spec_path.name} modify_b")
    src_a = edit(src, spec["control_a"]["find"], spec["control_a"]["replace"], f"{spec_path.name} control_a")
    dB, pB = build(src_b, ".mod_b")
    dA, pA = build(src_a, ".mod_a")

    eb = edit_confined_to(d0, dB, spec["modify_b"]["title"])
    ea = edit_confined_to(d0, dA, spec["target"]["title"])
    edits = {
        "modify_b": eb,
        "control_a": ea,
        "b_disjoint_from_a": not (subtree_ids(d0, node_b["id"]) & subtree_ids(d0, target["id"])),
        "c_disjoint_from_a": not (subtree_ids(d0, node_c["id"]) & subtree_ids(d0, target["id"])),
    }
    checks = [eb["subtree_changed"], eb["outside_unchanged"], ea["subtree_changed"], ea["outside_unchanged"],
              edits["b_disjoint_from_a"], edits["c_disjoint_from_a"]]
    if not all(checks):
        raise SystemExit(f"{spec_path.name}: edits / subtrees are not where the spec says: {edits}")

    raw = out_dir / f"{doc_id}.driver.json"
    cmd = [
        str(args.driver), "run-tests", "--model", str(args.model),
        "--doc", str(p0), "--doc-mod-b", str(pB), "--doc-mod-a", str(pA),
        "--target", target["id"], "--modify", node_b["id"], "--evict", node_c["id"],
        "--question", spec["question"], "--threads", str(args.threads),
        "--n-predict", str(args.n_predict), "--out", str(raw),
    ]
    if args.trace_generation:
        cmd.append("--trace-generation")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"driver failed on {doc_id}:\n{proc.stderr[-3000:]}")
    r = json.loads(raw.read_text())

    t1, t2, t2b = r["test1_modify_b"], r["test2_evict"], r["test2b_keep_only_target_path"]
    base_text = r["structured"]["answer"]["text"]
    grading = {
        "baseline_structured_correct": contains_any(base_text, spec["expected"]),
        "baseline_flat_correct": contains_any(r["flat"]["original"]["text"], spec["expected"]),
        "flat_with_distractor_correct": contains_any(r["flat"]["modified_b"]["text"], spec["expected"]),
        "test1_answer_correct": contains_any(t1["answer"]["text"], spec["expected"]),
        "test2_answer_correct": contains_any(t2["answer"]["text"], spec["expected"]),
        "control_a_answer_has_new_fact": contains_any(r["control_modify_a"]["answer"]["text"],
                                                       spec["control_a"]["expected"]),
    }
    summary = {
        "document_id": doc_id,
        "question": spec["question"],
        "expected": spec["expected"],
        "target": target["path"], "modified": node_b["path"], "evicted": node_c["path"],
        "test1_pass": t1["pass"] and grading["test1_answer_correct"],
        "test2_pass": t2["pass"] and grading["test2_answer_correct"],
        "test2b_pass": t2b["pass"],
        "control_a_pass": r["control_modify_a"]["target_seq_changed"] and grading["control_a_answer_has_new_fact"],
        "answers": {
            "flat": r["flat"]["original"]["text"],
            "flat_with_b_distractor": r["flat"]["modified_b"]["text"],
            "structured": base_text,
            "structured_after_b_change": t1["answer"]["text"],
            "structured_after_a_change": r["control_modify_a"]["answer"]["text"],
            "structured_after_c_eviction": t2["answer"]["text"],
        },
        "kv_cells": {
            "flat": r["flat"]["cells"],
            "structured": r["structured"]["cells"],
            "after_evict_c": t2["cells_after"],
            "freed_by_evict_c": t2["cells_freed"],
            "expected_freed": t2["cells_expected_freed"],
            "kv_bytes_freed": t2["kv_bytes_freed"],
            "keep_only_target_path": t2b["cells_after"],
        },
    }
    result = {"spec": spec, "edits": edits, "grading": grading, "summary": summary, "driver": r}
    (out_dir / f"{doc_id}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    raw.unlink()
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="markdown")
    ap.add_argument("--specs", nargs="*", type=Path)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--driver", type=Path, default=DEFAULT_DRIVER_BIN)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-predict", type=int, default=48)
    ap.add_argument("--min-unit-tokens", type=int, default=40)
    ap.add_argument("--max-sequences", type=int, default=32)
    ap.add_argument("--trace-generation", action="store_true")
    ap.add_argument("--results-dir", type=Path, help="Separate output directory for diagnostic runs")
    ap.add_argument("--strict", action="store_true", help="Also fail on the optional keep-path comparison")
    args = ap.parse_args()

    specs = args.specs or sorted((REPO / "tests" / "specs" / args.domain).glob("*.json"))
    out_dir = args.results_dir or REPO / "results" / args.domain
    work_dir = REPO / "build" / "test_docs" / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    tok = LlamaCppTokenizer(args.model)

    ok = True
    for sp in specs:
        s = run_spec(sp, tok, out_dir, work_dir, args)
        flags = " ".join(f"{k}={'PASS' if s[k] else 'FAIL'}" for k in ("test1_pass", "test2_pass", "test2b_pass", "control_a_pass"))
        print(f"{s['document_id']:<22} {flags}")
        print(f"    structured: {s['answers']['structured']!r}   flat+distractor: {s['answers']['flat_with_b_distractor']!r}")
        ok = ok and s["test1_pass"] and s["test2_pass"]
        if args.strict:
            ok = ok and s["test2b_pass"] and s["control_a_pass"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
