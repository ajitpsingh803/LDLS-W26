"""Validate structured JSON documents against the shared schema and its invariants.

    python3 -m ldls.validate build/docs/*.json [--source-dir data/markdown] [--retokenize]

Checks (beyond the field types of schema/structured_doc.schema.json):
  * tree: unique ids, one root, parent/children agree, depth = parent depth + 1, node list in DFS order
  * tokens / positions / token_class have equal length; positions contiguous inside a node
  * every node's seq_ids equal its unit's seq_ids; each leaf unit owns exactly one sequence
  * units: child starts where the parent ends, pos_end - pos_start = n_tokens, nodes inside the range
  * every sequence holds positions 0..len-1 exactly once (llama.cpp requires consecutive positions)
  * along every sequence the set of sequences of a token never grows (shared tokens form a prefix,
    which is what llama.cpp's mask -- built from seq_id[i][0] -- needs to be exact)
  * merge rules respected; n_seq <= max_sequences <= LLAMA_MAX_SEQ (256)
  * concatenated node text reproduces the source file (sha256), optional re-tokenization check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LLAMA_MAX_SEQ = 256
REQUIRED_NODE_FIELDS = {"id": str, "node_type": str, "children": list, "tokens": list, "positions": list,
                        "seq_ids": list, "token_class": str, "text": str, "unit": str, "depth": int}


class ValidationError(Exception):
    pass


def validate_doc(doc: dict, source_text: str | None = None, tokenizer=None, max_doc_tokens: int = 2000) -> list[str]:
    """Raises ValidationError on the first hard error; returns a list of warnings."""
    warnings: list[str] = []

    def fail(msg: str) -> None:
        raise ValidationError(f"{doc.get('document_id', '?')}: {msg}")

    for key in ("schema_version", "document_id", "domain", "prompt", "layout", "nodes"):
        if key not in doc:
            fail(f"missing top-level field {key}")

    nodes = doc["nodes"]
    by_id: dict[str, dict] = {}
    for n in nodes:
        for f, t in REQUIRED_NODE_FIELDS.items():
            if not isinstance(n.get(f), t):
                fail(f"node {n.get('id')}: field {f} missing or not {t.__name__}")
        if "parent" not in n or not (n["parent"] is None or isinstance(n["parent"], str)):
            fail(f"node {n['id']}: parent must be a node id or null")
        if n["id"] in by_id:
            fail(f"duplicate node id {n['id']}")
        by_id[n["id"]] = n

    roots = [n for n in nodes if n["parent"] is None]
    if len(roots) != 1:
        fail(f"expected one root, found {len(roots)}")
    order = []
    stack = [roots[0]["id"]]
    while stack:
        nid = stack.pop()
        order.append(nid)
        n = by_id[nid]
        for c in n["children"]:
            if c not in by_id:
                fail(f"node {nid}: unknown child {c}")
            if by_id[c]["parent"] != nid:
                fail(f"node {c}: parent is {by_id[c]['parent']}, expected {nid}")
            if by_id[c]["depth"] != n["depth"] + 1:
                fail(f"node {c}: wrong depth")
        stack.extend(reversed(n["children"]))
    if order != [n["id"] for n in nodes]:
        fail("nodes are not listed in document (DFS) order or the tree is not connected")

    vocab = doc.get("tokenizer", {}).get("vocab_size")
    for n in nodes:
        if not (len(n["tokens"]) == len(n["positions"]) == len(n["token_class"])):
            fail(f"node {n['id']}: tokens/positions/token_class lengths differ")
        if any(b - a != 1 for a, b in zip(n["positions"], n["positions"][1:])):
            fail(f"node {n['id']}: positions not contiguous")
        if vocab and any(not (0 <= t < vocab) for t in n["tokens"]):
            fail(f"node {n['id']}: token id out of vocabulary")
        if set(n["token_class"]) - set("MCW"):
            fail(f"node {n['id']}: token_class must use M/C/W")

    # units
    layout = doc["layout"]
    n_seq = layout["n_seq"]
    rules = layout["merge_rules"]
    if not (1 <= n_seq <= rules["max_sequences"] <= LLAMA_MAX_SEQ):
        fail(f"n_seq={n_seq} violates 1 <= n_seq <= max_sequences <= {LLAMA_MAX_SEQ}")
    units = {u["id"]: u for u in layout["units"]}
    children_of: dict[str, list[dict]] = {uid: [] for uid in units}
    for u in layout["units"]:
        if u["parent_unit"] is not None:
            children_of[u["parent_unit"]].append(u)
    owners: dict[int, str] = {}
    for u in layout["units"]:
        kids = children_of[u["id"]]
        if kids:
            expect = sorted(s for k in kids for s in k["seq_ids"])
            if u["seq_ids"] != expect:
                fail(f"unit {u['id']}: seq_ids must be the union of its child units")
        else:
            if len(u["seq_ids"]) != 1:
                fail(f"leaf unit {u['id']} must own exactly one sequence")
            s = u["seq_ids"][0]
            if s in owners:
                fail(f"sequence {s} owned by {owners[s]} and {u['id']}")
            owners[s] = u["id"]
        parent = units.get(u["parent_unit"]) if u["parent_unit"] else None
        prefix_len = len(doc["prompt"]["prefix"]["tokens"])
        if u["pos_start"] != (parent["pos_end"] if parent else prefix_len):
            fail(f"unit {u['id']}: must start where its parent unit ends")
        ntok = sum(len(by_id[nid]["tokens"]) for nid in u["nodes"])
        if ntok != u["n_tokens"] or u["pos_end"] - u["pos_start"] != ntok:
            fail(f"unit {u['id']}: token count / position range mismatch")
        for nid in u["nodes"]:
            n = by_id[nid]
            if n["unit"] != u["id"]:
                fail(f"node {nid} listed in {u['id']} but has unit {n['unit']}")
            if n["tokens"] and n["seq_ids"] != u["seq_ids"]:
                fail(f"node {nid}: seq_ids differ from its unit")
            if n["positions"] and not (u["pos_start"] <= n["positions"][0] and n["positions"][-1] < u["pos_end"]):
                fail(f"node {nid}: positions outside its unit range")
        if u["kind"] == "subtree":
            sub = _subtree_tokens(by_id, u["root_nodes"][0])
            if sub < rules["effective_min_unit_tokens"]:
                fail(f"unit {u['id']}: subtree of {sub} tokens is below min_unit_tokens")
        if u["kind"] == "merged_group":
            total = sum(_subtree_tokens(by_id, r) for r in u["root_nodes"])
            if total < rules["effective_min_unit_tokens"] or len(u["root_nodes"]) < 2:
                fail(f"merged_group {u['id']} violates the merge rule")
    if sorted(owners) != list(range(n_seq)):
        fail("sequence ids must be 0..n_seq-1, one per leaf unit")
    if sum(len(u["nodes"]) for u in layout["units"]) != len(nodes):
        fail("every node must belong to exactly one unit")

    # per-sequence streams
    prefix = doc["prompt"]["prefix"]
    if prefix["positions"] != list(range(len(prefix["tokens"]))) or prefix["seq_ids"] != list(range(n_seq)):
        fail("prefix must occupy positions 0..P-1 in every sequence")
    if "{question}" not in doc["prompt"]["question_template"]:
        fail("question_template must contain {question}")
    streams: list[dict[int, tuple]] = [dict() for _ in range(n_seq)]
    for p in prefix["positions"]:
        for s in range(n_seq):
            streams[s][p] = tuple(prefix["seq_ids"])
    for n in nodes:
        for p in n["positions"]:
            for s in n["seq_ids"]:
                if p in streams[s]:
                    fail(f"sequence {s} has two tokens at position {p}")
                streams[s][p] = tuple(n["seq_ids"])
    for s, st in enumerate(streams):
        if sorted(st) != list(range(len(st))):
            fail(f"sequence {s}: positions are not 0..{len(st) - 1}")
        prev = None
        for p in range(len(st)):
            cur = set(st[p])
            if prev is not None and not cur <= prev:
                fail(f"sequence {s}: sequence set grows at position {p}")
            prev = cur

    n_doc = sum(len(n["tokens"]) for n in nodes)
    if n_doc >= max_doc_tokens:
        warnings.append(f"{n_doc} document tokens (week-1 limit is {max_doc_tokens})")

    if source_text is not None:
        rebuilt = "".join(n["text"] for n in nodes)
        if rebuilt != source_text:
            fail("concatenated node text does not reproduce the source document")
    if doc.get("source", {}).get("sha256"):
        rebuilt = "".join(n["text"] for n in nodes).encode("utf-8")
        if hashlib.sha256(rebuilt).hexdigest() != doc["source"]["sha256"]:
            fail("source sha256 does not match the concatenated node text")

    if tokenizer is not None:
        retok = tokenizer.tokenize_many([n["text"] for n in nodes])
        for n, (toks, _) in zip(nodes, retok):
            if toks != n["tokens"]:
                fail(f"node {n['id']}: tokens differ from the model tokenizer")
        if tokenizer.tokenize(prefix["text"], parse_special=True) != prefix["tokens"]:
            fail("prefix tokens differ from the model tokenizer")
    return warnings


def _subtree_tokens(by_id: dict, nid: str) -> int:
    n = by_id[nid]
    return len(n["tokens"]) + sum(_subtree_tokens(by_id, c) for c in n["children"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("docs", nargs="+", type=Path)
    ap.add_argument("--source-dir", type=Path, help="compare the reconstructed text with <source-dir>/<id>.<ext>")
    ap.add_argument("--retokenize", action="store_true", help="re-tokenize every node with the model tokenizer")
    args = ap.parse_args(argv)

    tok = None
    if args.retokenize:
        from .tokenizer import LlamaCppTokenizer
        tok = LlamaCppTokenizer()
    bad = 0
    for p in args.docs:
        doc = json.loads(p.read_text(encoding="utf-8"))
        src = None
        if args.source_dir:
            matches = list(args.source_dir.glob(doc["document_id"] + ".*"))
            src = matches[0].read_text(encoding="utf-8") if matches else None
        try:
            warns = validate_doc(doc, src, tok)
            print(f"OK   {p}" + "".join(f"\n     warning: {w}" for w in warns))
        except ValidationError as e:
            bad += 1
            print(f"FAIL {p}: {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
