"""Source document -> structured JSON (schema/structured_doc.schema.json).

    python3 -m ldls.build_doc --domain markdown data/markdown/*.md --out-dir build/docs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import DEFAULT_MODEL, DEFAULT_TOKENIZER_BIN, SCHEMA_VERSION
from .layout import layout
from .parsers import PARSERS
from .prompt import PREFIX, QUESTION_TEMPLATE
from .stats import compute_stats
from .tokenizer import LlamaCppTokenizer
from .tree import classify_tokens, finalize_tree, path_of

MAX_DOC_TOKENS = 2000  # week-1 limit (CPU only)


def build_document(source_text: str, domain: str, tokenizer: LlamaCppTokenizer, document_id: str,
                   source_path: str = "", min_unit_tokens: int = 40, max_sequences: int = 32) -> dict:
    root = PARSERS[domain](source_text)
    nodes = finalize_tree(root)

    for n, (toks, pieces) in zip(nodes, tokenizer.tokenize_many([n.text for n in nodes])):
        n.tokens = toks
        n.token_class = classify_tokens(n.byte_class, pieces)
    prefix_tokens = tokenizer.tokenize(PREFIX, parse_special=True)

    lay = layout(root, len(prefix_tokens), min_unit_tokens, max_sequences)
    all_seqs = list(range(lay["n_seq"]))
    split_types = sorted({n.node_type for n in nodes if n.splittable})

    doc = {
        "schema_version": SCHEMA_VERSION,
        "document_id": document_id,
        "domain": domain,
        "source": {
            "path": source_path,
            "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "n_bytes": len(source_text.encode("utf-8")),
        },
        "tokenizer": {"model": tokenizer.model.name, **tokenizer.info},
        "prompt": {
            "prefix": {
                "text": PREFIX,
                "tokens": prefix_tokens,
                "positions": list(range(len(prefix_tokens))),
                "seq_ids": all_seqs,
            },
            "question_template": QUESTION_TEMPLATE,
        },
        "layout": {
            "n_seq": lay["n_seq"],
            "merge_rules": {
                "split_node_types": split_types,
                "min_unit_tokens": min_unit_tokens,
                "effective_min_unit_tokens": lay["effective_min_unit_tokens"],
                "max_sequences": max_sequences,
                "small_splittable_siblings": "merged_group if >= 2 consecutive and together >= min_unit_tokens, else absorbed into the parent unit",
                "non_splittable": "absorbed into the parent unit with the whole subtree",
            },
            "units": [
                {
                    "id": u.id,
                    "kind": u.kind,
                    "parent_unit": u.parent.id if u.parent else None,
                    "root_nodes": [n.id for n in u.root_nodes],
                    "nodes": [n.id for n in u.nodes],
                    "seq_ids": u.seq_ids,
                    "pos_start": u.pos_start,
                    "pos_end": u.pos_end,
                    "n_tokens": u.n_tokens,
                    "reordered": u.reordered,
                }
                for u in lay["units"]
            ],
        },
        "nodes": [
            {
                "id": n.id,
                "node_type": n.node_type,
                "parent": n.parent.id if n.parent else None,
                "children": [c.id for c in n.children],
                "depth": n.depth,
                "title": n.title,
                "path": path_of(n),
                "attrs": n.attrs,
                "unit": n.unit.id,
                "text": n.text,
                "tokens": n.tokens,
                "positions": n.positions,
                "seq_ids": n.unit.seq_ids if n.tokens else [],
                "token_class": n.token_class,
            }
            for n in nodes
        ],
    }
    doc["stats"] = compute_stats(doc)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--domain", required=True, choices=sorted(PARSERS))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer-bin", type=Path, default=DEFAULT_TOKENIZER_BIN)
    ap.add_argument("--min-unit-tokens", type=int, default=40)
    ap.add_argument("--max-sequences", type=int, default=32)
    args = ap.parse_args(argv)

    tok = LlamaCppTokenizer(args.model, args.tokenizer_bin)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    status = 0
    for path in args.inputs:
        doc = build_document(path.read_text(encoding="utf-8"), args.domain, tok, path.stem, str(path),
                             args.min_unit_tokens, args.max_sequences)
        out = args.out_dir / f"{path.stem}.json"
        out.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
        s = doc["stats"]
        flag = ""
        if s["n_doc_tokens"] >= MAX_DOC_TOKENS:
            flag = f"  WARNING: >= {MAX_DOC_TOKENS} tokens"
            status = 1
        print(f"{out}: {s['n_nodes']} nodes, {s['n_doc_tokens']} tokens, depth {s['tree_depth']}, "
              f"{s['n_units']} units, {s['n_seq']} sequences, markup {100 * s['markup_share']:.1f}%{flag}")
    return status


if __name__ == "__main__":
    sys.exit(main())
