"""Tree statistics reported on the team slide (computed from a structured JSON document).

Definitions
-----------
tree_depth          : maximum node depth; the document root has depth 0.
subtree size        : number of document tokens in a node's subtree (own tokens + all descendants).
                      Reported for every splittable node (Markdown: every section) and per unit.
markup_share        : tokens whose non-whitespace bytes are all syntax / total document tokens.
whitespace_share    : tokens that are only spaces/newlines / total document tokens.
visible_fraction    : tokens a sequence attends to (prefix + its root-to-leaf units)
                      / tokens of the flat prompt (prefix + whole document).
The prompt prefix (chat template) is excluded from every document count.
"""

from __future__ import annotations

import statistics


def _summary(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": round(statistics.fmean(values), 1),
        "max": max(values),
    }


def compute_stats(doc: dict) -> dict:
    nodes = doc["nodes"]
    by_id = {n["id"]: n for n in nodes}
    units = doc["layout"]["units"]
    n_prefix = len(doc["prompt"]["prefix"]["tokens"])

    subtree: dict[str, int] = {}

    def size(nid: str) -> int:
        if nid not in subtree:
            n = by_id[nid]
            subtree[nid] = len(n["tokens"]) + sum(size(c) for c in n["children"])
        return subtree[nid]

    root = next(n for n in nodes if n["parent"] is None)
    n_doc = size(root["id"])

    classes = "".join(n["token_class"] for n in nodes)
    counts = {c: classes.count(c) for c in "MCW"}

    type_counts: dict[str, int] = {}
    for n in nodes:
        type_counts[n["node_type"]] = type_counts.get(n["node_type"], 0) + 1

    split_types = set(doc["layout"]["merge_rules"]["split_node_types"])
    split_nodes = [n for n in nodes if n["node_type"] in split_types]
    unit_roots = {r for u in units if u["kind"] == "subtree" for r in u["root_nodes"]}
    merged_small = [n for n in split_nodes if n["id"] not in unit_roots and n["parent"] is not None]

    n_seq = doc["layout"]["n_seq"]
    visible = []
    for s in range(n_seq):
        visible.append(n_prefix + sum(u["n_tokens"] for u in units if s in u["seq_ids"]))
    flat = n_prefix + n_doc

    return {
        "n_nodes": len(nodes),
        "node_type_counts": dict(sorted(type_counts.items())),
        "tree_depth": max(n["depth"] for n in nodes),
        "n_doc_tokens": n_doc,
        "n_prefix_tokens": n_prefix,
        "markup_tokens": counts["M"],
        "whitespace_tokens": counts["W"],
        "content_tokens": counts["C"],
        "markup_share": counts["M"] / n_doc if n_doc else 0.0,
        "whitespace_share": counts["W"] / n_doc if n_doc else 0.0,
        "markup_or_whitespace_share": (counts["M"] + counts["W"]) / n_doc if n_doc else 0.0,
        "n_units": len(units),
        "n_seq": n_seq,
        "n_merged_groups": sum(u["kind"] == "merged_group" for u in units),
        "n_small_subtrees_merged": len(merged_small),
        "section_subtree_tokens": _summary([size(n["id"]) for n in split_nodes]),
        "unit_tokens": _summary([u["n_tokens"] for u in units]),
        "leaf_unit_tokens": _summary([u["n_tokens"] for u in units if len(u["seq_ids"]) == 1 and not any(
            v["parent_unit"] == u["id"] for v in units)]),
        "visible_tokens_per_seq": _summary(visible),
        "visible_fraction_mean": (sum(visible) / len(visible) / flat) if visible else 0.0,
        "flat_prompt_tokens": flat,
    }
