"""Tree -> (units, sequence ids, positions).

Terminology
-----------
* node     : a document element from the parser (heading, paragraph, function, JSON key, ...).
* unit     : a group of nodes whose tokens are laid out together. Every unit has a parent unit,
             so units form a tree as well (coarser than the node tree).
* sequence : a llama.cpp sequence id. Every LEAF unit owns exactly one sequence. A token belongs to
             the sequences of all leaf units below its unit, so it is visible from each of them.

Positions
---------
The prompt prefix occupies 0..P-1. The root unit starts at P. Every child unit starts right after
its parent unit ends, so all children of a unit start at the SAME position: siblings overlap in
position but never in sequence id, and they never attend to each other. Along any sequence the
positions are 0..len-1 with no gap, which is what llama.cpp requires.

Merge rules (small subtrees)
----------------------------
1. Only nodes whose type the parser marked `splittable` can start a unit (Markdown: headings).
2. A splittable child whose subtree has >= min_unit_tokens tokens becomes its own `subtree` unit.
3. Consecutive smaller splittable siblings are collected; if there are at least two of them and
   together they reach min_unit_tokens they form one `merged_group` unit, otherwise they are
   absorbed (whole subtrees) into the parent's unit.
4. Non-splittable children are always absorbed into the parent's unit with their whole subtree.
5. If the number of leaf units exceeds max_sequences, min_unit_tokens is raised by 50% and the
   assignment is redone until it fits (the effective value is recorded in the JSON).

Absorbed nodes keep their document order among the unit's own nodes, but a unit's own tokens
always come before its child units; a small subsection that sits after a large one is therefore
moved in front of it inside the shared prefix (recorded as `reordered` in the unit).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .tree import TreeNode

LLAMA_MAX_SEQ = 256  # src/llama-cparams.h of the pinned llama.cpp


@dataclass
class Unit:
    id: str
    kind: str                      # root | subtree | merged_group
    parent: "Unit | None"
    root_nodes: list[TreeNode]
    nodes: list[TreeNode] = field(default_factory=list)
    children: list["Unit"] = field(default_factory=list)
    seq_ids: list[int] = field(default_factory=list)
    pos_start: int = 0
    pos_end: int = 0
    reordered: bool = False

    @property
    def n_tokens(self) -> int:
        return sum(len(n.tokens) for n in self.nodes)


def _subtree_tokens(root: TreeNode) -> dict[int, int]:
    sizes: dict[int, int] = {}

    def rec(n: TreeNode) -> int:
        s = len(n.tokens) + sum(rec(c) for c in n.children)
        sizes[id(n)] = s
        return s

    rec(root)
    return sizes


def _assign_units(root: TreeNode, min_tokens: int) -> list[Unit]:
    size = _subtree_tokens(root)
    units: list[Unit] = []

    def new_unit(kind: str, parent: Unit | None, roots: list[TreeNode]) -> Unit:
        u = Unit(id=f"u{len(units)}", kind=kind, parent=parent, root_nodes=roots)
        units.append(u)
        if parent is not None:
            parent.children.append(u)
        return u

    def absorb(node: TreeNode, unit: Unit) -> None:
        if unit.children:
            unit.reordered = True  # lands in front of child units that precede it in the document
        for n in node.walk():
            n.unit = unit
            unit.nodes.append(n)

    def expand(node: TreeNode, unit: Unit) -> None:
        node.unit = unit
        unit.nodes.append(node)
        pending: list[TreeNode] = []

        def flush() -> None:
            if not pending:
                return
            total = sum(size[id(c)] for c in pending)
            if len(pending) >= 2 and total >= min_tokens:
                group = new_unit("merged_group", unit, list(pending))
                for c in pending:
                    absorb(c, group)
            else:
                for c in pending:
                    absorb(c, unit)
            pending.clear()

        for c in node.children:
            if c.splittable and size[id(c)] >= min_tokens:
                flush()
                expand(c, new_unit("subtree", unit, [c]))
            elif c.splittable:
                pending.append(c)
            else:
                flush()
                absorb(c, unit)
        flush()

    expand(root, new_unit("root", None, [root]))
    return units


def layout(root: TreeNode, prefix_len: int, min_unit_tokens: int = 40, max_sequences: int = 32) -> dict:
    """Assigns units, sequence ids and positions in place; returns the layout summary."""
    if max_sequences < 1 or max_sequences > LLAMA_MAX_SEQ:
        raise ValueError(f"max_sequences must be in [1, {LLAMA_MAX_SEQ}]")

    effective = min_unit_tokens
    while True:
        for n in root.walk():
            n.unit = None
        units = _assign_units(root, effective)
        leaves = [u for u in units if not u.children]
        if len(leaves) <= max_sequences:
            break
        effective = max(effective + 1, math.ceil(effective * 1.5))

    # sequence ids: leaf units in unit-tree DFS order
    next_seq = 0

    def number(u: Unit) -> list[int]:
        nonlocal next_seq
        if not u.children:
            u.seq_ids = [next_seq]
            next_seq += 1
        else:
            u.seq_ids = sorted(s for c in u.children for s in number(c))
        return u.seq_ids

    root_unit = units[0]
    number(root_unit)

    # positions: a unit starts where its parent ends; siblings share their start position
    def place(u: Unit, start: int) -> None:
        u.pos_start = start
        pos = start
        for n in u.nodes:
            n.positions = list(range(pos, pos + len(n.tokens)))
            pos += len(n.tokens)
        u.pos_end = pos
        for c in u.children:
            place(c, u.pos_end)

    place(root_unit, prefix_len)

    return {
        "n_seq": next_seq,
        "effective_min_unit_tokens": effective,
        "units": units,
    }
