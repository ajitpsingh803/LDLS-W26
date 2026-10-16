"""Unit tests for the domain-agnostic layout (units, merge rules, sequence ids, positions)."""

from __future__ import annotations

import pytest

from ldls.layout import layout
from ldls.tree import TreeNode, finalize_tree


def node(kind: str, ntok: int, *children: TreeNode, split: bool = False, title: str = "") -> TreeNode:
    n = TreeNode(node_type=kind, splittable=split, title=title)
    n.tokens = list(range(ntok))
    n.children = list(children)
    return n


def section(title: str, ntok: int, *children: TreeNode) -> TreeNode:
    return node("heading", ntok, *children, split=True, title=title)


def streams(root: TreeNode, n_seq: int, prefix: int) -> list[dict[int, tuple]]:
    st = [{p: ("prefix",) for p in range(prefix)} for _ in range(n_seq)]
    for n in root.walk():
        for i, p in enumerate(n.positions):
            for s in n.unit.seq_ids:
                assert p not in st[s], f"two tokens at position {p} in sequence {s}"
                st[s][p] = (n.title or n.node_type, i)
    return st


def test_big_sections_get_their_own_sequence_and_share_start_position():
    a = section("A", 50)
    b = section("B", 60)
    root = node("document", 5, a, b)
    finalize_tree(root)
    out = layout(root, prefix_len=10, min_unit_tokens=40, max_sequences=8)
    assert out["n_seq"] == 2
    assert a.unit is not b.unit
    assert a.positions[0] == b.positions[0] == 15  # prefix 10 + root 5
    assert a.unit.seq_ids == [0] and b.unit.seq_ids == [1]
    assert root.unit.seq_ids == [0, 1]


def test_every_sequence_is_contiguous_from_zero():
    root = node("document", 3,
                section("A", 10, node("p", 40), section("A1", 45), section("A2", 50)),
                section("B", 70),
                section("C", 5))
    finalize_tree(root)
    out = layout(root, prefix_len=7, min_unit_tokens=40)
    for s, st in enumerate(streams(root, out["n_seq"], 7)):
        assert sorted(st) == list(range(len(st))), f"sequence {s} has a gap"


def test_small_section_is_absorbed_into_parent():
    small = section("small", 10)
    big = section("big", 80)
    root = node("document", 2, small, big)
    finalize_tree(root)
    out = layout(root, prefix_len=0, min_unit_tokens=40)
    assert small.unit is root.unit
    assert out["n_seq"] == 1
    assert big.unit.seq_ids == [0]


def test_consecutive_small_sections_form_a_merged_group():
    s1, s2, s3 = section("s1", 20), section("s2", 15), section("s3", 10)
    root = node("document", 2, section("big", 90), s1, s2, s3)
    finalize_tree(root)
    out = layout(root, prefix_len=0, min_unit_tokens=40)
    assert s1.unit is s2.unit is s3.unit
    assert s1.unit.kind == "merged_group"
    assert out["n_seq"] == 2


def test_single_small_section_after_big_one_is_absorbed_and_marked_reordered():
    tail = section("tail", 10)
    root = node("document", 2, section("big", 90), tail)
    finalize_tree(root)
    layout(root, prefix_len=0, min_unit_tokens=40)
    assert tail.unit is root.unit
    assert root.unit.reordered
    # the absorbed tail comes before the big child unit in position order
    assert tail.positions[-1] < root.unit.children[0].pos_start


def test_non_splittable_children_stay_with_parent_even_if_large():
    para = node("paragraph", 500)
    root = node("document", 1, para, section("A", 50))
    finalize_tree(root)
    layout(root, prefix_len=0, min_unit_tokens=40)
    assert para.unit is root.unit


def test_nested_sections_share_parent_tokens():
    a1, a2 = section("A1", 50), section("A2", 50)
    a = section("A", 10, a1, a2)
    root = node("document", 1, a, section("B", 60))
    finalize_tree(root)
    out = layout(root, prefix_len=4, min_unit_tokens=40)
    assert out["n_seq"] == 3
    assert a.unit.seq_ids == [0, 1]
    assert a1.positions[0] == a2.positions[0] == a.positions[-1] + 1


def test_sequence_budget_raises_the_threshold():
    root = node("document", 1, *[section(f"s{i}", 45) for i in range(10)])
    finalize_tree(root)
    out = layout(root, prefix_len=0, min_unit_tokens=40, max_sequences=4)
    assert out["n_seq"] <= 4
    assert out["effective_min_unit_tokens"] > 40


def test_max_sequences_must_respect_llama_limit():
    root = node("document", 1)
    finalize_tree(root)
    with pytest.raises(ValueError):
        layout(root, prefix_len=0, max_sequences=257)
