"""Domain-agnostic document tree produced by every parser."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

# Per-byte classes used for the "share of markup tokens" statistic.
MARKUP = "M"      # syntax: heading markers, list bullets, code fences, tags, braces, ...
CONTENT = "C"     # text a reader would keep if all syntax were stripped (code bodies count as content)
WHITESPACE = "W"  # spaces and newlines


@dataclass
class TreeNode:
    node_type: str
    text: str = ""                 # the node's OWN source text (children excluded)
    byte_class: str = ""           # one of M/C/W per UTF-8 byte of `text`
    title: str = ""                # short label (heading text, key name, function name, ...)
    splittable: bool = False       # may become its own sequence unit
    attrs: dict = field(default_factory=dict)
    children: list["TreeNode"] = field(default_factory=list)

    # filled in by the pipeline
    id: str = ""
    parent: "TreeNode | None" = None
    depth: int = 0
    tokens: list[int] = field(default_factory=list)
    token_class: str = ""
    unit: "object | None" = None
    positions: list[int] = field(default_factory=list)

    def add(self, child: "TreeNode") -> "TreeNode":
        self.children.append(child)
        return child

    def walk(self) -> Iterator["TreeNode"]:
        """Pre-order (document order) traversal."""
        yield self
        for c in self.children:
            yield from c.walk()

    def subtree_text(self) -> str:
        return "".join(n.text for n in self.walk())


def finalize_tree(root: TreeNode) -> list[TreeNode]:
    """Assign ids, parents and depths in document order; returns the node list."""
    nodes = []
    stack = [(root, None, 0)]
    while stack:
        node, parent, depth = stack.pop()
        node.id = f"n{len(nodes)}"
        node.parent = parent
        node.depth = depth
        nodes.append(node)
        for c in reversed(node.children):
            stack.append((c, node, depth + 1))
    for n in nodes:
        if len(n.byte_class) != len(n.text.encode("utf-8")):
            raise ValueError(f"{n.id}: byte_class length does not match text")
    return nodes


def path_of(node: TreeNode) -> str:
    parts = []
    n = node
    while n is not None:
        if n.title:
            parts.append(n.title)
        n = n.parent
    return " > ".join(reversed(parts))


def classify_tokens(byte_class: str, piece_bytes: list[int]) -> str:
    """A token is markup if every non-whitespace byte is markup, whitespace if all bytes are."""
    out = []
    off = 0
    for n in piece_bytes:
        cls = byte_class[off:off + n]
        off += n
        if not cls or set(cls) == {WHITESPACE}:
            out.append(WHITESPACE)
        elif CONTENT in cls:
            out.append(CONTENT)
        else:
            out.append(MARKUP)
    if off != len(byte_class):
        raise ValueError("token pieces do not cover the text exactly")
    return "".join(out)
