"""Unit tests for the Markdown parser (tree shape, text ownership, markup classes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ldls.parsers.markdown import parse
from ldls.tree import MARKUP, finalize_tree

REPO = Path(__file__).resolve().parent.parent

SAMPLE = """# Title

Intro **bold** text.

## Section A

- item one
- item two
  - nested

```python
print("hi")
```

### Sub A

| a | b |
|---|---|
| 1 | 2 |

## Section B

> quoted line

---
"""


def by_title(root, title):
    return next(n for n in root.walk() if n.title == title)


def test_text_is_owned_exactly_once():
    root = parse(SAMPLE)
    assert root.subtree_text() == SAMPLE


@pytest.mark.parametrize("path", sorted((REPO / "data" / "markdown").glob("*.md")))
def test_sample_documents_reconstruct(path):
    src = path.read_text(encoding="utf-8")
    assert parse(src).subtree_text() == src


def test_headings_nest_by_level():
    root = parse(SAMPLE)
    finalize_tree(root)
    title = by_title(root, "Title")
    a, sub = by_title(root, "Section A"), by_title(root, "Sub A")
    assert title.parent is root
    assert a.parent is title
    assert sub.parent is a
    assert by_title(root, "Section B").parent is title


def test_block_types_and_nesting():
    root = parse(SAMPLE)
    finalize_tree(root)
    a = by_title(root, "Section A")
    kinds = [c.node_type for c in a.children]
    assert kinds == ["bullet_list", "code_block", "heading"]
    lst = a.children[0]
    assert [c.node_type for c in lst.children] == ["list_item", "list_item"]
    assert lst.children[1].children[0].node_type == "bullet_list"  # nested list stays nested
    table = by_title(root, "Sub A").children[0]
    assert table.node_type == "table" and len(table.children) == 2
    assert "|---|" in table.children[0].text  # header row owns the separator line


def test_only_headings_are_splittable():
    root = parse(SAMPLE)
    assert {n.node_type for n in root.walk() if n.splittable} == {"heading"}


def _markup_chars(node) -> str:
    raw = node.text.encode("utf-8")
    return bytes(b for b, c in zip(raw, node.byte_class) if c == MARKUP).decode("utf-8", "ignore")


def test_markup_classification():
    root = parse(SAMPLE)
    finalize_tree(root)
    assert _markup_chars(by_title(root, "Section A")) == "##"
    para = by_title(root, "Title").children[0]
    assert _markup_chars(para) == "****"
    code = by_title(root, "Section A").children[1]
    assert _markup_chars(code) == "```python```"  # fences + info string; the code itself is content
    item = by_title(root, "Section A").children[0].children[0]
    assert _markup_chars(item) == "-"
