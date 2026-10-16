"""Markdown -> TreeNode tree, using markdown-it-py.

Tree shape
----------
document
  heading (h1)                      headings nest by level: an h3 is a child of the preceding h2
    paragraph / code_block / table / blockquote / thematic_break / html_block
    bullet_list | ordered_list
      list_item                     a tight item's paragraph is folded into the item
        bullet_list ...             nested lists stay nested
    table
      table_row                     the header row also owns the |---| separator line
    heading (h2)
      ...

Every node owns a contiguous range of source lines: its own lines start at the node and stop where
its first child starts; the children's ranges tile the rest of the node. Blank lines between blocks
belong to the block before them. Concatenating the nodes' text in document order therefore gives the
original file back byte for byte.

Only headings are `splittable`, i.e. sections are the units that can get their own sequence.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

from ..tree import CONTENT, MARKUP, WHITESPACE, TreeNode

_BLOCK_TYPES = {
    "paragraph_open": "paragraph",
    "bullet_list_open": "bullet_list",
    "ordered_list_open": "ordered_list",
    "list_item_open": "list_item",
    "blockquote_open": "blockquote",
    "table_open": "table",
    "tr_open": "table_row",
    "heading_open": "heading",
}
_LEAF_TYPES = {"fence": "code_block", "code_block": "code_block", "hr": "thematic_break", "html_block": "html_block"}
_TRANSPARENT = {"thead_open", "tbody_open", "th_open", "td_open"}


def _split_lines(src: str) -> list[str]:
    """Split on "\\n" only (markdown-it's line maps count "\\n"; str.splitlines also splits on \\u2028 etc.)."""
    parts = src.split("\n")
    lines = [p + "\n" for p in parts[:-1]]
    if parts[-1]:
        lines.append(parts[-1])
    return lines


class _Block:
    def __init__(self, node_type: str, start: int, end: int, attrs: dict | None = None):
        self.node_type = node_type
        self.start = start
        self.end = end
        self.attrs = attrs or {}
        self.children: list[_Block] = []
        self.title = ""
        self.in_blockquote = False


def _block_tree(src: str) -> tuple[_Block, list[str]]:
    md = MarkdownIt("commonmark").enable("table").enable("strikethrough")
    tokens = md.parse(src)
    lines = _split_lines(src)
    root = _Block("document", 0, len(lines))
    stack: list[_Block | None] = [root]  # None marks a transparent container

    def top() -> _Block:
        for b in reversed(stack):
            if b is not None:
                return b
        return root

    for t in tokens:
        if t.nesting == 1:
            if t.type in _BLOCK_TYPES and not (t.type == "paragraph_open" and t.hidden):
                attrs = {}
                if t.type == "heading_open":
                    attrs["level"] = int(t.tag[1])
                elif t.type == "ordered_list_open":
                    attrs["start"] = int(t.attrs.get("start", 1))
                b = _Block(_BLOCK_TYPES[t.type], t.map[0], t.map[1], attrs)
                b.in_blockquote = top().in_blockquote or t.type == "blockquote_open"
                top().children.append(b)
                stack.append(b)
            else:
                stack.append(None)  # thead/tbody/th/td and hidden (tight list) paragraphs
        elif t.nesting == -1:
            stack.pop()
        elif t.type == "inline":
            b = top()
            if b.node_type == "heading" and not b.title:
                b.title = t.content.strip()
        elif t.type in _LEAF_TYPES and t.map:
            attrs = {"language": t.info.strip()} if t.type == "fence" and t.info.strip() else {}
            if t.type == "fence":
                attrs["fenced"] = True
            b = _Block(_LEAF_TYPES[t.type], t.map[0], t.map[1], attrs)
            b.in_blockquote = top().in_blockquote
            top().children.append(b)
    return root, lines


def _nest_headings(root: _Block) -> None:
    """Turn the flat list of top-level blocks into a section tree by heading level."""
    flat = root.children
    root.children = []
    stack: list[tuple[int, _Block]] = [(0, root)]
    for b in flat:
        if b.node_type == "heading":
            level = b.attrs["level"]
            while stack[-1][0] >= level:
                stack.pop()
            stack[-1][1].children.append(b)
            stack.append((level, b))
        else:
            stack[-1][1].children.append(b)


def _tile(b: _Block, end: int) -> None:
    """Children tile [first_child.start, end); the node keeps [start, first_child.start)."""
    b.end = end
    b.children.sort(key=lambda c: c.start)
    for i, c in enumerate(b.children):
        nxt = b.children[i + 1].start if i + 1 < len(b.children) else end
        _tile(c, nxt)


# ------------------------------------------------------------------------------------------------
# markup classification (per character, converted to per byte at the end)

_INLINE_MARKUP = re.compile(
    r"(?P<img_open>!\[)"
    r"|(?P<link_open>\[)(?=[^\]]*\]\()"
    r"|(?P<link_tail>\]\([^)]*\))"
    r"|(?P<ref_tail>\]\[[^\]]*\])"
    r"|(?P<strong>\*\*|__|~~)"
    r"|(?P<em_open>(?<![\w*_])[*_](?=[^\s*_]))"
    r"|(?P<em_close>(?<=[^\s*_])[*_](?![\w*_]))"
    r"|(?P<html><[A-Za-z/!][^>\n]*>)"
    r"|(?P<escape>\\(?=[!-/:-@\[-`{-~]))"
)
_CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.S)
_WS = re.compile(r"\s")


def _inline(text: str, cls: list[str], off: int) -> None:
    """Mark inline markup inside text (cls is the class list of the whole line, off its start)."""
    protected = [False] * len(text)
    for m in _CODE_SPAN.finditer(text):
        tick = len(m.group(1))
        for i in range(m.start(), m.start() + tick):
            cls[off + i] = MARKUP
        for i in range(m.end() - tick, m.end()):
            cls[off + i] = MARKUP
        for i in range(m.start(), m.end()):
            protected[i] = True
    for m in _INLINE_MARKUP.finditer(text):
        if any(protected[m.start():m.end()]):
            continue
        for i in range(m.start(), m.end()):
            if not _WS.match(text[i]):
                cls[off + i] = MARKUP


_LIST_MARKER = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])(\s+)(\[[ xX]\]\s+)?")
_ATX = re.compile(r"^(\s{0,3})(#{1,6})(\s+|$)")
_ATX_CLOSE = re.compile(r"(\s+#+\s*)$")
_QUOTE = re.compile(r"^(\s*>\s?)+")
_SETEXT = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")


def _classify_line(line: str, kind: str, first_line: bool, in_quote: bool, state: dict) -> str:
    cls = [WHITESPACE if _WS.match(ch) else CONTENT for ch in line]
    body_start = 0
    if in_quote:
        m = _QUOTE.match(line)
        if m:
            for i in range(m.start(), m.end()):
                if not _WS.match(line[i]):
                    cls[i] = MARKUP
            body_start = m.end()
    body = line[body_start:]
    stripped = body.strip()

    if not stripped:
        return "".join(cls)

    if kind == "code_block":
        if state.get("fenced"):
            if first_line or (state.get("open") and _FENCE.match(body)):
                if not first_line:
                    state["open"] = False
                for i in range(body_start, len(line)):
                    if not _WS.match(line[i]):
                        cls[i] = MARKUP
            # code body stays content
        return "".join(cls)

    if kind == "thematic_break" or (kind == "table_row" and _TABLE_SEP.match(body)):
        return "".join(MARKUP if c == CONTENT else c for c in cls)

    if kind == "heading":
        m = _ATX.match(body)
        if m and first_line:
            for i in range(body_start + m.start(2), body_start + m.end(2)):
                cls[i] = MARKUP
            rest_start = body_start + m.end()
            mc = _ATX_CLOSE.search(line.rstrip("\n"))
            rest_end = mc.start() if mc else len(line.rstrip("\n"))
            if mc:
                for i in range(mc.start(), mc.end()):
                    if not _WS.match(line[i]):
                        cls[i] = MARKUP
            _inline(line[rest_start:rest_end], cls, rest_start)
            return "".join(cls)
        if _SETEXT.match(body) and not first_line:
            return "".join(MARKUP if c == CONTENT else c for c in cls)

    if kind == "html_block":
        for m in re.finditer(r"<[^>\n]*>", line):
            for i in range(m.start(), m.end()):
                if not _WS.match(line[i]):
                    cls[i] = MARKUP
        return "".join(cls)

    start = body_start
    if kind == "list_item" and first_line:
        m = _LIST_MARKER.match(body)
        if m:
            for i in range(body_start + m.start(2), body_start + m.end()):
                if not _WS.match(line[i]):
                    cls[i] = MARKUP
            start = body_start + m.end()

    if kind == "table_row":
        _inline(line[start:], cls, start)
        for i, ch in enumerate(line):
            if ch == "|" and cls[i] != MARKUP and (i == 0 or line[i - 1] != "\\"):
                cls[i] = MARKUP
        return "".join(cls)

    _inline(line[start:], cls, start)
    return "".join(cls)


def _to_bytes(text: str, cls: str) -> str:
    return "".join(c * len(ch.encode("utf-8")) for ch, c in zip(text, cls))


# ------------------------------------------------------------------------------------------------

def parse(src: str) -> TreeNode:
    blocks, lines = _block_tree(src)
    _nest_headings(blocks)
    _tile(blocks, len(lines))

    def convert(b: _Block) -> TreeNode:
        own_end = b.children[0].start if b.children else b.end
        own_lines = lines[b.start:own_end]
        state = {"fenced": b.attrs.get("fenced", False), "open": True}
        byte_cls = []
        for i, ln in enumerate(own_lines):
            byte_cls.append(_to_bytes(ln, _classify_line(ln, b.node_type, i == 0, b.in_blockquote, state)))
        node = TreeNode(
            node_type=b.node_type,
            text="".join(own_lines),
            byte_class="".join(byte_cls),
            title=b.title,
            splittable=b.node_type == "heading",
            attrs=dict(b.attrs),
        )
        for c in b.children:
            node.add(convert(c))
        return node

    root = convert(blocks)
    if root.subtree_text() != src:
        raise AssertionError("markdown parser lost or duplicated source text")
    return root
