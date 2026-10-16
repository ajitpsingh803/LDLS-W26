"""Domain parsers. Each one exposes `parse(source_text) -> TreeNode` (see ldls/tree.py).

To add a domain: write ldls/parsers/<domain>.py with a `parse` function that returns the root
TreeNode, fills `text` / `byte_class` for every node so that concatenating the nodes' text in
document order reproduces the source, marks the node types that may become sequence units as
`splittable`, and register it below.
"""

from . import markdown

PARSERS = {
    "markdown": markdown.parse,
}
