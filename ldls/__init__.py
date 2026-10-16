"""Structured-document tooling for the efficient-attention project (week 1).

Domain parsers turn a document into a tree of `TreeNode`s; the domain-agnostic code in this
package tokenizes it with the model's tokenizer, assigns sequence ids and positions
(`layout.py`), and writes the shared JSON schema (`schema/structured_doc.schema.json`).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "models" / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DEFAULT_TOKENIZER_BIN = REPO_ROOT / "build" / "bin" / "ldls_tokenize"
DEFAULT_DRIVER_BIN = REPO_ROOT / "build" / "bin" / "ldls_driver"
SCHEMA_VERSION = "1.0"
