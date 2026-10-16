"""Client for build/bin/ldls_tokenize, i.e. the exact llama.cpp tokenizer of the model."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from ldls import DEFAULT_MODEL, DEFAULT_TOKENIZER_BIN
else:
    from . import DEFAULT_MODEL, DEFAULT_TOKENIZER_BIN


class LlamaCppTokenizer:
    def __init__(self, model: Path | str = DEFAULT_MODEL, binary: Path | str = DEFAULT_TOKENIZER_BIN):
        self.model = Path(model)
        self.binary = Path(binary)
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary} not found; build the project first (scripts/build.sh)")
        if not self.model.exists():
            raise FileNotFoundError(f"{self.model} not found; run scripts/download_model.sh")
        self.info: dict = {}

    def tokenize_many(self, texts: list[str], parse_special: bool = False) -> list[tuple[list[int], list[int]]]:
        """Returns (token ids, detokenized byte length of every token) for each text."""
        req = json.dumps({"texts": texts, "parse_special": parse_special})
        proc = subprocess.run([str(self.binary), "-m", str(self.model)], input=req, capture_output=True,
                              text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ldls_tokenize failed: {proc.stderr.strip()}")
        res = json.loads(proc.stdout)
        self.info = {k: res[k] for k in ("vocab_size", "bos", "eos", "eot")}
        return list(zip(res["tokens"], res["piece_bytes"]))

    def tokenize(self, text: str, parse_special: bool = False) -> list[int]:
        return self.tokenize_many([text], parse_special)[0][0]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Tokenize text with the project's llama.cpp model.")
    parser.add_argument("text", nargs="?", help="Text to tokenize; omit to read standard input")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--binary", type=Path, default=DEFAULT_TOKENIZER_BIN)
    parser.add_argument("--parse-special", action="store_true")
    args = parser.parse_args()
    if args.text is None and sys.stdin.isatty():
        parser.print_help()
        return 0
    text = args.text if args.text is not None else sys.stdin.read()
    tokenizer = LlamaCppTokenizer(args.model, args.binary)
    tokens = tokenizer.tokenize(text, parse_special=args.parse_special)
    print(json.dumps({"text": text, "tokens": tokens, "count": len(tokens)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
