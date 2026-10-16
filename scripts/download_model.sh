#!/usr/bin/env bash
# Download Llama-3.2-1B-Instruct, Q4_K_M quantization (GGUF), about 0.8 GB.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$REPO/models"
FILE="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
URL="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/$FILE"

mkdir -p "$DEST"
if [ -s "$DEST/$FILE" ]; then
    echo "$DEST/$FILE already present"
    exit 0
fi
if python3 -c "import huggingface_hub" 2>/dev/null; then
    python3 -c "from huggingface_hub import hf_hub_download as d; print(d('bartowski/Llama-3.2-1B-Instruct-GGUF', '$FILE', local_dir='$DEST'))"
else
    curl -L --fail -o "$DEST/$FILE" "$URL"
fi
ls -l "$DEST/$FILE"
