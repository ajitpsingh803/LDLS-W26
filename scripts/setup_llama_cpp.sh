#!/usr/bin/env bash
# Clone llama.cpp at the pinned commit and build its CPU-only tools (llama-cli, llama-batched, ...).
# The project's own driver is built separately by scripts/build.sh (it compiles libllama itself).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$REPO/third_party/llama.cpp"
LLAMA_COMMIT="${LLAMA_COMMIT:-318a471a6295e53bfce7ca96ae0cf26071cbdd6e}"
JOBS="${JOBS:-16}"

if [ ! -d "$LLAMA_DIR/.git" ]; then
    mkdir -p "$LLAMA_DIR"
    git -C "$LLAMA_DIR" init -q
    git -C "$LLAMA_DIR" remote add origin https://github.com/ggml-org/llama.cpp
fi
if [ "$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || true)" != "$LLAMA_COMMIT" ]; then
    git -C "$LLAMA_DIR" fetch -q --depth 1 origin "$LLAMA_COMMIT"
    git -C "$LLAMA_DIR" checkout -q FETCH_HEAD
fi
echo "llama.cpp at $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"

# CPU only. On x86 CPUs that report AMX we build without the AMX kernels: on the course machine
# (Linux 5.15) they crash with SIGILL ("tilezero") once prompts reach a few hundred tokens.
CPU_FLAGS=()
if [ -r /proc/cpuinfo ] && grep -qw amx_tile /proc/cpuinfo; then
    f="$(grep -m1 '^flags' /proc/cpuinfo)"
    has() { echo "$f" | grep -qw "$1"; }
    CPU_FLAGS+=(-DGGML_NATIVE=OFF -DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON)
    has avx_vnni    && CPU_FLAGS+=(-DGGML_AVX_VNNI=ON)
    has avx512f     && CPU_FLAGS+=(-DGGML_AVX512=ON)
    has avx512vbmi  && CPU_FLAGS+=(-DGGML_AVX512_VBMI=ON)
    has avx512_vnni && CPU_FLAGS+=(-DGGML_AVX512_VNNI=ON)
    has avx512_bf16 && CPU_FLAGS+=(-DGGML_AVX512_BF16=ON)
    echo "AMX detected: building without AMX kernels (${CPU_FLAGS[*]})"
fi

cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
      "${CPU_FLAGS[@]}" > "$LLAMA_DIR/build-configure.log"
cmake --build "$LLAMA_DIR/build" --config Release -j "$JOBS" --target llama-cli llama-completion llama-batched llama-tokenize \
      > "$LLAMA_DIR/build.log"
ls "$LLAMA_DIR/build/bin" | grep -E "^llama-(cli|completion|batched|tokenize)$"
