#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
PORT="${PORT:-30000}"
HOST="${HOST:-127.0.0.1}"
ENV_NAME="${ENV_NAME:-sglang}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_RUNNING="${MAX_RUNNING:-64}"
TP="${TP:-1}"
DTYPE="${DTYPE:-auto}"
ATTN_BACKEND="${ATTN_BACKEND:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$(conda info --base)"
ENV_BIN="$CONDA_BASE/envs/$ENV_NAME/bin"
PY="$ENV_BIN/python"
[ -x "$PY" ] || { echo "conda env '$ENV_NAME' not found — run scripts/x86_cuda/install_sglang.sh first"; exit 1; }
export PATH="$ENV_BIN:$PATH"

SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
SM="${SM:-80}"

if [ "$DTYPE" = "auto" ]; then
  if [ "$SM" -lt 80 ]; then DTYPE=float16; else DTYPE=bfloat16; fi
fi
if [ "$ATTN_BACKEND" = "auto" ]; then
  if [ "$SM" -lt 80 ]; then
    ATTN_BACKEND=triton
  elif [ "$SM" -ge 120 ]; then
    ATTN_BACKEND=triton
  else
    ATTN_BACKEND=flashinfer
  fi
fi

LAUNCHER=(-m sglang.launch_server)
COMPAT_ARGS=()
if [ "$SM" -lt 80 ]; then
  LAUNCHER=("$HERE/sglang_turing_launch.py")
  COMPAT_ARGS=(--sampling-backend pytorch)
  if [ -z "${CUDA_HOME:-}" ]; then
    for d in /usr/local/cuda-12.9 /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
      if [ -x "$d/bin/nvcc" ] && "$d/bin/nvcc" --version | grep -q "release 12"; then CUDA_HOME="$d"; break; fi
    done
  fi
  if [ -n "${CUDA_HOME:-}" ]; then
    export CUDA_HOME; export PATH="$CUDA_HOME/bin:$PATH"
  else
    echo "[serve] WARNING: no CUDA 12 toolkit found — JIT norm/rope will fail and fall back to PyTorch ops (slower)"
  fi
  if [ -x "$ENV_BIN/x86_64-conda-linux-gnu-g++" ]; then
    export CXX="$ENV_BIN/x86_64-conda-linux-gnu-g++"
    export NVCC_PREPEND_FLAGS="-ccbin $CXX ${NVCC_PREPEND_FLAGS:-}"
  fi
  echo "[serve] sm$SM < 80 → compat launcher (JIT norm/rope, Triton AWQ, PyTorch sampling)"
  echo "[serve] CUDA_HOME=${CUDA_HOME:-<none>}  CXX=${CXX:-<system>}"
fi

echo "[serve] model=$MODEL  port=$PORT  sm=$SM  dtype=$DTYPE  attn=$ATTN_BACKEND"
echo "[serve] ctx=$CONTEXT_LEN  mem_fraction=$MEM_FRACTION  max_running=$MAX_RUNNING  tp=$TP"
echo "[serve] clients should set: export SGLANG_BASE_URL=http://$HOST:$PORT/v1"

exec "$PY" "${LAUNCHER[@]}" \
  --model-path "$MODEL" \
  --host "$HOST" --port "$PORT" \
  --dtype "$DTYPE" \
  --attention-backend "$ATTN_BACKEND" \
  --context-length "$CONTEXT_LEN" \
  --mem-fraction-static "$MEM_FRACTION" \
  --max-running-requests "$MAX_RUNNING" \
  --tp "$TP" \
  --trust-remote-code \
  "${COMPAT_ARGS[@]}" \
  $EXTRA_ARGS
