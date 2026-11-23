#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-scitrera/dgx-spark-sglang:0.5.17}"
MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
PORT="${PORT:-30000}"
CONTEXT_LEN="${CONTEXT_LEN:-8192}"
MEM_FRACTION="${MEM_FRACTION:-0.70}"
MEM_GB="${MEM_GB-27}"
if [ -n "${MEM_GB:-}" ]; then
  TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
  if ! [ "$TOTAL_MIB" -gt 0 ] 2>/dev/null; then
    TOTAL_MIB="$(docker run --rm --gpus all --privileged "$IMAGE" python3 -c \
      'import torch; print(torch.cuda.mem_get_info()[1] // 2**20)' 2>/dev/null | tail -1 | tr -d ' ')"
  fi
  if ! [ "$TOTAL_MIB" -gt 0 ] 2>/dev/null; then
    TOTAL_MIB="$(awk '/MemTotal/ {printf "%d", $2/1024}' /proc/meminfo 2>/dev/null)"
  fi
  if [ -n "$TOTAL_MIB" ] && [ "$TOTAL_MIB" -gt 0 ] 2>/dev/null; then
    MEM_FRACTION="$(awk -v gb="$MEM_GB" -v tot="$TOTAL_MIB" 'BEGIN{f=gb*1024/tot; if(f>0.95)f=0.95; printf "%.3f", f}')"
    echo "[serve-docker] MEM_GB=$MEM_GB of ${TOTAL_MIB}MiB → mem_fraction=$MEM_FRACTION"
  else
    echo "[serve-docker] WARNING: could not read total GPU memory, ignoring MEM_GB (using MEM_FRACTION=$MEM_FRACTION)"
  fi
fi
MAX_RUNNING="${MAX_RUNNING:-32}"
ATTN_BACKEND="${ATTN_BACKEND:-triton}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$HF_CACHE"
echo "[serve-docker] image=$IMAGE  model=$MODEL  port=$PORT  attn=$ATTN_BACKEND"
echo "[serve-docker] ctx=$CONTEXT_LEN  mem_fraction=$MEM_FRACTION  max_running=$MAX_RUNNING"
echo "[serve-docker] clients: export SGLANG_BASE_URL=http://127.0.0.1:$PORT/v1"
if [ -n "${DRY_RUN:-}" ]; then
  if [ -n "${TOTAL_MIB:-}" ] && [ "$TOTAL_MIB" -gt 0 ] 2>/dev/null; then
    echo "[serve-docker] DRY_RUN: detected ${TOTAL_MIB}MiB total memory → would allocate $(awk -v f="$MEM_FRACTION" -v t="$TOTAL_MIB" 'BEGIN{printf "%.1f", f*t/1024}')GB"
  fi
  echo "[serve-docker] DRY_RUN: exiting, server NOT started."
  exit 0
fi

exec docker run --rm -it \
  --gpus all --privileged \
  --network host --ipc=host \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --context-length "$CONTEXT_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "$MAX_RUNNING" \
    --attention-backend "$ATTN_BACKEND" \
    --trust-remote-code \
    $EXTRA_ARGS
