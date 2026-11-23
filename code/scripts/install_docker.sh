#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-scitrera/dgx-spark-sglang:0.5.17}"
MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[install] 1/4 sanity"
[ "$(uname -m)" = "aarch64" ] || echo "  WARNING: not aarch64 ($(uname -m)) — this script is meant for aarch64 servers; x86 servers use scripts/x86_cuda/"
command -v docker >/dev/null || { echo "  docker not found — install docker + nvidia-container-toolkit first"; exit 1; }
if ! docker info >/dev/null 2>&1; then echo "  docker daemon not reachable (add user to 'docker' group or run with sudo)"; exit 1; fi
docker info 2>/dev/null | grep -qi nvidia || echo "  WARNING: nvidia runtime not listed in 'docker info' — check nvidia-container-toolkit"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || echo "  WARNING: nvidia-smi failed"

echo "[install] 2/4 client deps (openai, scipy)"
PIP="${PIP:-python3 -m pip}"
$PIP install -q -r "$HERE/../requirements.txt"

echo "[install] 3/4 docker pull $IMAGE"
docker pull "$IMAGE"

echo "[install] 4/4 pre-download $MODEL into $HF_CACHE"
mkdir -p "$HF_CACHE"
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/hf -e HOME=/tmp \
  -v "$HF_CACHE:/hf" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  "$IMAGE" \
  python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
if find "$HF_CACHE" -maxdepth 3 -user root -print -quit 2>/dev/null | grep -q .; then
  echo "  NOTE: some files in $HF_CACHE are owned by root (created by an earlier container run)."
  echo "        fix once with:  sudo chown -R \"$(id -u):$(id -g)\" \"$HF_CACHE\""
fi

echo
echo "[install] done. Next:"
echo "   tmux new -s sglang"
echo "   bash scripts/serve_docker.sh                       # MODEL=$MODEL on :30000"
echo "   python scripts/sglang_smoke_test.py --model sglang-qwen3-32b"
