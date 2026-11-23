#!/usr/bin/env bash
set -euo pipefail

if [ "$(uname -m)" = "aarch64" ] && [ "${FORCE_NATIVE:-0}" != "1" ]; then
  echo "[install] aarch64 detected."
  echo "          Do NOT use this script here. Serve with:  bash scripts/serve_docker.sh"
  echo "          and install only the client:            bash scripts/install_docker.sh"
  echo "          (set FORCE_NATIVE=1 to bypass this check)"
  exit 2
fi

ENV_NAME="${1:-sglang}"
PY_VER="${PY_VER:-3.11}"
CONDA_BASE="$(conda info --base)"
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"

if [ ! -x "$PY" ]; then
  echo "[install] creating conda env $ENV_NAME (python $PY_VER)"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

PKGS=("sglang[all]" openai "huggingface_hub[cli]" ninja)
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$PY" "${PKGS[@]}"
else
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install "${PKGS[@]}"
fi

SM="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)"
if [ -n "$SM" ] && [ "$SM" -lt 80 ]; then
  echo "[install] sm$SM < 80 → installing GCC 13 toolchain into env for JIT kernels"
  conda install -y -n "$ENV_NAME" -c conda-forge "gxx_linux-64=13" "gcc_linux-64=13"
  if ! ls /usr/local/cuda-12*/bin/nvcc /usr/local/cuda/bin/nvcc >/dev/null 2>&1; then
    echo "[install] WARNING: no CUDA 12 toolkit (nvcc) found under /usr/local — JIT kernels"
    echo "          will be skipped and PyTorch fallbacks used (works, but slower)."
  fi
fi

echo
echo "[install] versions:"
"$PY" - <<'PYEOF'
import torch, sglang, openai
print("  torch  ", torch.__version__, "| cuda", torch.version.cuda, "| gpu ok:", torch.cuda.is_available())
print("  sglang ", sglang.__version__)
print("  openai ", openai.__version__)
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print("  device ", torch.cuda.get_device_name(0), "| sm%d%d" % cc)
    if cc < (8, 0):
        print("  NOTE: sm < 80 → serve_sglang.sh will use scripts/sglang_turing_launch.py automatically")
PYEOF
echo
echo "[install] done. Next: bash scripts/x86_cuda/serve_sglang.sh   (in a tmux window)"
echo "          optional check on sm<80: python scripts/x86_cuda/sglang_turing_probe.py"
