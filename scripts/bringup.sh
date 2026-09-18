#!/usr/bin/env bash
# Sol-H3 on ONE H100 (p5.4xlarge): environment only. No render here.
#
#   bash bringup.sh            # venv + wheels + adapter, ~15 min, idempotent
#
# WHY A FRESH VENV AND NOT THE SGLANG IMAGE ALREADY ON THIS BOX. Sol-H3 pins its own stack:
# torch 2.10.0+cu130, transformers 5.8.1, peft 0.20.0 and diffusers at commit abc5e9bf, which is
# the revision that carries the MiniMax-H3 *modular* pipeline the runtime imports by name
# (diffusers.modular_pipelines.minimax_h3). lmsysorg/sglang:dev carries its own diffusers and its
# own torch, and Sol-H3 monkey-patches diffusers internals (before_encoder.MINIMAX_H3_MAX_DURATION,
# resolve_reference_image_size), so a different revision is a different set of names.
#
# THE WEIGHTS ARE ALREADY HERE. /opt/dlami/nvme/vdn/hf/hub holds MiniMaxAI/MiniMax-H3 (268 GiB,
# both the t2v `transformer/` and the `transformer_ref/` partition), pulled for the sglang work on
# this same box. HF_HOME points at it, so nothing re-downloads. The only new artifact is the
# FastH3 four-step LoRA, which is what makes Sol-H3 a four-forward profile instead of a 49-forward
# one, and it is small.
#
# PYTHON 3.12, NOT THE DLAMI'S 3.14. The requirements pin triton 3.6.0 and torch 2.10+cu130; the
# cu130 wheel index does not publish cp314 for this release. uv fetches its own 3.12.
set -uo pipefail

ROOT=${ROOT:-/opt/dlami/nvme/sol}
SRC=${SRC:-$ROOT/Sol-H3}
VENV=${VENV:-$ROOT/venv}
export HF_HOME=${HF_HOME:-/opt/dlami/nvme/vdn/hf}
LOG=${LOG:-$ROOT/logs/bringup.log}

mkdir -p "$ROOT/logs" "$ROOT/out"
exec > >(tee -a "$LOG") 2>&1
echo "=== bringup $(date -u +%FT%TZ) ROOT=$ROOT ==="

step() { echo; echo "--- $* ---"; }

step "uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version || exit 1

step "venv (python 3.12)"
[ -d "$VENV" ] || uv venv --python 3.12 "$VENV" || exit 1
PY="$VENV/bin/python"
$PY -V

step "torch 2.10.0+cu130"
if ! $PY -c "import torch,sys; sys.exit(0 if torch.__version__.startswith('2.10') else 1)" 2>/dev/null; then
  uv pip install --python "$PY" --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.10.0+cu130 torchvision==0.25.0+cu130 torchaudio==2.10.0+cu130 || exit 1
fi
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# The dense single-GPU profile never calls cuDNN block-sparse attention, and the cudnn-frontend
# build from git is the slowest and least certain item in the file, so it is installed LAST and its
# failure is not fatal: `--attention-backend dense` is the only backend one GPU is allowed anyway
# (engine.py refuses sol/sol_bsa below two processes).
step "requirements, minus cudnn-frontend"
grep -v cudnn-frontend "$SRC/requirements.txt" > /tmp/req.core.txt
cat /tmp/req.core.txt
uv pip install --python "$PY" -r /tmp/req.core.txt || exit 1

step "import check"
$PY - <<'EOF'
import torch, diffusers, transformers, peft
print("torch", torch.__version__, "| diffusers", diffusers.__version__,
      "| transformers", transformers.__version__, "| peft", peft.__version__)
from diffusers.modular_pipelines import minimax_h3
print("modular minimax_h3 OK:", [n for n in dir(minimax_h3) if "MiniMax" in n][:8])
print("device", torch.cuda.get_device_name(0), "sm", torch.cuda.get_device_capability())
EOF

step "FastH3 four-step LoRA"
$PY - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA",
                      allow_patterns=["dense-datafree/*"])
print("adapter root:", p)
import os
for d, _, fs in os.walk(p):
    for f in fs:
        q = os.path.join(d, f)
        print(f"  {os.path.getsize(q)/2**20:9.1f} MiB  {os.path.relpath(q, p)}")
EOF

step "cudnn-frontend (optional; sol_bsa only, needs >=2 GPUs anyway)"
uv pip install --python "$PY" \
  "nvidia-cudnn-frontend[cutedsl] @ git+https://github.com/NVIDIA/cudnn-frontend.git@29106622617bfd9031a53099a6fbbc5e74a474e9" \
  && $PY -c "from cudnn import BSA; print('cudnn BSA import OK')" \
  || echo "NOTE: cudnn-frontend unavailable; dense backend unaffected"

echo; echo "=== bringup done $(date -u +%FT%TZ) ==="
