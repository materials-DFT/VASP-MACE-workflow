#!/usr/bin/env bash
set -euo pipefail

# Rebuild (or create) the nequip environment for:
# 1) safe .ckpt -> .nequip.pt2 conversion via convert_ckpt_to_pt2.py
# 2) LAMMPS runtime compatibility with pt2 models on this cluster stack
#
# Usage:
#   bash setup_nequip_env_for_pt2_lammps.sh
#   ENV_NAME=nequip bash setup_nequip_env_for_pt2_lammps.sh

ENV_NAME="${ENV_NAME:-nequip}"
CONDA_ROOT="${CONDA_ROOT:-/Users/924322630/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

echo "Rebuilding conda env: ${ENV_NAME}"
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda create -y -n "${ENV_NAME}" python=3.11 pip

conda activate "${ENV_NAME}"

python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.6.0+cu118" \
  "torchvision==0.21.0+cu118" \
  "torchaudio==2.6.0+cu118"

python -m pip install --no-cache-dir \
  "nequip==0.16.2" \
  "nequip-allegro==0.8.1"

echo
echo "Environment ready:"
python - <<'PY'
import sys
import importlib.metadata as m
import torch, triton, nequip
print("python :", sys.version.split()[0])
print("torch  :", torch.__version__, "cuda", torch.version.cuda)
print("triton :", triton.__version__)
print("nequip :", nequip.__version__)
print("allegro:", m.version("nequip-allegro"))
PY

echo
echo "Use this converter for LAMMPS-compatible pt2 export:"
echo "  python ~/VASP-MACE-workflow/MLFF/ALLEGRO_scripts/convert_ckpt_to_pt2.py /path/to/best.ckpt"
