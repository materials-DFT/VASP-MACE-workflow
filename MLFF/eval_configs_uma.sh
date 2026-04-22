#!/bin/bash
#SBATCH --job-name=UMA_Eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=eval_output.log
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1

# Evaluate configs with UMA checkpoint:
#   ~/lammps_md/uma/models/uma-m-1p1.pt
# Usage is intentionally similar to eval_configs.sh:
#   sbatch eval_configs_uma.sh

set -euo pipefail

export PATH="$HOME/miniconda3/bin:$PATH"
set +u
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
set -u

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

usage() {
  cat <<'USAGE'
Usage:
  sbatch eval_configs_uma.sh

From the directory where you submit:
  - Put exactly one .xyz file (structures to evaluate), or pass --configs FILE.
  - Optional: --model FILE to override default UMA checkpoint.
  - Optional: --task omat|omol|oc20|odac|omc (default: omat).
  - Optional: --device cuda|cpu (default: cuda).
USAGE
}

CONFIGS_FILE=""
MODEL_FILE="${HOME}/lammps_md/uma/models/uma-m-1p1.pt"
TASK_NAME="${UMA_TASK_NAME:-omat}"
DEVICE="${UMA_DEVICE:-cuda}"
NUM_WORKERS="${UMA_NUM_WORKERS:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--configs|--config)
      [[ $# -ge 2 ]] || { echo "Error: --configs requires a value"; exit 2; }
      CONFIGS_FILE="$2"; shift 2;;
    -m|--model)
      [[ $# -ge 2 ]] || { echo "Error: --model requires a value"; exit 2; }
      MODEL_FILE="$2"; shift 2;;
    --task)
      [[ $# -ge 2 ]] || { echo "Error: --task requires a value"; exit 2; }
      TASK_NAME="$2"; shift 2;;
    --device)
      [[ $# -ge 2 ]] || { echo "Error: --device requires a value"; exit 2; }
      DEVICE="$2"; shift 2;;
    --workers)
      [[ $# -ge 2 ]] || { echo "Error: --workers requires a value"; exit 2; }
      NUM_WORKERS="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      shift;;
  esac
done

if [[ -z "${CONFIGS_FILE}" ]]; then
  shopt -s nullglob
  ALL_XYZ=( ./*.xyz )
  shopt -u nullglob
  XYZ_FILES=()
  for f in "${ALL_XYZ[@]}"; do
    [[ "$(basename "$f")" == "output.xyz" ]] && continue
    XYZ_FILES+=( "$f" )
  done
  if [[ ${#XYZ_FILES[@]} -eq 0 ]]; then
    echo "Error: No .xyz file in current directory. Put exactly one .xyz here."
    exit 1
  elif [[ ${#XYZ_FILES[@]} -gt 1 ]]; then
    echo "Error: Multiple .xyz files found. Keep only one or use --configs FILE:"
    printf '  %s\n' "${XYZ_FILES[@]}"
    exit 1
  else
    CONFIGS_FILE="${XYZ_FILES[0]}"
  fi
fi

if [[ ! -f "$CONFIGS_FILE" ]]; then
  echo "Error: Configs file not found: $CONFIGS_FILE"
  exit 1
fi
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Error: Model file not found: $MODEL_FILE"
  exit 1
fi

echo "Using configs file: $CONFIGS_FILE"
echo "Using UMA model file: $MODEL_FILE"
echo "Using UMA task: $TASK_NAME"
echo "Using device: $DEVICE"
echo "Using workers: $NUM_WORKERS"

if grep -q 'stress=' "$CONFIGS_FILE"; then
  echo "Detected stress data in $CONFIGS_FILE"
  COMPUTE_STRESS=1
else
  echo "No stress data in $CONFIGS_FILE"
  COMPUTE_STRESS=0
fi

if [[ "$(basename "$CONFIGS_FILE")" != "output.xyz" ]] && [[ -f output.xyz ]]; then
  echo "Removing existing output.xyz before writing new results."
  rm -f output.xyz
fi

export FINAL_FRAMES="$CONFIGS_FILE"
export MODEL="$MODEL_FILE"
export TASK_NAME
export DEVICE
export NUM_WORKERS
export COMPUTE_STRESS

python <<'PYTHON_SCRIPT'
import os
from pathlib import Path

import ase.io
import numpy as np
from fairchem.core.calculate import FAIRChemCalculator

configs_path = Path(os.environ["FINAL_FRAMES"])
model_path = Path(os.environ["MODEL"])
task_name = os.environ.get("TASK_NAME", "omat")
device = os.environ.get("DEVICE", "cuda")
workers = int(os.environ.get("NUM_WORKERS", "1"))
compute_stress = bool(int(os.environ.get("COMPUTE_STRESS", "0")))
output_path = Path("output.xyz")

if not configs_path.exists():
    raise FileNotFoundError(f"Configs file not found: {configs_path}")
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"Using configs file: {configs_path}")
print(f"Using model file: {model_path}")
print(f"Task: {task_name} | Device: {device} | Workers: {workers}")
print(f"Compute stress: {compute_stress}")


def _normalize_stress(stress_like):
    if stress_like is None:
        return None
    s = np.asarray(stress_like, dtype=float)
    if s.shape == (3, 3):
        return s
    if s.shape == (6,):
        from ase.stress import voigt_6_to_full_3x3_stress

        return voigt_6_to_full_3x3_stress(s)
    return None


calc = FAIRChemCalculator.from_model_checkpoint(
    str(model_path),
    task_name=task_name,
    device=device,
    workers=workers,
)

frames = ase.io.read(str(configs_path), index=":")
n_frames = len(frames)
print(f"Loaded {n_frames} frames")

ref_energies = []
ref_forces_list = []
pred_energies = []
pred_forces_list = []
ref_stresses_list = []
pred_stresses_list = []

for i, at in enumerate(frames):
    if at.calc is not None and hasattr(at.calc, "results"):
        if "energy" in at.calc.results:
            ref_e = float(at.calc.results["energy"])
            ref_energies.append(ref_e)
            at.info["REF_energy"] = ref_e
        if "forces" in at.calc.results:
            ref_f = np.asarray(at.calc.results["forces"]).copy()
            ref_forces_list.append(ref_f)
            at.arrays["REF_forces"] = ref_f
        if compute_stress:
            ref_s = _normalize_stress(at.calc.results.get("stress"))
            if ref_s is None:
                ref_s = _normalize_stress(at.info.get("stress"))
            if ref_s is not None:
                ref_stresses_list.append(ref_s.copy())
                at.info["REF_stress"] = ref_s

    at.calc = calc
    pred_e = at.get_potential_energy()
    pred_f = at.get_forces()
    pred_energies.append(pred_e)
    pred_forces_list.append(pred_f.copy())
    at.info["UMA_energy"] = pred_e
    at.arrays["UMA_forces"] = pred_f

    if compute_stress:
        try:
            pred_s = _normalize_stress(at.get_stress())
        except Exception:
            pred_s = None
        if pred_s is not None:
            pred_stresses_list.append(pred_s.copy())
            at.info["UMA_stress"] = pred_s

    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Frame {i + 1}/{n_frames}")

print("Writing output file...")
with open(str(output_path), "w") as f:
    for i, at in enumerate(frames):
        ase.io.write(f, [at], format="extxyz", write_results=False)
        if (i + 1) % 100 == 0:
            print(f"  Wrote frame {i + 1}/{n_frames}")
print(f"Wrote {output_path}")

if ref_energies and ref_forces_list:
    ref_e = np.array(ref_energies)
    pred_e = np.array(pred_energies)
    mae_e = np.mean(np.abs(pred_e - ref_e))
    rmse_e = np.sqrt(np.mean((pred_e - ref_e) ** 2))
    n_atoms = np.mean([len(f) for f in frames])
    mae_e_per_atom = mae_e / n_atoms
    rmse_e_per_atom = rmse_e / n_atoms
    ref_f = np.concatenate(ref_forces_list, axis=0)
    pred_f = np.concatenate(pred_forces_list, axis=0)
    mae_f = np.mean(np.abs(pred_f - ref_f))
    rmse_f = np.sqrt(np.mean((pred_f - ref_f) ** 2))
    print("UMA evaluation completed")
    print(f"  Energy  MAE = {mae_e:.6f} eV  (per atom: {mae_e_per_atom:.6f})")
    print(f"  Energy  RMSE = {rmse_e:.6f} eV  (per atom: {rmse_e_per_atom:.6f})")
    print(f"  Forces  MAE = {mae_f:.6f} eV/A")
    print(f"  Forces  RMSE = {rmse_f:.6f} eV/A")
    if compute_stress and ref_stresses_list and pred_stresses_list:
        n_stress = min(len(ref_stresses_list), len(pred_stresses_list))
        ref_s = np.array(ref_stresses_list[:n_stress]).reshape(n_stress, 9)
        pred_s = np.array(pred_stresses_list[:n_stress]).reshape(n_stress, 9)
        mae_s = np.mean(np.abs(pred_s - ref_s))
        rmse_s = np.sqrt(np.mean((pred_s - ref_s) ** 2))
        print(f"  Stress  MAE = {mae_s:.6f} eV/A^3")
        print(f"  Stress  RMSE = {rmse_s:.6f} eV/A^3")
    elif compute_stress:
        print("  Stress metrics unavailable (missing REF_stress or UMA_stress in some/all frames)")
else:
    print("UMA evaluation completed (no REF_energy/REF_forces in configs for metrics)")
PYTHON_SCRIPT

echo "Done. Check output.xyz and eval_output.log"
