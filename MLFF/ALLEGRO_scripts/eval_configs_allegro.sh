#!/bin/bash
#SBATCH --job-name=Allegro_eval_oms6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpucluster
#SBATCH --time=01:00:00
#SBATCH --chdir=.
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Run ALLEGRO evaluation on OMS-6 frames on a compute-node GPU (same idea as MACE evaluation).

set -e

# Run from the directory where sbatch was executed (evaluation_oms6_1k)
WORK_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$WORK_DIR"

# Conda + nequip
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate nequip

# Optional CUDA env (match your training node)
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "=== ALLEGRO OMS-6 evaluation on compute node ==="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-none}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-auto}"
python --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# Input configs: auto-detect a single .xyz in submit dir, or set ALLEGRO_CONFIGS
if [ -n "${ALLEGRO_CONFIGS:-}" ]; then
  FINAL_FRAMES="$ALLEGRO_CONFIGS"
  if [[ "$FINAL_FRAMES" != /* ]]; then
    FINAL_FRAMES="$WORK_DIR/$FINAL_FRAMES"
  fi
  if [ ! -f "$FINAL_FRAMES" ]; then
    echo "Error: ALLEGRO_CONFIGS points to missing file: $FINAL_FRAMES"
    exit 1
  fi
else
  shopt -s nullglob
  xyz_files=("$WORK_DIR"/*.xyz)
  shopt -u nullglob

  if [ "${#xyz_files[@]}" -eq 0 ]; then
    echo "Error: No .xyz file found in $WORK_DIR"
    echo "Hint: place exactly one .xyz there or set ALLEGRO_CONFIGS=/path/to/configs.xyz"
    exit 1
  elif [ "${#xyz_files[@]}" -gt 1 ]; then
    echo "Error: Multiple .xyz files found in $WORK_DIR. Please set ALLEGRO_CONFIGS explicitly."
    printf '  %s\n' "${xyz_files[@]}"
    exit 1
  fi

  FINAL_FRAMES="${xyz_files[0]}"
fi

# Checkpoint: best.ckpt or last.ckpt from parent models/, or pass via sbatch --export
MODEL="${ALLEGRO_CKPT:-}"
if [ -z "$MODEL" ]; then
  for d in "$WORK_DIR/../models" "$WORK_DIR/../../models" "$WORK_DIR/../../../models"; do
    [ -f "$d/best.ckpt" ] && MODEL="$d/best.ckpt" && break
  done
  [ -z "$MODEL" ] && for d in "$WORK_DIR/../models" "$WORK_DIR/../../models" "$WORK_DIR/../../../models"; do
    [ -f "$d/last.ckpt" ] && MODEL="$d/last.ckpt" && break
  done
fi
if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "Error: No checkpoint found. Set ALLEGRO_CKPT or put best.ckpt/last.ckpt in ../models"
  exit 1
fi

echo "Configs: $FINAL_FRAMES"
echo "Model: $MODEL"

# Match MACE behavior: only evaluate stress if configs include stress
if grep -q 'stress=' "$FINAL_FRAMES"; then
  echo "Detected stress data in $FINAL_FRAMES - enabling stress evaluation"
  COMPUTE_STRESS=1
else
  echo "No stress data in $FINAL_FRAMES - skipping stress evaluation"
  COMPUTE_STRESS=0
fi

# Run embedded Python evaluation script
python << PYTHON_SCRIPT 2>&1 | tee eval_output.log
import numpy as np
from pathlib import Path

import ase.io
from nequip.ase import NequIPCalculator

# Get arguments from shell variables
configs_path = Path("$FINAL_FRAMES")
model_path = Path("$MODEL")
output_path = Path("output.xyz")
device = "cuda"
compute_stress = bool(int("$COMPUTE_STRESS"))

if not configs_path.exists():
    raise FileNotFoundError(f"Configs file not found: {configs_path}")
if not model_path.exists():
    raise FileNotFoundError(f"Model file not found: {model_path}")

print(f"Using configs file: {configs_path}")
print(f"Using model file: {model_path}")
print(f"Compute stress: {compute_stress}")

def _normalize_stress(stress_like):
    """Normalize stress to 3x3 matrix for stable extxyz writing."""
    if stress_like is None:
        return None
    s = np.asarray(stress_like, dtype=float)
    if s.shape == (3, 3):
        return s
    if s.shape == (6,):
        from ase.stress import voigt_6_to_full_3x3_stress
        return voigt_6_to_full_3x3_stress(s)
    return None

# Load calculator from checkpoint
calc = NequIPCalculator._from_saved_model(
    str(model_path),
    device=device,
    chemical_species_to_atom_type_map=True,
)

# Load all frames
frames = ase.io.read(str(configs_path), index=":")
n_frames = len(frames)
print(f"Loaded {n_frames} frames")

# Collect reference values if present
ref_energies = []
ref_forces_list = []
pred_energies = []
pred_forces_list = []
ref_stresses_list = []
pred_stresses_list = []

for i, at in enumerate(frames):
    # Extract reference from calculator BEFORE replacing (input has energy/forces in calc.results)
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
    # Predict
    pred_e = at.get_potential_energy()
    pred_f = at.get_forces()
    pred_energies.append(pred_e)
    pred_forces_list.append(pred_f.copy())

    # Attach ALLEGRO predictions to atoms for writing
    at.info["ALLEGRO_energy"] = pred_e
    at.arrays["ALLEGRO_forces"] = pred_f

    if compute_stress:
        try:
            pred_s = _normalize_stress(at.get_stress())
        except Exception:
            pred_s = None
        if pred_s is not None:
            pred_stresses_list.append(pred_s.copy())
            at.info["ALLEGRO_stress"] = pred_s

    if (i + 1) % 100 == 0 or i == 0:
        print(f"  Frame {i + 1}/{n_frames}")

# Write output XYZ (ASE will write info and arrays)
# write_results=False: all frames share one calculator whose results are from the last frame;
# copying those into earlier frames would cause shape mismatch (e.g. magmoms (13,) into 67-atom frame)
print("Writing output file...")
with open(str(output_path), "w") as f:
    for i, at in enumerate(frames):
        ase.io.write(f, [at], format="extxyz", write_results=False)
        if (i + 1) % 100 == 0:
            print(f"  Wrote frame {i + 1}/{n_frames}")
print(f"Wrote {output_path}")

# Metrics vs reference
if ref_energies and ref_forces_list:
    ref_e = np.array(ref_energies)
    pred_e = np.array(pred_energies)
    mae_e = np.mean(np.abs(pred_e - ref_e))
    rmse_e = np.sqrt(np.mean((pred_e - ref_e) ** 2))
    mae_e_per_atom = mae_e / np.mean([len(f) for f in frames])
    rmse_e_per_atom = rmse_e / np.mean([len(f) for f in frames])

    ref_f = np.concatenate(ref_forces_list, axis=0)
    pred_f = np.concatenate(pred_forces_list, axis=0)
    mae_f = np.mean(np.abs(pred_f - ref_f))
    rmse_f = np.sqrt(np.mean((pred_f - ref_f) ** 2))

    print("ALLEGRO evaluation completed")
    print(f"  Energy  MAE = {mae_e:.6f} eV  (per atom: {mae_e_per_atom:.6f})")
    print(f"  Energy  RMSE = {rmse_e:.6f} eV  (per atom: {rmse_e_per_atom:.6f})")
    print(f"  Forces  MAE = {mae_f:.6f} eV/Å")
    print(f"  Forces  RMSE = {rmse_f:.6f} eV/Å")
    if compute_stress and ref_stresses_list and pred_stresses_list:
        n_stress = min(len(ref_stresses_list), len(pred_stresses_list))
        ref_s = np.array(ref_stresses_list[:n_stress]).reshape(n_stress, 9)
        pred_s = np.array(pred_stresses_list[:n_stress]).reshape(n_stress, 9)
        mae_s = np.mean(np.abs(pred_s - ref_s))
        rmse_s = np.sqrt(np.mean((pred_s - ref_s) ** 2))
        print(f"  Stress  MAE = {mae_s:.6f} eV/Å^3")
        print(f"  Stress  RMSE = {rmse_s:.6f} eV/Å^3")
    elif compute_stress:
        print("  Stress metrics unavailable (missing REF_stress or ALLEGRO_stress in some/all frames)")
else:
    print("ALLEGRO evaluation completed (no REF_energy/REF_forces in configs for metrics)")
PYTHON_SCRIPT

echo "Done. Check output.xyz and eval_output.log"
