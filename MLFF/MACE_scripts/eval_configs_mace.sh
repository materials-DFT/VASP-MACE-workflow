#!/bin/bash
#SBATCH --job-name=Mace_Eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=eval_output.log
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1

set -euo pipefail

# ===== Initialize Conda =====
export PATH="$HOME/miniconda3/bin:$PATH"
# Disable strict error checking for conda activation
set +u
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate mace
set -u

# ===== Move to submit directory =====
cd "$SLURM_SUBMIT_DIR"

usage() {
  cat <<'USAGE'
Usage:
  sbatch eval_configs.sh [--model MODEL_FILE] [--configs CONFIGS_FILE]
  sbatch eval_configs.sh
  sbatch eval_configs.sh final_frames.xyz
  sbatch eval_configs.sh Test_01.model final_frames.xyz
Notes:
  - CONFIGS_FILE is your structures (e.g., .xyz)
  - MODEL_FILE is your trained MACE model (.model)
  - If --configs is omitted, the script auto-detects a single *.xyz in the current dir
  - If --model is omitted, the script auto-detects a *.model in the current dir
USAGE
}

MODEL_FILE=""
CONFIGS_FILE=""

# ===== Parse args (flags or positional, any order) =====
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      [[ $# -ge 2 ]] || { echo "Error: --model requires a value"; exit 2; }
      MODEL_FILE="$2"; shift 2;;
    -c|--configs|--config)
      [[ $# -ge 2 ]] || { echo "Error: --configs requires a value"; exit 2; }
      CONFIGS_FILE="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      # Heuristics for positional args
      if [[ "$1" == *.model ]]; then
        MODEL_FILE="$1"
      else
        CONFIGS_FILE="$1"
      fi
      shift;;
  esac
done

# ===== Auto-detect configs if not provided =====
if [[ -z "${CONFIGS_FILE}" ]]; then
  shopt -s nullglob
  XYZ_FILES=( ./*.xyz )
  shopt -u nullglob

  if [[ ${#XYZ_FILES[@]} -eq 0 ]]; then
    echo "❌ Error: No .xyz file provided and none found in current directory."
    exit 1
  elif [[ ${#XYZ_FILES[@]} -gt 1 ]]; then
    echo "❌ Error: Multiple .xyz files found. Please specify one with --configs:"
    printf '  %s\n' "${XYZ_FILES[@]}"
    exit 1
  else
    CONFIGS_FILE="${XYZ_FILES[0]}"
  fi
fi

# ===== Auto-detect model if not provided =====
if [[ -z "${MODEL_FILE}" ]]; then
  # Prefer a single .model in cwd
  mapfile -t MODELS < <(find . -maxdepth 1 -type f -name "*.model" | sort)
  if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "❌ Error: No .model file provided and none found in current directory."
    exit 1
  elif [[ ${#MODELS[@]} -gt 1 ]]; then
    echo "❌ Error: Multiple .model files found. Please specify one with --model:"
    printf '  %s\n' "${MODELS[@]}"
    exit 1
  else
    MODEL_FILE="${MODELS[0]}"
  fi
fi

# ===== Validate files =====
if [[ ! -f "$CONFIGS_FILE" ]]; then
  echo "❌ Error: Configs file not found: $CONFIGS_FILE"
  exit 1
fi
if [[ ! -f "$MODEL_FILE" ]]; then
  echo "❌ Error: Model file not found: $MODEL_FILE"
  exit 1
fi

echo "✅ Using configs file: $CONFIGS_FILE"
echo "✅ Using model file: $MODEL_FILE"

# ===== Debug information =====
echo "🔍 Debug info:"
echo "  - SLURM_JOB_ID: ${SLURM_JOB_ID:-'Not set'}"
echo "  - SLURM_JOB_NODELIST: ${SLURM_JOB_NODELIST:-'Not set'}"
echo "  - CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'Not set'}"
echo "  - Python path: $(which python)"
echo "  - Python version: $(python --version)"

# ===== Check if configs file has stress data (extxyz format) =====
if grep -q 'stress=' "$CONFIGS_FILE"; then
  echo "✅ Detected stress data in $CONFIGS_FILE — enabling --compute_stress"
  COMPUTE_STRESS=(--compute_stress)
else
  echo "ℹ No stress data in $CONFIGS_FILE — skipping stress computation"
  COMPUTE_STRESS=()
fi

# ===== Run evaluation =====
python -m mace.cli.eval_configs \
  --configs="$CONFIGS_FILE" \
  --model="$MODEL_FILE" \
  --output="output.xyz" \
  --default_dtype="float64" \
  --device="cuda" \
  --batch_size=8 \
  "${COMPUTE_STRESS[@]}"

# ===== Add REF_stress to output when configs have stress (ASE moves it to calc; MACE sets calc=None) =====
# Safe to run always: adds REF_stress when present, no-op when absent
python "$HOME/scripts/add_ref_stress_to_output.py" --configs="$CONFIGS_FILE" --output="output.xyz"

echo "✅ MACE evaluation completed"
