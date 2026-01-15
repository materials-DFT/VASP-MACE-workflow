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
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate mace

# ===== Move to submit directory =====
cd "$SLURM_SUBMIT_DIR"

usage() {
  cat <<'USAGE'
Usage:
  sbatch eval_configs.sh [--model MODEL_FILE] [--configs CONFIGS_FILE]
  sbatch eval_configs.sh final_frames.xyz
  sbatch eval_configs.sh Test_01.model final_frames.xyz
Notes:
  - CONFIGS_FILE is your structures (e.g., .xyz)
  - MODEL_FILE is your trained MACE model (.model)
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

# ===== Defaults =====
CONFIGS_FILE="${CONFIGS_FILE:-combined_100.xyz}"

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

# ===== Run evaluation =====
python -m mace.cli.eval_configs \
  --configs="$CONFIGS_FILE" \
  --model="$MODEL_FILE" \
  --output="output.xyz" \
  --default_dtype="float64" \
  --device="cuda" \
  --batch_size=8

echo "✅ MACE evaluation completed"
