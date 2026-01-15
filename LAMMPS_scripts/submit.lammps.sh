#!/bin/bash

#SBATCH --partition=gpucluster
#SBATCH --ntasks=1

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -J lmp-mace
#SBATCH --time=08:00:00
#SBATCH -o job.out
#SBATCH -e job.err

set -euo pipefail

### ---- user settings (edit these) ----
# Automatically detect input files in current directory
export INPUT_FILE=""  # Will be auto-detected
export DATA_FILE=""   # Will be auto-detected  
export MODEL_FILE=""  # Will be auto-detected
# Force use of custom LAMMPS build with MACE support
export LAMMPS_PREFIX="$HOME/src/lammps/build-gpu/install"
echo "Using custom LAMMPS with MACE support: $LAMMPS_PREFIX"
export LIBTORCH_DIR="$HOME/src/libtorch-gpu"                  # libtorch unpacked here
### -----------------------------------

# Auto-detect input files in current directory
echo "Auto-detecting input files in current directory..."

# Find LAMMPS input file (.in, .txt, or files starting with "in.")
INPUT_FILE=$(find . -maxdepth 1 \( -name "*.in" -o -name "*.txt" -o -name "in.*" \) | head -1)
if [[ -z "$INPUT_FILE" ]]; then
    echo "ERROR: No LAMMPS input file (.in, .txt, or in.*) found in current directory"
    exit 1
fi
echo "Found INPUT_FILE: $INPUT_FILE"

# Find data file (.lammps, .data, or data.*)
# Prioritize files starting with "data" to avoid picking up log.lammps
if [[ -f "data.lammps" ]]; then
    DATA_FILE="./data.lammps"
elif [[ -f "data.data" ]]; then
    DATA_FILE="./data.data"
else
    DATA_FILE=$(find . -maxdepth 1 \( -name "data.*" -o -name "*.data" -o -name "*.lammps" \) | head -1)
fi
if [[ -z "$DATA_FILE" ]]; then
    echo "ERROR: No data file (.lammps, .data, or data.*) found in current directory"
    exit 1
fi
echo "Found DATA_FILE: $DATA_FILE"

# Find model file (prefer *-lammps.pt, then any .pt, then .model)
if [[ -f *-lammps.pt ]]; then
    MODEL_FILE=$(ls -1 *-lammps.pt | head -1)
elif ls -1 *.pt >/dev/null 2>&1; then
    MODEL_FILE=$(ls -1 *.pt | head -1)
elif ls -1 *.model >/dev/null 2>&1; then
    MODEL_FILE=$(ls -1 *.model | head -1)
else
    echo "ERROR: No model file (*-lammps.pt, .pt, or .model) found in current directory"
    exit 1
fi
echo "Found MODEL_FILE: $MODEL_FILE"

# toolchain/modules - DON'T load MPI module to avoid PMIx errors
module purge
module load gcc-toolset/12

# CUDA 12.2 (headers present, matches our build)
export CUDA_HOME=/usr/local/cuda-12.2.old
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# conda env (provides PyTorch/MACE + MKL headers at build time; ok if not used at runtime)
source "$HOME/miniconda3/etc/profile.d/conda.sh"
set +u
conda activate mace
set -u

# runtime libs for mkl, lammps, libtorch (keep conda first so MKL is found)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LAMMPS_PREFIX/lib64:$LIBTORCH_DIR/lib:$LD_LIBRARY_PATH"

# Optional: reduce surprises
export MKL_THREADING_LAYER=INTEL
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

# minimal sanity output
echo "Node: $(hostname)"
echo "CUDA: $(nvcc --version | head -n1 || true)"
echo "GPU:  $(nvidia-smi -L | tr -d '\n' || true)"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "Using INPUT: $INPUT_FILE"
echo "Using DATA: $DATA_FILE"
echo "Using MODEL: $MODEL_FILE"

# Preflight checks (fail early with clear messages)
[[ -x "$LAMMPS_PREFIX/bin/lmp" ]] || { echo "ERROR: lmp not found at $LAMMPS_PREFIX/bin/lmp"; exit 1; }
[[ -f "$INPUT_FILE" ]] || { echo "ERROR: INPUT_FILE not found: $INPUT_FILE"; exit 1; }
[[ -f "$DATA_FILE" ]] || { echo "ERROR: DATA_FILE not found: $DATA_FILE"; exit 1; }
[[ -f "$MODEL_FILE" ]] || { echo "ERROR: MODEL_FILE not found: $MODEL_FILE"; exit 1; }

# cd to the input file's directory so relative paths (e.g., read_data data.carbon) work
WORKDIR="$(dirname "$INPUT_FILE")"
BASENAME="$(basename "$INPUT_FILE")"
cd "$WORKDIR"

# Run LAMMPS directly (no srun needed for single-node jobs)
"$LAMMPS_PREFIX/bin/lmp" -k on g 1 -sf kk -pk kokkos neigh half -var model "$MODEL_FILE" -var datafile "$DATA_FILE" -in "$BASENAME"



