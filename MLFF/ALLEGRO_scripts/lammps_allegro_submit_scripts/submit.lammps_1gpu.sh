#!/bin/bash

# 1 MPI rank x 1 GPU
# Uses in.npt_md_allegro.1gpu (processors 1 1 1) so `mpirun -np 1` matches LAMMPS decomposition.
# Do not override CUDA_VISIBLE_DEVICES; Slurm sets it for the allocated GPUs.
#SBATCH --partition=gpuquick
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -J lmp-allegro-1g
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH -o job.out
#SBATCH -e job.err

set -euo pipefail

### ---- user settings (edit these) ----
export INPUT_FILE=""  # Will be auto-detected
export DATA_FILE=""   # Will be auto-detected
export MODEL_FILE=""  # Will be auto-detected

export LAMMPS_PREFIX="/Apps/chem/lammps_allegro"
export LMP_BINARY="${LAMMPS_PREFIX}/bin/lmp"
export LIBTORCH_DIR="/opt/pytorch/libtorch-gpu"
### -----------------------------------

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-notset}"
echo "Auto-detecting input files in current directory..."

# Prefer 1-GPU input variant when present.
if [[ -f "./in.npt_md_allegro.1gpu" ]]; then
    INPUT_FILE="./in.npt_md_allegro.1gpu"
elif [[ -f "./in.npt_md_allegro" ]]; then
    INPUT_FILE="./in.npt_md_allegro"
else
    INPUT_FILE=$(find . -maxdepth 1 \( -name "*.in" -o -name "*.txt" -o -name "in.*" \) | head -1)
fi
if [[ -z "$INPUT_FILE" ]]; then
    echo "ERROR: No LAMMPS input file (.in, .txt, or in.*) found in current directory"
    exit 1
fi
echo "Found INPUT_FILE: $INPUT_FILE"

# Find data file (.lammps, .data, or data.*)
if [[ -n "${LAMMPS_DATA_FILE:-}" ]]; then
    DATA_FILE="${LAMMPS_DATA_FILE}"
elif [[ -f "data.lammps" ]]; then
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

# Find model file.
if ls -1 *.nequip.pth >/dev/null 2>&1; then
    MODEL_FILE=$(ls -1 *.nequip.pth | head -1)
elif ls -1 *.nequip.pt2 >/dev/null 2>&1; then
    MODEL_FILE=$(ls -1 *.nequip.pt2 | head -1)
elif ls -1 *.pt >/dev/null 2>&1; then
    MODEL_FILE=$(ls -1 *.pt | head -1)
else
    echo "ERROR: No ALLEGRO model file (*.nequip.pt2, *.nequip.pth, or *.pt) found in current directory"
    exit 1
fi
echo "Found MODEL_FILE: $MODEL_FILE"

module purge
module load gcc-toolset/12
module load mpi/openmpi-x86_64
module load cuda-toolkit/12.6

export LD_LIBRARY_PATH="${LIBTORCH_DIR}/lib:${LAMMPS_PREFIX}/lib64"

export MKL_THREADING_LAYER=INTEL
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export TORCH_NUM_INTRAOP_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

# Single-node: use ob1 + shared memory only.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader

echo "Node: $(hostname)"
echo "CUDA: $(nvcc --version 2>/dev/null | grep release || true)"
echo "GPU:  $(nvidia-smi -L 2>/dev/null | tr -d '\n' || true)"
echo "Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "OMP_PROC_BIND=${OMP_PROC_BIND} OMP_PLACES=${OMP_PLACES} OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "TORCH_NUM_INTRAOP_THREADS=${TORCH_NUM_INTRAOP_THREADS} TORCH_NUM_INTEROP_THREADS=${TORCH_NUM_INTEROP_THREADS}"
echo "OMPI_MCA_pml=${OMPI_MCA_pml} OMPI_MCA_btl=${OMPI_MCA_btl}"
echo "Using INPUT: $INPUT_FILE"
echo "Using DATA: $DATA_FILE"
echo "Using MODEL: $MODEL_FILE"

if [[ ! -x "$LMP_BINARY" ]]; then
    echo "ERROR: lmp not found at: $LMP_BINARY"
    exit 1
fi
[[ -f "$INPUT_FILE" ]] || { echo "ERROR: INPUT_FILE not found: $INPUT_FILE"; exit 1; }
[[ -f "$DATA_FILE" ]] || { echo "ERROR: DATA_FILE not found: $DATA_FILE"; exit 1; }
[[ -f "$MODEL_FILE" ]] || { echo "ERROR: MODEL_FILE not found: $MODEL_FILE"; exit 1; }

WORKDIR="$(dirname "$INPUT_FILE")"
BASENAME="$(basename "$INPUT_FILE")"
cd "$WORKDIR"

if [[ -f ".force_fresh_run" ]]; then
    rm -f restart.lmp.last restart.lmp.a restart.lmp.b .force_fresh_run
    echo "Cleared restart checkpoints (.force_fresh_run was present)."
fi

RESTART_FILE="none"
RESTART_FLAG=0
if [[ "${FORCE_FRESH_RUN:-0}" == "1" ]]; then
    echo "FORCE_FRESH_RUN=1 => running without restart (read_data path)."
else
    if [[ -f "restart.lmp.last" ]]; then
        RESTART_FILE="restart.lmp.last"
        RESTART_FLAG=1
    else
        if [[ -f "restart.lmp.a" ]] || [[ -f "restart.lmp.b" ]]; then
            RESTART_CANDIDATES=()
            [[ -f "restart.lmp.a" ]] && RESTART_CANDIDATES+=("restart.lmp.a")
            [[ -f "restart.lmp.b" ]] && RESTART_CANDIDATES+=("restart.lmp.b")
            if [[ ${#RESTART_CANDIDATES[@]} -gt 0 ]]; then
                RESTART_FILE="$(ls -t "${RESTART_CANDIDATES[@]}" | head -1)"
                RESTART_FLAG=1
            fi
        fi
    fi
fi
echo "Using RESTART_FILE: ${RESTART_FILE} (restartflag=${RESTART_FLAG})"

TRICLINIC_FIX="${TRICLINIC_FIX:-0}"
echo "TRICLINIC_FIX=${TRICLINIC_FIX}"

mpirun -np 1 \
    "$LMP_BINARY" -log none -k on g 1 -sf kk -pk kokkos newton on neigh half \
    -var model "$MODEL_FILE" -var datafile "$DATA_FILE" -var restartfile "$RESTART_FILE" -var restartflag "$RESTART_FLAG" -var triclinic_fix "$TRICLINIC_FIX" -var use_extra_dump 1 -in "$BASENAME"

TARGET_STEP=8000000
LAST_STEP=0
if [[ -f "log.lammps" ]]; then
    LAST_STEP="$(awk '($1 ~ /^[0-9]+$/){s=$1} END{print s+0}' log.lammps 2>/dev/null || echo 0)"
fi
echo "LAST_STEP=${LAST_STEP} TARGET_STEP=${TARGET_STEP}"

if [[ "${LAST_STEP}" -ge "${TARGET_STEP}" ]]; then
    echo "Simulation complete. Not requeueing."
    exit 0
fi

echo "Simulation not complete. Requeueing job ${SLURM_JOB_ID}..."
scontrol requeue "${SLURM_JOB_ID}" || {
    echo "WARNING: requeue failed; you may need to resubmit manually."
    exit 1
}
exit 0

