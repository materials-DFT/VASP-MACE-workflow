#!/bin/bash

#SBATCH --partition=gpuquick
# 4 MPI ranks x 4 GPUs, processors 2x2x1 (see in.npt_md_allegro).
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH -J lmp-allegro
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH -o job.out
#SBATCH -e job.err

set -euo pipefail

### ---- user settings (edit these) ----
# Automatically detect input files in current directory
export INPUT_FILE=""  # Will be auto-detected
export DATA_FILE=""   # Will be auto-detected  
export MODEL_FILE=""  # Will be auto-detected
# Systemwide LAMMPS+ALLEGRO build (installed by sysadmin)
export LAMMPS_PREFIX="/Apps/chem/lammps_allegro"
export LMP_BINARY="${LAMMPS_PREFIX}/bin/lmp"
export LIBTORCH_DIR="/opt/pytorch/libtorch-gpu"
### -----------------------------------

# Auto-detect input files in current directory
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-notset}"
echo "Auto-detecting input files in current directory..."

# Prefer the standard production input so stray *.in (e.g. tests/smoke_test.in) is not picked first.
if [[ -f "./in.npt_md_allegro" ]]; then
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
# Prioritize files starting with "data" to avoid picking up log.lammps
# Control test: set LAMMPS_DATA_FILE=./data.lammps.from2 and triclinic_fix=1 in input to match ../2.
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
# Prefer TorchScript (.nequip.pth) first because current cluster libtorch AOTI ABI
# is not compatible with some newer .nequip.pt2 exports.
# Fallback order: .nequip.pth -> .nequip.pt2 -> any .pt
if ls -1 *.nequip.pth >/dev/null 2>&1; then
    MODEL_FILE=$(ls -1 *.nequip.pth | head -1)
elif ls -1 *.nequip.pt2 >/dev/null 2>&1; then
    MODEL_FILE=$(ls -1 *.nequip.pt2 | head -1)
elif ls -1 *.pt >/dev/null 2>&1; then
    MODEL_FILE=$(ls -1 *.pt | head -1)
else
    echo "ERROR: No ALLEGRO model file (*.nequip.pt2, *.nequip.pth, or *.pt) found in current directory"
    echo "Please compile your .ckpt model first using convert_allegro_to_lammps.sh"
    exit 1
fi
echo "Found MODEL_FILE: $MODEL_FILE"
if [[ "$MODEL_FILE" == *.nequip.pth ]]; then
    echo "Model selection: preferring TorchScript (.nequip.pth) for runtime compatibility."
elif [[ "$MODEL_FILE" == *.nequip.pt2 ]]; then
    echo "Model selection: using AOTInductor (.nequip.pt2). If model load fails with missing aoti_torch_* symbols, re-export with a runtime-matched stack or use .nequip.pth."
fi

# Modules (per sysadmin instructions for systemwide LAMMPS+ALLEGRO)
module purge
module load gcc-toolset/12
module load mpi/openmpi-x86_64
module load cuda-toolkit/12.6

# Build LD_LIBRARY_PATH from scratch -- avoid conda/miniconda libs leaking in
# and conflicting with system OpenMPI/PMIx
export LD_LIBRARY_PATH="${LIBTORCH_DIR}/lib:${LAMMPS_PREFIX}/lib64"


# Optional: reduce surprises
export MKL_THREADING_LAYER=INTEL
export OMP_NUM_THREADS=1
# Kokkos OpenMP host backend: silence warnings and pin consistently (1 thread/rank)
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export TORCH_NUM_INTRAOP_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE

# Single-node: use ob1 + shared memory only. Avoids UCX shared-memory/CUDA edge cases that
# segfault with Kokkos gpu/aware on this stack, and avoids mlx5 OpenIB UD QP noise.
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,vader

# minimal sanity output
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

# Preflight checks (fail early with clear messages)
# Check for lmp binary
if [[ ! -x "$LMP_BINARY" ]]; then
    echo "ERROR: lmp not found at: $LMP_BINARY"
    echo "Checking ${LAMMPS_PREFIX}/bin:"
    ls -la "${LAMMPS_PREFIX}/bin" 2>/dev/null | head -10 || echo "Directory does not exist"
    echo "Checking ${LAMMPS_PREFIX}:"
    ls -la "${LAMMPS_PREFIX}" 2>/dev/null | head -10 || echo "Directory does not exist"
    exit 1
fi
echo "Using LAMMPS binary: $LMP_BINARY"
ls -lh "$LMP_BINARY"
[[ -f "$INPUT_FILE" ]] || { echo "ERROR: INPUT_FILE not found: $INPUT_FILE"; exit 1; }
[[ -f "$DATA_FILE" ]] || { echo "ERROR: DATA_FILE not found: $DATA_FILE"; exit 1; }
[[ -f "$MODEL_FILE" ]] || { echo "ERROR: MODEL_FILE not found: $MODEL_FILE"; exit 1; }

# cd to the input file's directory so relative paths (e.g., read_data data.carbon) work
WORKDIR="$(dirname "$INPUT_FILE")"
BASENAME="$(basename "$INPUT_FILE")"
cd "$WORKDIR"

# If present, delete restart checkpoints then remove (use after changing data.lammps; Slurm --export is unreliable).
if [[ -f ".force_fresh_run" ]]; then
    rm -f restart.lmp.last restart.lmp.a restart.lmp.b .force_fresh_run
    echo "Cleared restart checkpoints (.force_fresh_run was present)."
fi

# Detect latest restart (prefer explicit final restart from write_restart)
# Optional debug knob: `FORCE_FRESH_RUN=1` disables restart usage so we start from `read_data`.
RESTART_FILE="none"
RESTART_FLAG=0
if [[ "${FORCE_FRESH_RUN:-0}" == "1" ]]; then
    echo "FORCE_FRESH_RUN=1 => running without restart (read_data path)."
else
    if [[ -f "restart.lmp.last" ]]; then
        RESTART_FILE="restart.lmp.last"
        RESTART_FLAG=1
    else
        # Periodic restarts toggle between restart.lmp.a and restart.lmp.b; pick the newer one
        if [[ -f "restart.lmp.a" ]] || [[ -f "restart.lmp.b" ]]; then
            # Be robust when only one of the two files exists.
            # With `set -euo pipefail`, `ls` returning non-zero (e.g. missing restart.lmp.b)
            # would otherwise terminate the job before LAMMPS starts.
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

# 0 = orthogonal supercell (default); 1 = triclinic + change_box path (see ../2).
TRICLINIC_FIX="${TRICLINIC_FIX:-0}"
echo "TRICLINIC_FIX=${TRICLINIC_FIX}"

# -log none: LAMMPS otherwise truncates log.lammps at startup before the input script runs;
# the input file opens log.lammps with "append" so restarts continue one continuous log.
mpirun -np 4 \
    "$LMP_BINARY" -log none -k on g 4 -sf kk -pk kokkos newton on neigh half \
    -var model "$MODEL_FILE" -var datafile "$DATA_FILE" -var restartfile "$RESTART_FILE" -var restartflag "$RESTART_FLAG" -var triclinic_fix "$TRICLINIC_FIX" -var use_extra_dump 1 -in "$BASENAME"

# Decide whether to requeue based on last completed timestep (must match
# "variable target equal ..." in the LAMMPS input used for "run ${target} upto")
TARGET_STEP=8000000
LAST_STEP=0
if [[ -f "log.lammps" ]]; then
    # Grab last thermo line where first column is an integer step
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
