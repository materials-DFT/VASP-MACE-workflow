#!/bin/bash
#SBATCH --job-name=vasp_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=dft_vasp6
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=user@domain.edu
#SBATCH --partition=gpucluster

# Initialize the module system (uncomment the one that works for your cluster)
source /etc/profile.d/modules.sh

module purge
# module load gcc-toolset/12  # Disabled: conflicts with NVIDIA HPC SDK OpenMP runtime

# Remove conda/miniconda from LD_LIBRARY_PATH to avoid OpenMP conflicts
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -E '(miniconda|anaconda|conda)' | tr '\n' ':' | sed 's/:$//')

# Set up NVIDIA HPC SDK environment for GPU support
export PATH=/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/compilers/bin:$PATH

# Set up HPCX (CUDA-aware MPI) environment - matching VASP compilation version
source /Users/924322630/nvidia_hpc_sdk/nvhpc_2025_259_Linux_x86_64_cuda_multi/install_components/Linux_x86_64/25.9/comm_libs/12.9/hpcx/hpcx-2.24/hpcx-init-ompi.sh
hpcx_load

# Add NVIDIA HPC SDK libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/compilers/lib:/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/compilers/extras/qd/lib:$LD_LIBRARY_PATH

# Let SLURM handle CUDA_VISIBLE_DEVICES - DO NOT override it!
# Enable OpenACC GPU support
export ACC_DEVICE_TYPE=nvidia

# Use NVIDIA HPC SDK CUDA libraries (matching system version)
export CUDA_HOME=/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Also add CUDA libraries from NVIDIA HPC SDK
export LD_LIBRARY_PATH=/Users/924322630/nvidia_hpc_sdk/install/Linux_x86_64/25.9/cuda/12.9/lib64:$LD_LIBRARY_PATH

# Additional OpenACC environment variables
export ACC_NOTIFY=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Match threads to allocated CPUs
# DO NOT set ACC_DEVICE_NUM - let OpenACC auto-detect from SLURM's CUDA_VISIBLE_DEVICES
echo "LD_LIBRARY_PATH is: $LD_LIBRARY_PATH"

# Test GPU visibility
echo "====================================================="
echo " GPU Detection Test:"
echo " SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo " CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo " ACC_DEVICE_TYPE: $ACC_DEVICE_TYPE"
if command -v nvidia-smi &> /dev/null; then
    echo " nvidia-smi output:"
    nvidia-smi
else
    echo " nvidia-smi not found"
fi
echo "====================================================="

# Detect if inside SLURM or interactive
if [ -z "$SLURM_JOB_ID" ]; then
    echo "🔧 Running interactively"
    WORKDIR=$(pwd)
else
    echo "🚀 Running under SLURM: job ID $SLURM_JOB_ID"
    WORKDIR=$SLURM_SUBMIT_DIR
    cd "$WORKDIR"
fi

echo "====================================================="
echo " Working in directory: $WORKDIR"
echo " Listing input files:"
ls -lh INCAR POSCAR POTCAR KPOINTS
echo "====================================================="

# Path to VASP binary
BIN=$HOME/vasp6/vasp.6.4.3/bin/vasp_std

# Sanity check for binary
if [ ! -x "$BIN" ]; then
    echo "❌ ERROR: VASP binary not found or not executable: $BIN"
    exit 1
fi

# Run VASP — use srun only if under SLURM
if [ -z "$SLURM_JOB_ID" ]; then
    echo "🔧 Running VASP interactively (no srun)"
    $BIN
else
    echo "🚀 Launching VASP via srun"
    srun --export=ALL,LD_LIBRARY_PATH --chdir="$WORKDIR" "$BIN"
fi

exit 0
