#!/bin/bash
#SBATCH --job-name=Allegro_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --partition=gpucluster
#SBATCH --time=08:00:00
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --chdir=.

set -e

# Create logs directory if it doesn't exist (for nequip training logs)
mkdir -p logs

# Force matplotlib non-interactive backend
export MPLBACKEND=Agg

# Activate conda environment
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate nequip

# Show Python and Torch info
echo "Using Python: $CONDA_PREFIX/bin/python"
$CONDA_PREFIX/bin/python --version
$CONDA_PREFIX/bin/python -c "import torch; print('Torch CUDA available:', torch.cuda.is_available()); print('Number of GPUs:', torch.cuda.device_count())"

# CUDA environment
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc

# Automatically locate xyz file
TRAIN_FILE=$(find . -maxdepth 1 \( -name "*.xyz" -o -name "*.extxyz" \) | head -n 1)
if [ -z "$TRAIN_FILE" ]; then
    echo "Error: No .xyz or .extxyz training file found in $PWD"
    exit 1
fi
# Remove ./ prefix if present for cleaner path
TRAIN_FILE="${TRAIN_FILE#./}"
echo "Found training file: $TRAIN_FILE"

# Run training with 4 GPUs
# Note: PyTorch Lightning automatically uses DDP when devices > 1
# Use + prefix to add new config keys that don't exist in the config file
python -m nequip.scripts.train --config-path $PWD \
                               --config-name allegro \
                               data.split_dataset.dataset.file_path=$TRAIN_FILE \
                               trainer.accelerator=gpu \
                               trainer.devices=4 \
                               +trainer.strategy=ddp \
                               trainer.log_every_n_steps=1
