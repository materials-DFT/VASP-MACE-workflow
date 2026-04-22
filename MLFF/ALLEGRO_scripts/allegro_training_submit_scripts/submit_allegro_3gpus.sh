#!/bin/bash
#SBATCH --job-name=Allegro_train_3g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
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

# Prefer dataset with stress added for stress-inclusive training
if [ -f "npt+_dataset_with_stress.xyz" ]; then
  TRAIN_FILE=npt+_dataset_with_stress.xyz
else
  TRAIN_FILE=$(find . -maxdepth 1 \( -name "*.xyz" -o -name "*.extxyz" \) | head -n 1)
fi
if [ -z "$TRAIN_FILE" ]; then
    echo "Error: No .xyz or .extxyz training file found in $PWD"
    exit 1
fi
# Remove ./ prefix if present for cleaner path
TRAIN_FILE="${TRAIN_FILE#./}"
export TRAIN_FILE
echo "Found training file: $TRAIN_FILE"

# Trainer args: use GPU if available, else CPU (e.g. when running locally without sbatch)
NGPU=$($CONDA_PREFIX/bin/python -c "import torch; print(torch.cuda.device_count())")
if [ "$NGPU" -gt 0 ]; then
  TRAINER_ARGS="trainer.accelerator=gpu trainer.devices=$NGPU +trainer.strategy=ddp"
  echo "Using $NGPU GPU(s) with DDP"
else
  TRAINER_ARGS="trainer.accelerator=cpu trainer.devices=1"
  echo "No GPU found - using CPU"
fi

python -m nequip.scripts.train --config-path $PWD \
                               --config-name allegro \
                               $TRAINER_ARGS \
                               trainer.log_every_n_steps=1
