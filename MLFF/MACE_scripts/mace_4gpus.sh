#!/bin/bash
#SBATCH --job-name=Mace_4GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=Test_01.log
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:4

# Use your user-local Miniconda
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate mace

# Debug: Print environment information
echo "=== Environment Debug Info ==="
echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
python -c "import mace; print(f'MACE version: {mace.__version__}')"
echo "=== End Debug Info ==="

cd "$SLURM_SUBMIT_DIR"

TRAIN_FILE=$(find . -maxdepth 1 -name "*.xyz" | head -n 1)
if [ -z "$TRAIN_FILE" ]; then
    echo "Error: No .xyz training file found in $SLURM_SUBMIT_DIR"
    exit 1
fi

srun python -m mace.cli.run_train \
    --name="Test_01" \
    --train_file="$TRAIN_FILE" \
    --valid_fraction=0.05 \
    --config_type_weights='{"Default": 1.0}' \
    --E0s="average" \
    --model="MACE" \
    --hidden_irreps='64x0e + 64x1o + 64x2e' \
    --r_max=5.0 \
    --batch_size=1 \
    --valid_batch_size=1 \
    --max_num_epochs=500 \
    --default_dtype="float64" \
    --max_L=2 \
    --device=cuda \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --restart_latest \
    --distributed
