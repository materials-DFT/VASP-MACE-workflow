#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess

# Note: torch, e3nn, and mace imports are not needed on the login node
# They are only used in the compute node script that gets generated

def run_on_compute_node(model_path):
    """Submit the conversion to run on compute node with GPU"""
    
    workdir = os.getcwd()
    out_path = os.path.join(workdir, f"convert_{os.path.basename(model_path)}.out")
    err_path = os.path.join(workdir, f"convert_{os.path.basename(model_path)}.err")

    # Create a temporary script for the compute node
    script_content = f'''#!/bin/bash
set -euxo pipefail
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output={out_path}
#SBATCH --error={err_path}

echo "Converting MACE model on GPU compute node..."
cd {workdir}

# Use the specific Python from the mace environment
PYTHON="/Users/924322630/miniconda3/envs/mace/bin/python"
which "$PYTHON" || true
"$PYTHON" -V || true

# Verify required packages in the selected interpreter
"$PYTHON" - <<'PY'
import sys
print('python:', sys.version)
try:
    import torch
    print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
except Exception as e:
    print('ERROR torch:', e)
    sys.exit(1)
try:
    import e3nn
    from e3nn.util import jit
    print('e3nn:', e3nn.__version__)
except Exception as e:
    print('ERROR e3nn:', e)
    sys.exit(1)
try:
    import numpy as np
    print('numpy:', np.__version__)
except Exception as e:
    print('ERROR numpy:', e)
    sys.exit(1)
PY

# Write conversion Python script
cat > convert_inline.py << 'PYCODE'
import torch
from e3nn.util import jit
from mace.calculators import LAMMPS_MACE
import os

print('🚀 Starting MACE to LAMMPS conversion...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('💻 Using device:', device)

model_path = r"{model_path}"
one_path = os.path.abspath(model_path)
print('📂 Loading model:', model_path, '->', one_path)
model = torch.load(one_path, map_location=device)
model = model.double().to(device)

print('🔧 Wrapping model for LAMMPS...')
lammps_model = LAMMPS_MACE(model)

print('🧠 Compiling TorchScript model...')
lammps_model_compiled = jit.compile(lammps_model)

out_path = r"{os.path.splitext(model_path)[0]}-lammps.pt"
lammps_model_compiled.save(out_path)
print('✅ Successfully saved:', out_path)
print('🎉 Conversion completed!')
PYCODE

# Run conversion using the specified Python
"$PYTHON" convert_inline.py

# Cleanup
rm -f convert_inline.py

echo "Conversion job finished!"
'''
    
    # Write the script to a temporary file
    script_name = f"convert_{os.path.basename(model_path)}.sh"
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_name, 0o755)
    
    # Submit the job with explicit partition/gres and chdir
    print(f"🚀 Submitting conversion job to compute node...")
    result = subprocess.run(['sbatch', '-p', 'gpucluster', '--gres', 'gpu:1', '--chdir', workdir, script_name], capture_output=True, text=True)
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        print(f"✅ Job submitted successfully! Job ID: {job_id}")
        print(f"📋 Monitor progress: tail -f {out_path}")
        print(f"🔍 Check errors: tail -f {err_path}")
        print(f"⏱️  Job typically takes 1-2 minutes to complete")
        return job_id
    else:
        print(f"❌ Failed to submit job: {result.stderr}")
        return None

def main():
    # Check if arguments were provided without --model_path flag
    if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
        # Simple usage: python convert_mace_to_lammps.py model.model
        model_path = sys.argv[1]
        
        if not os.path.exists(model_path):
            print(f"❌ File not found: {model_path}")
            sys.exit(1)
        
        if not model_path.endswith('.model'):
            print(f"⚠ Warning: {model_path} doesn't end with .model")
        
        # Always submit to compute node for simplicity
        print(f"🖥️  Submitting {model_path} to compute node for conversion...")
        job_id = run_on_compute_node(model_path)
        if job_id:
            print(f"✅ Conversion job submitted for {model_path}")
        return
    
    # Original argument parsing for backward compatibility
    parser = argparse.ArgumentParser(description="Convert MACE .model files to LAMMPS-compatible .pt format")
    parser.add_argument(
        "--model_path",
        nargs="+",
        help="Path(s) to one or more MACE .model files",
    )
    args = parser.parse_args()

    if not args.model_path:
        print("Usage: python convert_mace_to_lammps.py model_file.model")
        print("   or: python convert_mace_to_lammps.py --model_path model_file.model")
        sys.exit(1)

    for model_path in args.model_path:
        if not model_path.endswith(".model"):
            print(f"⚠ Skipping {model_path} (not a .model file)")
            continue

        print(f"🖥️  Submitting {model_path} to compute node...")
        job_id = run_on_compute_node(model_path)
        if job_id:
            print(f"✅ Conversion job submitted for {model_path}")

if __name__ == "__main__":
    main()