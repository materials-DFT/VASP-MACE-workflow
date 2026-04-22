#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess

# Note: nequip and allegro imports are not needed on the login node
# They are only used in the compute node script that gets generated


def resolve_checkpoint_path(path):
    """
    Return absolute path to a .ckpt file.

    *path* may be a checkpoint file or a directory (uses best.ckpt if present,
    otherwise the only *.ckpt in the directory).
    """
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if os.path.isfile(path):
        return path

    if os.path.isdir(path):
        preferred = os.path.join(path, "best.ckpt")
        if os.path.isfile(preferred):
            print(f"📂 Using {preferred}")
            return preferred

        ckpts = sorted(
            name
            for name in os.listdir(path)
            if name.endswith(".ckpt")
            and os.path.isfile(os.path.join(path, name))
        )
        if len(ckpts) == 1:
            chosen = os.path.join(path, ckpts[0])
            print(f"📂 Using sole .ckpt in directory: {chosen}")
            return chosen
        if len(ckpts) == 0:
            print(f"❌ No .ckpt file found in directory: {path}")
            sys.exit(1)
        print(f"❌ Multiple .ckpt files in {path}:")
        for name in ckpts:
            print(f"   {name}")
        print("   Pass the checkpoint file explicitly, or use best.ckpt for auto-selection.")
        sys.exit(1)

    print(f"❌ Not a file or directory: {path}")
    sys.exit(1)


def run_on_compute_node(model_path, output_name=None, device='cuda', mode='torchscript'):
    """Submit the conversion to run on compute node with GPU"""
    
    workdir = os.getcwd()
    
    # Convert to absolute path to avoid issues with relative paths
    if not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)
    
    model_basename = os.path.basename(model_path)
    model_dir = os.path.dirname(model_path)
    
    out_path = os.path.join(workdir, f"convert_{model_basename}.out")
    err_path = os.path.join(workdir, f"convert_{model_basename}.err")

    # Determine output filename
    if output_name:
        if mode == 'aotinductor':
            output_file = f"{output_name}.nequip.pt2"
        else:
            output_file = f"{output_name}.nequip.pth"
    else:
        base_name = os.path.splitext(model_basename)[0]
        if mode == 'aotinductor':
            output_file = f"{base_name}.nequip.pt2"
        else:
            output_file = f"{base_name}.nequip.pth"
    
    output_path = os.path.join(workdir, output_file)

    # Create a temporary script for the compute node
    script_content = f'''#!/bin/bash
set -euxo pipefail
#SBATCH --partition=gpucluster
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output={out_path}
#SBATCH --error={err_path}

echo "Converting ALLEGRO model on GPU compute node..."

# Use absolute path for model (already converted in Python)
MODEL_ABS="{model_path}"

# Change to the directory containing the model to resolve relative dataset paths
MODEL_DIR="{model_dir}"
if [ -d "$MODEL_DIR" ]; then
    cd "$MODEL_DIR"
    echo "Changed to model directory: $MODEL_DIR"
    echo "Model absolute path: $MODEL_ABS"
else
    echo "Warning: Model directory not found: $MODEL_DIR"
    cd {workdir}
fi

# Use the specific Python from the nequip environment
PYTHON="/Users/924322630/miniconda3/envs/nequip/bin/python"
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
    import nequip
    print('nequip: installed')
except Exception as e:
    print('ERROR nequip:', e)
    sys.exit(1)
try:
    import allegro
    print('allegro: installed')
except Exception as e:
    print('ERROR allegro:', e)
    sys.exit(1)
try:
    import numpy as np
    print('numpy:', np.__version__)
except Exception as e:
    print('ERROR numpy:', e)
    sys.exit(1)
PY

# Always use Python module directly to avoid shebang issues
# The nequip-compile command may have incorrect interpreter paths
NEQUIP_COMPILE="$PYTHON -m nequip.scripts.compile"
echo "Using: $NEQUIP_COMPILE"

# AOTInductor host builds:
# 1) Triton autotune uses $CC; conda's cross-compiler + sysroot breaks cuda.h+Python.h
#    (conflicting timespec_get in time.h).
# 2) If PATH lists nvidia_hpc_sdk/.../compilers before /usr/bin, Inductor C++ can pick
#    NVHPC intrinsics headers (ammintrin.h etc.) that are incompatible with GNU g++.
# 3) CUDA headers for system g++: prefer system toolkit if present.
if [ "{mode}" = "aotinductor" ] && [ -x /usr/bin/gcc ] && [ -x /usr/bin/g++ ]; then
    echo "🔧 AOT: CC/CXX -> /usr/bin/gcc (was CC=${{CC:-unset}})"
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    if echo ":$PATH:" | grep -q nvidia_hpc_sdk; then
        export PATH="$(printf '%s' "$PATH" | tr ':' '\\n' | grep -v nvidia_hpc_sdk | paste -sd: -)"
        echo "🔧 AOT: removed nvidia_hpc_sdk entries from PATH for GNU-compatible includes"
    fi
    if [ -d /usr/local/cuda/include ] && [ -f /usr/local/cuda/include/cuda.h ]; then
        export CUDA_HOME="${{CUDA_HOME:-/usr/local/cuda}}"
        echo "🔧 AOT: CUDA_HOME=$CUDA_HOME"
    fi
fi

echo "🚀 Starting ALLEGRO to LAMMPS conversion..."
echo "📂 Input model: {model_path}"
echo "📤 Output file: {output_path}"
echo "💻 Device: {device}"
echo "🔧 Mode: {mode}"

# Compile the model
if [ "{mode}" = "aotinductor" ]; then
    echo "🔨 Compiling with AOTInductor mode (recommended for LAMMPS)..."
    $NEQUIP_COMPILE \\
        "$MODEL_ABS" \\
        "{output_path}" \\
        --device {device} \\
        --mode aotinductor \\
        --target pair_allegro
else
    echo "🔨 Compiling with TorchScript mode..."
    $NEQUIP_COMPILE \\
        "$MODEL_ABS" \\
        "{output_path}" \\
        --device {device} \\
        --mode torchscript
fi

if [ $? -eq 0 ]; then
    echo "✅ Successfully compiled model: {output_path}"
    echo "🎉 Conversion completed!"
    echo ""
    echo "To use in LAMMPS, add these lines to your input script:"
    echo "  pair_style allegro"
    echo "  pair_coeff * * {output_file} <type1> <type2> ..."
    echo ""
    echo "Note: Replace <type1> <type2> with the actual chemical symbols"
    echo "      matching your model's type_names (in order)."
else
    echo "❌ Compilation failed!"
    exit 1
fi

echo "Conversion job finished!"
'''
    
    # Write the script to a temporary file
    script_name = f"convert_{model_basename}.sh"
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(script_name, 0o755)
    
    # Submit the job with explicit partition/gres and chdir
    # Use --output and --error to prevent default slurm-*.out files
    print(f"🚀 Submitting conversion job to compute node...")
    result = subprocess.run([
        'sbatch',
        '-p', 'gpucluster',
        '--gres', 'gpu:1',
        '--chdir', workdir,
        '--output', out_path,
        '--error', err_path,
        script_name
    ], capture_output=True, text=True)
    
    # Clean up the temporary script file
    try:
        os.remove(script_name)
    except OSError:
        pass  # Ignore if file doesn't exist or can't be removed
    
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
    parser = argparse.ArgumentParser(
        description="Convert ALLEGRO .ckpt files to LAMMPS-compatible .nequip.pt2/.nequip.pth format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (TorchScript .nequip.pth — stable; default)
  python convert_allegro_to_lammps.py best.ckpt

  # AOTInductor .nequip.pt2 (faster in LAMMPS; can segfault on some torch/GPU stacks)
  python convert_allegro_to_lammps.py best.ckpt --mode aotinductor

  # Directory containing best.ckpt (or a single *.ckpt)
  python convert_allegro_to_lammps.py .

  # Specify output name
  python convert_allegro_to_lammps.py best.ckpt --output best_model

  # Use CPU device
  python convert_allegro_to_lammps.py best.ckpt --device cpu
        """
    )
    
    # Check if arguments were provided without flags (simple usage)
    if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
        raw_path = sys.argv[1]
        try:
            model_path = resolve_checkpoint_path(raw_path)
        except FileNotFoundError:
            print(f"❌ Path not found: {raw_path}")
            sys.exit(1)

        if not model_path.endswith('.ckpt'):
            print(f"⚠ Warning: {model_path} doesn't end with .ckpt")

        # Always submit to compute node for simplicity
        print(f"🖥️  Submitting {model_path} to compute node for conversion...")
        job_id = run_on_compute_node(model_path)
        if job_id:
            print(f"✅ Conversion job submitted for {model_path}")
        return
    
    parser.add_argument(
        "model_path",
        nargs="?",
        help="Path to ALLEGRO .ckpt file, or a directory (best.ckpt or sole *.ckpt)",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output_name",
        help="Output filename (without extension). Default: same as input filename",
    )
    parser.add_argument(
        "--device",
        choices=['cpu', 'cuda'],
        default='cuda',
        help="Device to use for compilation (default: cuda)",
    )
    parser.add_argument(
        "--mode",
        choices=['torchscript', 'aotinductor'],
        default='torchscript',
        help="Compilation mode: 'torchscript' (.nequip.pth, default, works with PyTorch 2.4+) or 'aotinductor' (.nequip.pt2, requires PyTorch >= 2.6)",
    )
    
    args = parser.parse_args()

    if not args.model_path:
        parser.print_help()
        print("\n❌ Error: model_path is required")
        sys.exit(1)

    raw_path = args.model_path
    try:
        model_path = resolve_checkpoint_path(raw_path)
    except FileNotFoundError:
        print(f"❌ Path not found: {raw_path}")
        sys.exit(1)

    if not model_path.endswith(".ckpt"):
        print(f"⚠ Warning: {model_path} doesn't end with .ckpt (may not be an ALLEGRO checkpoint)")

    print(f"🖥️  Submitting {model_path} to compute node...")
    print(f"   Device: {args.device}")
    print(f"   Mode: {args.mode}")
    if args.output_name:
        print(f"   Output: {args.output_name}.nequip.{'pt2' if args.mode == 'aotinductor' else 'pth'}")
    
    job_id = run_on_compute_node(
        model_path,
        output_name=args.output_name,
        device=args.device,
        mode=args.mode
    )
    
    if job_id:
        print(f"✅ Conversion job submitted for {model_path}")

if __name__ == "__main__":
    main()

