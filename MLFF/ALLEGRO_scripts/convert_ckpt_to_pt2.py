#!/usr/bin/env python3
import argparse
import os
import stat
import subprocess
import sys
from pathlib import Path


def resolve_checkpoint_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    if p.is_file():
        return p
    if p.is_dir():
        preferred = p / "best.ckpt"
        if preferred.is_file():
            print(f"Using {preferred}")
            return preferred
        ckpts = sorted(x for x in p.glob("*.ckpt") if x.is_file())
        if len(ckpts) == 1:
            print(f"Using sole .ckpt in directory: {ckpts[0]}")
            return ckpts[0]
        if len(ckpts) == 0:
            raise FileNotFoundError(f"No .ckpt found in directory: {p}")
        names = ", ".join(x.name for x in ckpts)
        raise RuntimeError(f"Multiple .ckpt files found: {names}")
    raise FileNotFoundError(str(p))


def ensure_pt2_output(ckpt: Path, output: str | None) -> Path:
    if output:
        out = Path(output).expanduser().resolve()
    else:
        out = ckpt.with_suffix(".nequip.pt2")
    if not str(out).endswith(".nequip.pt2"):
        raise ValueError(f"Output must end with .nequip.pt2, got: {out}")
    return out


def submit_safe_export(
    ckpt_path: Path,
    out_path: Path,
    libtorch_dir: str,
    torch_ver: str,
    torch_channel: str,
    nequip_ver: str,
    allegro_ver: str,
    venv_dir: str | None,
) -> int:
    workdir = ckpt_path.parent
    job_script = workdir / f".tmp_export_{ckpt_path.name}.{os.getpid()}.sh"

    script = """#!/usr/bin/env bash
set -euo pipefail
module purge
module load gcc-toolset/12 || true
module load mpi/openmpi-x86_64 || true
module load cuda-toolkit/12.6 || module load cuda-toolkit/12.4 || true

export CC="${CC:-/opt/rh/gcc-toolset-12/root/usr/bin/gcc}"
export CXX="${CXX:-/opt/rh/gcc-toolset-12/root/usr/bin/g++}"
if [[ ! -x "$CC" ]]; then
  export CC="$(command -v gcc)"
  export CXX="$(command -v g++)"
fi

GCC_LIBSTDCPP_DIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LD_LIBRARY_PATH="${LIBTORCH_DIR}/lib:${GCC_LIBSTDCPP_DIR}:${LD_LIBRARY_PATH:-}"
if [[ -d /usr/local/cuda/include ]] && [[ -f /usr/local/cuda/include/cuda.h ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
fi
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-0}"

if echo ":${PATH}:" | grep -q nvidia_hpc_sdk; then
  export PATH="$(printf '%s' "$PATH" | tr ':' '\\n' | grep -v nvidia_hpc_sdk | paste -sd: -)"
fi

TMPROOT="${SLURM_TMPDIR:-/tmp}"
VENV_DIR="${VENV_DIR:-${TMPROOT}/nequip_pt2_libtorch_${SLURM_JOB_ID:-local}}"
python3.11 -m venv "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install -q -U pip setuptools wheel
python -m pip install --no-cache-dir "torch==${TORCH_VER}+${TORCH_CHANNEL}" torchvision torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_CHANNEL}"
python -m pip install --no-cache-dir "nequip==${NQ_VER}" "nequip-allegro==${ALLEGRO_VER}"

cd "$WORKDIR"
set +e
python -m nequip.scripts.compile \
  "$CKPT_PATH" \
  "$OUT_PATH" \
  --device cuda \
  --mode aotinductor \
  --target pair_allegro
RC=$?
set -e
if [[ "$RC" -ne 0 ]] && [[ -f "$OUT_PATH" ]]; then
  python - <<PY
import torch
m = torch._inductor.aoti_load_package(r"$OUT_PATH")
print("AOTI load smoke test passed:", type(m).__name__)
PY
  exit 0
fi
exit "$RC"
"""

    job_script.write_text(script, encoding="utf-8")
    job_script.chmod(job_script.stat().st_mode | stat.S_IXUSR)

    export_items = {
        "ALL": None,
        "LIBTORCH_DIR": libtorch_dir,
        "TORCH_VER": torch_ver,
        "TORCH_CHANNEL": torch_channel,
        "NQ_VER": nequip_ver,
        "ALLEGRO_VER": allegro_ver,
        "CKPT_PATH": str(ckpt_path),
        "OUT_PATH": str(out_path),
        "WORKDIR": str(workdir),
    }
    if venv_dir:
        export_items["VENV_DIR"] = venv_dir
    export_arg = ",".join(
        [k if v is None else f"{k}={v}" for k, v in export_items.items()]
    )

    cmd = [
        "sbatch",
        "--partition",
        "gpuquick",
        "--gres",
        "gpu:1",
        "--nodes",
        "1",
        "--time",
        "02:00:00",
        "--output",
        "/dev/null",
        "--error",
        "/dev/null",
        "--export",
        export_arg,
        "--chdir",
        str(workdir),
        str(job_script),
    ]

    print("Submitting safe cluster-libtorch export")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output:     {out_path}")
    print(f"Torch:      {torch_ver}+{torch_channel}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        job_script.unlink()
    except OSError:
        pass

    if proc.returncode != 0:
        print(proc.stderr.strip())
        return proc.returncode
    print(proc.stdout.strip())
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safe .ckpt -> .nequip.pt2 conversion for LAMMPS runtime."
    )
    parser.add_argument("model_path", help=".ckpt file or directory containing best.ckpt")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path ending in .nequip.pt2 (default: <ckpt_basename>.nequip.pt2)",
    )
    parser.add_argument(
        "--libtorch-dir",
        default=os.environ.get("LIBTORCH_DIR", "/opt/pytorch/libtorch-gpu"),
    )
    parser.add_argument("--torch-ver", default=os.environ.get("TORCH_VER", "2.6.0"))
    parser.add_argument(
        "--torch-channel", default=os.environ.get("TORCH_CHANNEL", "cu118")
    )
    parser.add_argument("--nequip-ver", default=os.environ.get("NQ_VER", "0.16.2"))
    parser.add_argument("--allegro-ver", default=os.environ.get("ALLEGRO_VER", "0.8.1"))
    parser.add_argument("--venv-dir", default=os.environ.get("VENV_DIR"))
    args = parser.parse_args()

    try:
        ckpt = resolve_checkpoint_path(args.model_path)
        out = ensure_pt2_output(ckpt, args.output)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    # Login nodes may not expose the same /opt mount points as compute nodes.
    # Warn locally, but let the batch job validate on the GPU node.
    if not Path(args.libtorch_dir, "lib").is_dir():
        print(
            f"Warning: {args.libtorch_dir}/lib not visible on this node; "
            "submission will continue and the compute node will validate it."
        )

    rc = submit_safe_export(
        ckpt_path=ckpt,
        out_path=out,
        libtorch_dir=args.libtorch_dir,
        torch_ver=args.torch_ver,
        torch_channel=args.torch_channel,
        nequip_ver=args.nequip_ver,
        allegro_ver=args.allegro_ver,
        venv_dir=args.venv_dir,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
