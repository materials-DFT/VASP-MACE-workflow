#!/usr/bin/env python3
# The following SBATCH directives are optional. They were originally used
# to set job parameters when submitting this script directly with `sbatch`.
# The `--output` and `--error` directives in particular cause Slurm to write
# files like `extract_mlff_<jobid>.out` and `extract_mlff_<jobid>.err`.
# They are commented out here so that only the internal log
# (`mlff_training_data.log`) and xyz file are written.
#
#SBATCH --job-name=extract_mlff
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
##SBATCH --output=extract_mlff_%j.out
##SBATCH --error=extract_mlff_%j.err
"""
Extract frames from VASP MD trajectories for ALLEGRO/MACE MLFF training.

Run interactively:  python extract_frames_for_mlff.py .
Run on cluster:    sbatch extract_frames_for_mlff.py .
(For sbatch, make script executable: chmod +x extract_frames_for_mlff.py)

Combines:
1. Base extraction rule: stride=10, skip first 10%, base cap=400
2. Temperature weighting: prioritize 700K and 900K for phase transformations
3. Supercell bonus: extra frames for large systems
4. Handles incomplete runs gracefully

Outputs extended xyz format compatible with ALLEGRO/MACE.
Uses ASE to read OUTCAR (trajectory, energy, forces, stress, lattice).
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    from ase import io
    from ase import Atoms
except ImportError:
    print("ERROR: ASE (Atomic Simulation Environment) is required.")
    print("Install with: pip install ase")
    sys.exit(1)


# ============================================================================
# Configuration
# ============================================================================

# Base extraction parameters
STRIDE = 10  # Extract every Nth frame
SKIP_FRACTION = 0.1  # Skip first 10% of trajectory (equilibration)
BASE_CAP = 400  # Base maximum frames per trajectory
MAX_CAP = 600  # Maximum frames even with 2x weight

# Temperature weights (for phase transformation emphasis)
TEMP_WEIGHTS = {
    300: 0.5,   # Low T: equilibrium structures, less exploration
    500: 1.5,   # Low-intermediate: early transformation regime
    700: 2.0,   # Intermediate: SWEET SPOT for phase transformations
    900: 2.0,   # Intermediate-high: SWEET SPOT for phase transformations
    1100: 1.0,  # High: broad phase space exploration
    1300: 0.75  # Very high: may be too disordered
}

# Supercell bonus (for large systems closer to 10-100K atom target)
SUPERCELL_BONUS = 100
SUPERCELL_MAX_CAP = 700

# Supercell threshold (atoms per cell)
SUPERCELL_THRESHOLD = 100  # If >100 atoms, consider it a supercell


# ============================================================================
# Utility Functions
# ============================================================================

def extract_temperature_from_path(path: str) -> Optional[int]:
    """Extract temperature from directory path (e.g., '300K', '700K')."""
    path_str = str(path)
    # Look for patterns like "300K", "700K", "1300K"
    import re
    match = re.search(r'(\d+)K', path_str)
    if match:
        return int(match.group(1))
    return None


def is_supercell(poscar_path: str) -> bool:
    """Check if structure is a supercell based on atom count."""
    try:
        atoms = io.read(poscar_path)
        return len(atoms) >= SUPERCELL_THRESHOLD
    except:
        return False


def calculate_frames_to_extract(nsw: int, temperature: int, is_supercell: bool = False) -> int:
    """
    Calculate number of frames to extract based on:
    - Base rule (stride, skip, cap)
    - Temperature weight
    - Supercell bonus
    """
    # Base frames after stride and skip
    available_steps = int(nsw * (1 - SKIP_FRACTION))
    base_frames = min(available_steps // STRIDE, BASE_CAP)
    
    # Apply temperature weight
    weight = TEMP_WEIGHTS.get(temperature, 1.0)
    weighted_frames = int(base_frames * weight)
    
    # Apply cap
    frames = min(weighted_frames, MAX_CAP)
    
    # Supercell bonus
    if is_supercell:
        frames = min(frames + SUPERCELL_BONUS, SUPERCELL_MAX_CAP)
    
    return max(frames, 1)  # At least 1 frame


def find_md_directories(base_path: str) -> List[Tuple[str, str, str]]:
    """
    Find all MD directories with XDATCAR files.
    
    Returns: List of (xdatcar_path, poscar_path, outcar_path) tuples
    """
    md_dirs = []
    
    # Search patterns for the three main directories
    search_patterns = [
        "kmno/md/neutral/monkhorst-pack_calculated/*/*/XDATCAR",
        "kmno2/md/monkhorst-pack_calculated_optimized/neutral/monkhorst-pack_calculated/*/*/XDATCAR",
        "mno2_phases+k/md/neutral/unitcells/monkhorst-pack_calculated/isif3/*/*/XDATCAR",
        "mno2_phases+k/md/neutral/supercells/monkhorst-pack_calculated/*/*/XDATCAR",
    ]
    
    for pattern in search_patterns:
        full_pattern = os.path.join(base_path, pattern)
        xdatcar_files = glob.glob(full_pattern)
        
        for xdatcar_path in xdatcar_files:
            dir_path = os.path.dirname(xdatcar_path)
            poscar_path = os.path.join(dir_path, "POSCAR")
            outcar_path = os.path.join(dir_path, "OUTCAR")
            
            # Only use OUTCAR (not vasprun.xml)
            if os.path.exists(outcar_path):
                md_dirs.append((xdatcar_path, poscar_path, outcar_path))
            else:
                print(f"Warning: No OUTCAR found in {dir_path}")
                # Still add it, but will skip energy/force extraction
    
    return md_dirs


def make_run_id(base_path: str, outcar_path: str) -> str:
    """Full directory path from ~ as run_id."""
    abs_dir = str(Path(outcar_path).resolve().parent)
    home = str(Path.home())
    if abs_dir.startswith(home):
        return "~" + abs_dir[len(home):]
    return abs_dir


def extract_frames_from_outcar(outcar_path: str, frame_indices: List[int], run_id: str) -> List[Atoms]:
    """Read OUTCAR with ASE and return frames at the given indices (ionic steps)."""
    try:
        images = io.read(outcar_path, index=':')
        frames = []
        for idx in frame_indices:
            if 0 <= idx < len(images):
                atoms = images[idx]
                atoms.info["run_id"] = run_id
                frames.append(atoms)
            else:
                print(f"Warning: Frame index {idx} out of range (max: {len(images)-1})")
        return frames
    except Exception as e:
        print(f"Error reading OUTCAR with ASE {outcar_path}: {e}")
        return []


# ============================================================================
# Main Extraction Function
# ============================================================================

def extract_frames_for_mlff(base_path: str, output_path: str, nsw_override: dict = None):
    """
    Main function to extract frames from all MD trajectories.
    Uses ASE to read OUTCAR (trajectory, energy, forces, stress). NSW = len(images).

    Args:
        base_path: Base directory containing kmno, kmno2, mno2_phases+k
        output_path: Output extended xyz file path
        nsw_override: Unused (kept for API compatibility). NSW is taken from OUTCAR.
    """
    print("=" * 80)
    print("MLFF Frame Extraction with Temperature Weighting")
    print("=" * 80)
    print(f"Base path: {base_path}")
    print(f"Output: {output_path}")
    print()
    
    # Find all MD directories
    print("Searching for MD trajectories...")
    md_dirs = find_md_directories(base_path)
    print(f"Found {len(md_dirs)} MD trajectories with XDATCAR files")
    print()
    
    # Statistics
    stats = {
        'total_trajectories': len(md_dirs),
        'processed': 0,
        'skipped': 0,
        'frames_extracted': 0,
        'by_temperature': {}
    }
    
    all_frames = []

    # Process each trajectory (read OUTCAR with ASE; NSW = len(images))
    for idx, (xdatcar_path, poscar_path, outcar_path) in enumerate(md_dirs):
        dir_path = os.path.dirname(xdatcar_path)

        temperature = extract_temperature_from_path(dir_path)
        if temperature is None:
            print(f"[{idx+1}/{len(md_dirs)}] Skipping {dir_path}: Could not determine temperature")
            stats['skipped'] += 1
            continue

        is_super = False
        if os.path.exists(poscar_path):
            is_super = is_supercell(poscar_path)

        # Read OUTCAR with ASE (trajectory = all ionic steps)
        if not os.path.exists(outcar_path) or not outcar_path.endswith('OUTCAR'):
            print(f"[{idx+1}/{len(md_dirs)}] Skipping {dir_path}: No OUTCAR")
            stats['skipped'] += 1
            continue
        try:
            images = io.read(outcar_path, index=':')
        except Exception as e:
            print(f"[{idx+1}/{len(md_dirs)}] Skipping {dir_path}: ASE failed to read OUTCAR ({e})")
            stats['skipped'] += 1
            continue

        nsw = len(images)
        if nsw == 0:
            print(f"[{idx+1}/{len(md_dirs)}] Skipping {dir_path}: OUTCAR has no frames")
            stats['skipped'] += 1
            continue

        n_frames_to_extract = calculate_frames_to_extract(nsw, temperature, is_super)
        skip_steps = int(nsw * SKIP_FRACTION)
        available_steps = nsw - skip_steps
        frame_indices = list(range(skip_steps, skip_steps + available_steps, STRIDE))
        frame_indices = frame_indices[:n_frames_to_extract]

        if len(frame_indices) == 0:
            print(f"[{idx+1}/{len(md_dirs)}] Skipping {dir_path}: No frames to extract")
            stats['skipped'] += 1
            continue

        print(f"[{idx+1}/{len(md_dirs)}] {dir_path}")
        print(f"  Temperature: {temperature}K (weight: {TEMP_WEIGHTS.get(temperature, 1.0):.2f}x)")
        print(f"  NSW: {nsw}, Supercell: {is_super}")
        print(f"  Extracting {len(frame_indices)} frames (indices: {frame_indices[0]}..{frame_indices[-1]})")

        run_id = make_run_id(base_path, outcar_path)
        frames = extract_frames_from_outcar(outcar_path, frame_indices, run_id)

        if len(frames) == 0:
            print(f"  Warning: No frames extracted")
            stats['skipped'] += 1
            continue

        all_frames.extend(frames)
        stats['processed'] += 1
        stats['frames_extracted'] += len(frames)
        if temperature not in stats['by_temperature']:
            stats['by_temperature'][temperature] = {'trajectories': 0, 'frames': 0}
        stats['by_temperature'][temperature]['trajectories'] += 1
        stats['by_temperature'][temperature]['frames'] += len(frames)
        print(f"  ✓ Extracted {len(frames)} frames")
        print()

    # Write output with ASE (extended XYZ: lattice, energy, forces, stress)
    #
    # Name the xyz file with the final frame count, e.g.
    #   mlff_training_data_59413_frames.xyz
    #
    # This matches counting frames via:
    #   grep Lattice mlff_training_data.xyz | wc -l
    # because ASE writes one "Lattice=..." header per extxyz frame.
    out_p = Path(output_path)
    if out_p.suffix.lower() == ".xyz" and "_frames" not in out_p.stem:
        out_p = out_p.with_name(f"{out_p.stem}_{len(all_frames)}_frames{out_p.suffix}")

    print("=" * 80)
    print("Writing output...")
    io.write(str(out_p), all_frames, format="extxyz")
    print(f"✓ Written {len(all_frames)} frames to {out_p}")
    print()
    
    # Print statistics
    print("=" * 80)
    print("Extraction Statistics")
    print("=" * 80)
    print(f"Total trajectories found: {stats['total_trajectories']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Total frames extracted: {stats['frames_extracted']}")
    print()
    print("Frames by temperature:")
    for temp in sorted(stats['by_temperature'].keys()):
        t_stats = stats['by_temperature'][temp]
        weight = TEMP_WEIGHTS.get(temp, 1.0)
        print(f"  {temp}K (weight: {weight:.2f}x): {t_stats['trajectories']} trajectories, {t_stats['frames']} frames")
    print()
    
    # Calculate expected vs actual
    print("Temperature distribution:")
    for temp in sorted(stats['by_temperature'].keys()):
        pct = 100 * stats['by_temperature'][temp]['frames'] / stats['frames_extracted']
        print(f"  {temp}K: {pct:.1f}%")
    print()


# ============================================================================
# Logging (tee to stdout + log file)
# ============================================================================

class Tee:
    """Write to both stdout and a log file (like converged_global_extract_frames_500+.py style)."""
    def __init__(self, log_path, stream=None):
        self._stream = stream if stream is not None else sys.stdout
        self._log_path = log_path
        self._file = open(log_path, 'w', encoding='utf-8')

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def close(self):
        self._file.close()


# ============================================================================
# Command-line Interface
# ============================================================================

def main():
    global STRIDE, SKIP_FRACTION, BASE_CAP
    parser = argparse.ArgumentParser(
        description="Extract frames from VASP MD trajectories for MLFF training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from current directory
  python extract_frames_for_mlff.py .
  
  # Custom output and stride
  python extract_frames_for_mlff.py . --output training_data.xyz --stride 20
  
  # No temperature weighting (equal frames from all T)
  python extract_frames_for_mlff.py . --no-temperature-weighting
        """
    )
    
    parser.add_argument('base_path', nargs='?', default='.',
                       help='Base directory to scan (default: .)')
    parser.add_argument('--output', type=str, default='mlff_training_data.xyz',
                       help='Output xyz file path (default: mlff_training_data.xyz)')
    parser.add_argument('--stride', type=int, default=10,
                       help=f'Extract every Nth frame (default: {STRIDE})')
    parser.add_argument('--skip-fraction', type=float, default=0.1,
                       help=f'Skip first fraction of trajectory (default: {SKIP_FRACTION})')
    parser.add_argument('--base-cap', type=int, default=400,
                       help=f'Base maximum frames per trajectory (default: {BASE_CAP})')
    parser.add_argument('--no-temperature-weighting', action='store_true',
                       help='Disable temperature weighting (equal frames from all temperatures)')
    parser.add_argument('--log', type=str, default=None,
                       help='Log file path (default: <output_stem>.log, e.g. mlff_training_data.log)')
    
    args = parser.parse_args()
    
    # Update global config
    STRIDE = args.stride
    SKIP_FRACTION = args.skip_fraction
    BASE_CAP = args.base_cap
    
    # Log file: default to <output_stem>.log
    if args.log is None:
        args.log = os.path.splitext(args.output)[0] + '.log'
    
    tee = None
    try:
        tee = Tee(args.log)
        sys.stdout = tee
        if args.no_temperature_weighting:
            for temp in TEMP_WEIGHTS:
                TEMP_WEIGHTS[temp] = 1.0
            print("Temperature weighting disabled - equal frames from all temperatures")
        extract_frames_for_mlff(args.base_path, args.output)
    finally:
        if tee is not None:
            sys.stdout = tee._stream
            tee.close()
            print(f"Log written to {args.log}")


if __name__ == '__main__':
    main()
