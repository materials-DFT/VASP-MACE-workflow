#!/usr/bin/env python3
"""
Extract frames from VASP MD trajectories for ALLEGRO/MACE MLFF training.

Combines:
1. Base extraction rule: stride=10, skip first 10%, base cap=400
2. Temperature weighting: prioritize 700K and 900K for phase transformations
3. Supercell bonus: extra frames for large systems
4. Handles incomplete runs gracefully

Outputs extended xyz format compatible with ALLEGRO/MACE.
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


def count_frames_in_xdatcar(xdatcar_path: str) -> int:
    """Count number of frames in XDATCAR file."""
    try:
        with open(xdatcar_path, 'r') as f:
            lines = f.readlines()
        
        # XDATCAR format: header (7 lines) + frames
        # Each frame: 7 lines (repeat header) + N_atoms lines
        # First frame: 7 header + N_atoms
        # Subsequent frames: 7 + N_atoms each
        
        if len(lines) < 7:
            return 0
        
        # Read atom count from header
        atom_line = lines[6].strip()
        n_atoms = sum(int(x) for x in atom_line.split())
        
        if n_atoms == 0:
            return 0
        
        # Calculate frames: (total_lines - 7) / (7 + n_atoms) + 1
        # But VASP may not repeat full header, so count "Direct configuration="
        frame_count = 0
        for line in lines:
            if "Direct configuration=" in line or "configuration=" in line:
                frame_count += 1
        
        return max(frame_count, 0)
    except Exception as e:
        print(f"Warning: Could not count frames in {xdatcar_path}: {e}")
        return 0


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


def extract_frames_from_xdatcar(xdatcar_path: str, indices: List[int]) -> List[Atoms]:
    """Extract specific frame indices from XDATCAR."""
    try:
        # Read all frames from XDATCAR
        all_frames = io.read(xdatcar_path, index=':')
        
        # Extract requested indices
        frames = []
        for idx in indices:
            if 0 <= idx < len(all_frames):
                frames.append(all_frames[idx])
            else:
                print(f"Warning: Frame index {idx} out of range (max: {len(all_frames)-1})")
        
        return frames
    except Exception as e:
        print(f"Error reading XDATCAR {xdatcar_path}: {e}")
        return []


def extract_energies_forces_from_outcar(outcar_path: str, frame_indices: List[int], n_atoms: int) -> Tuple[List[float], List[np.ndarray]]:
    """
    Extract energies and forces for specific frame indices from OUTCAR.
    
    In MD runs, each frame in XDATCAR corresponds to an ionic step.
    OUTCAR contains energy/force data for each ionic step.
    
    Args:
        outcar_path: Path to OUTCAR file
        frame_indices: List of frame indices (0-based, matching XDATCAR frames = ionic steps)
        n_atoms: Number of atoms in the system
    
    Returns:
        Tuple of (energies, forces_list) where energies is list of floats and
        forces_list is list of numpy arrays of shape (n_atoms, 3)
    """
    try:
        energies_list = []  # List of energies for each ionic step (in order)
        forces_list_all = []  # List of forces arrays for each ionic step (in order)
        
        with open(outcar_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        ionic_step = -1  # Track current ionic step
        
        while i < len(lines):
            line = lines[i]
            
            # Look for "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" - this marks end of ionic step
            if "FREE ENERGIE OF THE ION-ELECTRON SYSTEM" in line:
                # Find energy value in next few lines
                energy = None
                for j in range(i, min(i+10, len(lines))):
                    if "free  energy   TOTEN" in lines[j]:
                        # Extract: "free  energy   TOTEN  =      -669.96698010 eV"
                        parts = lines[j].split()
                        for k, part in enumerate(parts):
                            if part == "TOTEN":
                                if k + 2 < len(parts) and parts[k+1] == "=":
                                    try:
                                        energy = float(parts[k+2])
                                        break
                                    except (ValueError, IndexError):
                                        pass
                                elif k + 1 < len(parts):
                                    try:
                                        energy = float(parts[k+1])
                                        break
                                    except ValueError:
                                        pass
                        if energy is not None:
                            break
                
                if energy is not None:
                    ionic_step += 1
                    energies_list.append(energy)
                    # Initialize forces as zeros (will be filled if found)
                    if len(forces_list_all) <= ionic_step:
                        forces_list_all.append(None)
            
            # Look for forces section: "POSITION                                       TOTAL-FORCE (eV/Angst)"
            if "POSITION" in line and "TOTAL-FORCE" in line:
                # Next line is separator "---", then forces
                forces = []
                j = i + 2  # Skip header and separator line
                
                # Read forces until separator or empty line
                while j < len(lines) and len(forces) < n_atoms + 2:
                    force_line = lines[j].strip()
                    if not force_line:
                        break
                    if "---" in force_line and len(forces) > 0:
                        break
                    
                    # Parse force line: position (3 floats) + force (3 floats)
                    # Format: "     5.20770      2.06221      3.02547        -0.200583     -0.167464     -0.097446"
                    parts = force_line.split()
                    if len(parts) >= 6:
                        try:
                            fx = float(parts[3])
                            fy = float(parts[4])
                            fz = float(parts[5])
                            forces.append([fx, fy, fz])
                        except (ValueError, IndexError):
                            # Not a force line, might be separator
                            if len(forces) > 0:
                                break
                    
                    j += 1
                
                if len(forces) == n_atoms:
                    # This forces section belongs to the most recent ionic step
                    if ionic_step >= 0:
                        # Make sure we have enough entries
                        while len(forces_list_all) <= ionic_step:
                            forces_list_all.append(None)
                        forces_list_all[ionic_step] = np.array(forces)
            
            i += 1
        
        # Extract energies and forces for requested frame indices
        # In MD: frame_idx = ionic_step (both 0-based)
        energies = []
        forces_list = []
        
        for frame_idx in frame_indices:
            if frame_idx < len(energies_list):
                energy = energies_list[frame_idx]
            else:
                energy = 0.0
            
            if frame_idx < len(forces_list_all) and forces_list_all[frame_idx] is not None:
                forces = forces_list_all[frame_idx]
            else:
                forces = np.zeros((n_atoms, 3))
            
            energies.append(energy)
            forces_list.append(forces)
        
        if len(energies) == 0 or all(e == 0.0 for e in energies):
            print(f"  Warning: Could not extract energies/forces from {outcar_path}")
            print(f"    Found {len(energies_list)} ionic steps with energies")
            print(f"    Found {sum(1 for f in forces_list_all if f is not None)} ionic steps with forces")
            print(f"    Requested {len(frame_indices)} frame indices: {frame_indices[:5]}...")
        
        return energies, forces_list
        
    except Exception as e:
        print(f"Error parsing OUTCAR {outcar_path}: {e}")
        import traceback
        traceback.print_exc()
        return [], []


def write_extended_xyz(output_path: str, frames: List[Atoms], energies: List[float] = None, 
                       forces_list: List[np.ndarray] = None):
    """Write frames to extended xyz format compatible with ALLEGRO/MACE."""
    with open(output_path, 'w') as f:
        for i, atoms in enumerate(frames):
            # Write number of atoms
            f.write(f"{len(atoms)}\n")
            
            # Write properties line with lattice
            lattice = atoms.cell.array.flatten()
            lattice_str = " ".join(f"{x:.10f}" for x in lattice)
            
            props = ["species:S:1:pos:R:3"]
            if energies and i < len(energies):
                props.append("energy:R:1")
            if forces_list and i < len(forces_list):
                props.append("forces:R:3")
            
            props_str = ":".join(props)
            
            # Write properties line
            f.write(f'Lattice="{lattice_str}" Properties={props_str}')
            if energies and i < len(energies):
                f.write(f' energy={energies[i]:.10f}')
            f.write('\n')
            
            # Write atoms
            for j, atom in enumerate(atoms):
                line = f"{atom.symbol} {atom.position[0]:.10f} {atom.position[1]:.10f} {atom.position[2]:.10f}"
                if forces_list and i < len(forces_list) and j < len(forces_list[i]):
                    forces = forces_list[i][j]
                    line += f" {forces[0]:.10f} {forces[1]:.10f} {forces[2]:.10f}"
                f.write(line + "\n")


# ============================================================================
# Main Extraction Function
# ============================================================================

def extract_frames_for_mlff(base_path: str, output_path: str, nsw_override: dict = None):
    """
    Main function to extract frames from all MD trajectories.
    
    Args:
        base_path: Base directory containing kmno, kmno2, mno2_phases+k
        output_path: Output xyz file path
        nsw_override: Dict mapping directory paths to NSW values (for incomplete runs)
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
    all_energies = []
    all_forces = []
    
    # Process each trajectory
    for idx, (xdatcar_path, poscar_path, energy_path) in enumerate(md_dirs):
        dir_path = os.path.dirname(xdatcar_path)
        
        # Extract temperature
        temperature = extract_temperature_from_path(dir_path)
        if temperature is None:
            print(f"[{idx+1}/{len(md_dirs)}] Skipping {dir_path}: Could not determine temperature")
            stats['skipped'] += 1
            continue
        
        # Check if supercell
        is_super = False
        if os.path.exists(poscar_path):
            is_super = is_supercell(poscar_path)
        
        # Get NSW (number of MD steps)
        nsw = None
        if nsw_override and dir_path in nsw_override:
            nsw = nsw_override[dir_path]
        else:
            # Try to read from INCAR
            incar_path = os.path.join(dir_path, "INCAR")
            if os.path.exists(incar_path):
                try:
                    with open(incar_path, 'r') as f:
                        for line in f:
                            if line.strip().startswith('NSW'):
                                nsw = int(line.split('=')[1].strip())
                                break
                except:
                    pass
        
        # If still no NSW, estimate from XDATCAR frame count
        if nsw is None:
            frame_count = count_frames_in_xdatcar(xdatcar_path)
            # Rough estimate: frames ≈ steps (VASP writes every step to XDATCAR)
            nsw = frame_count
        
        if nsw is None or nsw == 0:
            print(f"[{idx+1}/{len(md_dirs)}] Skipping {dir_path}: Could not determine NSW")
            stats['skipped'] += 1
            continue
        
        # Calculate frames to extract
        n_frames_to_extract = calculate_frames_to_extract(nsw, temperature, is_super)
        
        # Determine frame indices (after skip, with stride)
        skip_steps = int(nsw * SKIP_FRACTION)
        available_steps = nsw - skip_steps
        frame_indices = list(range(skip_steps, skip_steps + available_steps, STRIDE))
        frame_indices = frame_indices[:n_frames_to_extract]  # Apply cap
        
        if len(frame_indices) == 0:
            print(f"[{idx+1}/{len(md_dirs)}] Skipping {dir_path}: No frames to extract")
            stats['skipped'] += 1
            continue
        
        # Extract frames
        print(f"[{idx+1}/{len(md_dirs)}] {dir_path}")
        print(f"  Temperature: {temperature}K (weight: {TEMP_WEIGHTS.get(temperature, 1.0):.2f}x)")
        print(f"  NSW: {nsw}, Supercell: {is_super}")
        print(f"  Extracting {len(frame_indices)} frames (indices: {frame_indices[0]}..{frame_indices[-1]})")
        
        frames = extract_frames_from_xdatcar(xdatcar_path, frame_indices)
        
        if len(frames) == 0:
            print(f"  Warning: No frames extracted")
            stats['skipped'] += 1
            continue
        
        # Extract energies and forces from OUTCAR
        energies = []
        forces_list = []
        
        if os.path.exists(energy_path) and energy_path.endswith('OUTCAR'):
            n_atoms = len(frames[0]) if len(frames) > 0 else 0
            if n_atoms > 0:
                energies, forces_list = extract_energies_forces_from_outcar(energy_path, frame_indices, n_atoms)
        
        # If no energies/forces extracted, create placeholders
        if len(energies) == 0:
            print(f"  Warning: No energies/forces extracted, using placeholders")
            energies = [0.0] * len(frames)
            forces_list = [np.zeros((len(frames[0]), 3))] * len(frames) if len(frames) > 0 else []
        
        # Add to collection
        all_frames.extend(frames)
        all_energies.extend(energies)
        all_forces.extend(forces_list)
        
        stats['processed'] += 1
        stats['frames_extracted'] += len(frames)
        
        # Update temperature stats
        if temperature not in stats['by_temperature']:
            stats['by_temperature'][temperature] = {'trajectories': 0, 'frames': 0}
        stats['by_temperature'][temperature]['trajectories'] += 1
        stats['by_temperature'][temperature]['frames'] += len(frames)
        
        print(f"  ✓ Extracted {len(frames)} frames")
        print()
    
    # Write output
    print("=" * 80)
    print("Writing output...")
    write_extended_xyz(output_path, all_frames, all_energies, all_forces)
    print(f"✓ Written {len(all_frames)} frames to {output_path}")
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
# Command-line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from VASP MD trajectories for MLFF training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from default paths
  python extract_frames_for_mlff.py --base-path . --output training_data.xyz
  
  # Extract with custom stride
  python extract_frames_for_mlff.py --base-path . --output training_data.xyz --stride 20
  
  # Extract without temperature weighting (equal frames from all T)
  python extract_frames_for_mlff.py --base-path . --output training_data.xyz --no-temperature-weighting
        """
    )
    
    parser.add_argument('--base-path', type=str, default='.',
                       help='Base directory containing kmno, kmno2, mno2_phases+k (default: current directory)')
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
    
    args = parser.parse_args()
    
    # Update global config if needed
    global STRIDE, SKIP_FRACTION, BASE_CAP
    STRIDE = args.stride
    SKIP_FRACTION = args.skip_fraction
    BASE_CAP = args.base_cap
    
    if args.no_temperature_weighting:
        # Set all weights to 1.0
        for temp in TEMP_WEIGHTS:
            TEMP_WEIGHTS[temp] = 1.0
        print("Temperature weighting disabled - equal frames from all temperatures")
    
    # Run extraction
    extract_frames_for_mlff(args.base_path, args.output)


if __name__ == '__main__':
    main()
