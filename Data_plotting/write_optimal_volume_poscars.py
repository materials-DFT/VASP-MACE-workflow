#!/usr/bin/env python3
"""
Write Optimal Volumes to POSCAR Files Script

This script analyzes bulk modulus calculations from VASP OUTCAR files,
predicts optimal cell volumes using Birch-Murnaghan EOS fitting,
and overwrites POSCAR files with the optimal volumes by scaling lattice vectors.

The script:
1. Scans for OUTCAR files and extracts volume/energy data
2. Fits Birch-Murnaghan EOS to predict optimal volume (V0)
3. Finds reference POSCAR files for each compound
4. Scales lattice vectors to match optimal volume
5. Overwrites POSCAR files with optimal volumes
"""

import os
import re
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from collections import defaultdict
import shutil


class OptimalVolumePOSCARWriter:
    """Write POSCAR files with optimal volumes from EOS fitting."""
    
    def __init__(self, base_dir, output_dir=None):
        """
        Initialize the writer.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing bulk modulus calculations
        output_dir : str, optional
            Output directory for POSCAR files. If None, uses base_dir structure.
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir) if output_dir else self.base_dir
        self.data = defaultdict(dict)  # {compound: {volume: {energy, volume}}}
        self.optimal_volumes = {}  # {compound: V0}
        
    def extract_volume_energy(self, outcar_path):
        """Extract volume and energy from OUTCAR file."""
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
                
            # Extract volume
            vol_pattern = r'volume of cell\s*:\s*([\d.]+)'
            vol_match = re.findall(vol_pattern, content)
            if vol_match:
                volume = float(vol_match[-1])
            else:
                return None, None
                
            # Extract energy
            energy_pattern = r'free\s+energy\s+TOTEN\s+=\s+([\d.-]+)\s+eV'
            energy_matches = re.findall(energy_pattern, content)
            if energy_matches:
                energy = float(energy_matches[-1])
            else:
                return None, None
                
            return volume, energy
            
        except Exception as e:
            return None, None
    
    def parse_directory_structure(self):
        """Parse directory structure and extract data from all OUTCAR files."""
        print("Scanning directory structure...")
        
        outcar_files = list(self.base_dir.rglob('OUTCAR'))
        print(f"Found {len(outcar_files)} OUTCAR files")
        
        for outcar_path in outcar_files:
            try:
                rel_path = outcar_path.relative_to(self.base_dir)
                parts = rel_path.parts
            except ValueError:
                parts = outcar_path.parts
            
            # Find volume directory and compound
            vol_dir = None
            compound = None
            
            for i, part in enumerate(parts):
                if part.startswith('V_'):
                    vol_dir = part
                    if i >= 1:
                        compound = parts[i-1]
                    break
            
            if not vol_dir or not compound:
                skip_dirs = {'optimization', 'neutral', 'monkhorst-pack_calculated', 
                            'unitcells', 'isif3', 'bulk_modulus_calculations'}
                for part in parts:
                    if part not in skip_dirs and not part.startswith('V_') and part != 'OUTCAR':
                        if not part.replace('_', '').replace('-', '').isdigit():
                            compound = part
                            break
                if not vol_dir:
                    continue
            
            # Extract volume percentage
            vol_pct_match = re.search(r'V_([+-]?[\d.]+)%', vol_dir)
            if not vol_pct_match:
                continue
            vol_pct = float(vol_pct_match.group(1))
            
            # Extract volume and energy
            volume, energy = self.extract_volume_energy(outcar_path)
            
            if volume is not None and energy is not None:
                if compound not in self.data:
                    self.data[compound] = {}
                self.data[compound][vol_pct] = {
                    'volume': volume,
                    'energy': energy,
                    'path': str(outcar_path)
                }
        
        print(f"Extracted data for {len(self.data)} compounds")
        for compound, vols in self.data.items():
            print(f"  {compound}: {len(vols)} volume points")
    
    def birch_murnaghan_eos(self, V, E0, V0, B0, B0_prime):
        """Birch-Murnaghan equation of state."""
        V = np.array(V)
        eta = (V0 / V) ** (1/3)
        E = E0 + (9 * V0 * B0 / 16) * (
            (eta**2 - 1)**3 * B0_prime +
            (eta**2 - 1)**2 * (6 - 4 * eta**2)
        )
        return E
    
    def fit_eos(self, volumes, energies):
        """Fit energy-volume data to Birch-Murnaghan EOS."""
        volumes = np.array(volumes)
        energies = np.array(energies)
        
        if len(volumes) < 4:
            return None
        
        # Initial guesses
        E0_guess = np.min(energies)
        V0_guess = volumes[np.argmin(energies)]
        
        # Estimate B0
        if len(volumes) >= 5:
            idx_min = np.argmin(energies)
            idx_range = max(2, min(3, len(volumes) // 3))
            start_idx = max(0, idx_min - idx_range)
            end_idx = min(len(volumes), idx_min + idx_range + 1)
            V_local = volumes[start_idx:end_idx]
            E_local = energies[start_idx:end_idx]
            coeffs = np.polyfit(V_local, E_local, 2)
            B0_guess = abs(V0_guess * 2 * coeffs[0])
            B0_guess = max(0.1, min(B0_guess, 10.0))
        else:
            B0_guess = 1.0
        
        B0_prime_guess = 4.0
        
        # Bounds
        E0_bounds = (np.min(energies) - 100, np.max(energies) + 100)
        V0_bounds = (volumes.min() * 0.8, volumes.max() * 1.2)
        B0_bounds = (0.01, 20.0)
        B0_prime_bounds = (1.0, 10.0)
        
        bounds = ([E0_bounds[0], V0_bounds[0], B0_bounds[0], B0_prime_bounds[0]],
                  [E0_bounds[1], V0_bounds[1], B0_bounds[1], B0_prime_bounds[1]])
        
        try:
            popt, pcov = curve_fit(
                self.birch_murnaghan_eos,
                volumes,
                energies,
                p0=[E0_guess, V0_guess, B0_guess, B0_prime_guess],
                bounds=bounds,
                maxfev=20000,
                method='trf'
            )
            
            E0, V0, B0, B0_prime = popt
            
            # Calculate errors
            perr = np.sqrt(np.diag(pcov))
            
            # Calculate R-squared
            energies_fit = self.birch_murnaghan_eos(volumes, *popt)
            ss_res = np.sum((energies - energies_fit)**2)
            ss_tot = np.sum((energies - np.mean(energies))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'E0': E0,
                'V0': V0,
                'V0_err': perr[1],
                'r_squared': r_squared,
                'popt': popt
            }
            
        except Exception as e:
            return None
    
    def calculate_optimal_volumes(self):
        """Calculate optimal volumes for all compounds."""
        print("\n" + "="*100)
        print("CALCULATING OPTIMAL VOLUMES")
        print("="*100)
        
        for compound, data in sorted(self.data.items()):
            vol_pcts = sorted(data.keys())
            volumes = np.array([data[v]['volume'] for v in vol_pcts])
            energies = np.array([data[v]['energy'] for v in vol_pcts])
            
            fit_result = self.fit_eos(volumes, energies)
            
            if fit_result:
                V0 = fit_result['V0']
                V0_err = fit_result['V0_err']
                r_squared = fit_result['r_squared']
                self.optimal_volumes[compound] = V0
                print(f"{compound:<25} V₀ = {V0:>10.3f} ± {V0_err:>6.3f} Å³  (R² = {r_squared:.4f})")
            else:
                # Use minimum energy point
                idx_min = np.argmin(energies)
                V0_approx = volumes[idx_min]
                self.optimal_volumes[compound] = V0_approx
                print(f"{compound:<25} V₀ = {V0_approx:>10.3f} Å³  (approx, insufficient data)")
        
        print("="*100 + "\n")
    
    def read_poscar(self, poscar_path):
        """
        Read a POSCAR file and return its contents as structured data.
        
        Returns:
        --------
        dict : Parsed POSCAR data
        """
        with open(poscar_path, 'r') as f:
            lines = [line.rstrip() for line in f.readlines()]
        
        # Remove empty lines
        lines = [line for line in lines if line.strip()]
        
        if len(lines) < 8:
            raise ValueError(f"POSCAR file too short: {poscar_path}")
        
        title = lines[0]
        scaling = float(lines[1])
        
        # Read lattice vectors
        lattice = np.array([
            [float(x) for x in lines[2].split()],
            [float(x) for x in lines[3].split()],
            [float(x) for x in lines[4].split()]
        ])
        
        # Read element names and counts
        elements_line = lines[5].split()
        counts_line = lines[6].split()
        
        # Check if line 5 has element names or counts
        try:
            int(elements_line[0])
            # Line 5 is counts, no element names
            element_names = None
            element_counts = [int(x) for x in elements_line]
            coord_type_line = 7
        except ValueError:
            # Line 5 has element names
            element_names = elements_line
            element_counts = [int(x) for x in counts_line]
            coord_type_line = 7
        
        coord_type = lines[coord_type_line].strip()
        
        # Read coordinates
        n_atoms = sum(element_counts)
        coords_start = coord_type_line + 1
        coords = []
        for i in range(n_atoms):
            if coords_start + i < len(lines):
                coord_line = lines[coords_start + i].split()
                if len(coord_line) >= 3:
                    coords.append([float(x) for x in coord_line[:3]])
        
        return {
            'title': title,
            'scaling': scaling,
            'lattice': lattice,
            'element_names': element_names,
            'element_counts': element_counts,
            'coord_type': coord_type,
            'coordinates': np.array(coords),
            'all_lines': lines  # Keep original for any additional lines
        }
    
    def calculate_volume_from_lattice(self, lattice, scaling):
        """Calculate cell volume from lattice vectors and scaling factor."""
        return abs(np.linalg.det(lattice)) * scaling**3
    
    def scale_lattice_to_volume(self, lattice, scaling, target_volume):
        """
        Scale lattice vectors to achieve target volume.
        
        Parameters:
        -----------
        lattice : np.array
            3x3 lattice vector matrix
        scaling : float
            Current scaling factor
        target_volume : float
            Target volume in Å³
            
        Returns:
        --------
        tuple : (scaled_lattice, new_scaling)
        """
        current_volume = self.calculate_volume_from_lattice(lattice, scaling)
        scale_factor = (target_volume / current_volume) ** (1/3)
        
        # Scale lattice vectors
        scaled_lattice = lattice * scale_factor
        
        # Keep scaling factor at 1.0 (scale is applied to lattice vectors)
        new_scaling = 1.0
        
        return scaled_lattice, new_scaling
    
    def write_poscar(self, poscar_data, output_path):
        """
        Write POSCAR data to a file.
        
        Parameters:
        -----------
        poscar_data : dict
            Parsed POSCAR data
        output_path : Path
            Output file path
        """
        with open(output_path, 'w') as f:
            # Title
            f.write(f"{poscar_data['title']}\n")
            
            # Scaling factor
            f.write(f"  {poscar_data['scaling']:.10f}\n")
            
            # Lattice vectors
            for vec in poscar_data['lattice']:
                f.write(f"  {vec[0]:>18.10f}  {vec[1]:>18.10f}  {vec[2]:>18.10f}\n")
            
            # Element names (if present)
            if poscar_data['element_names']:
                f.write("  " + "  ".join(poscar_data['element_names']) + "\n")
            
            # Element counts
            f.write("  " + "  ".join(str(x) for x in poscar_data['element_counts']) + "\n")
            
            # Coordinate type
            f.write(f"{poscar_data['coord_type']}\n")
            
            # Coordinates
            for coord in poscar_data['coordinates']:
                f.write(f"  {coord[0]:>18.10f}  {coord[1]:>18.10f}  {coord[2]:>18.10f}\n")
            
            # Write any additional lines (selective dynamics, etc.)
            if 'all_lines' in poscar_data:
                n_atoms = sum(poscar_data['element_counts'])
                coord_start_idx = 8 if poscar_data['element_names'] else 7
                if len(poscar_data['all_lines']) > coord_start_idx + n_atoms:
                    for line in poscar_data['all_lines'][coord_start_idx + n_atoms:]:
                        f.write(f"{line}\n")
    
    def find_reference_poscar(self, compound):
        """
        Find a reference POSCAR file for a compound.
        Prefers V_+00.0% or closest to 0%.
        
        Parameters:
        -----------
        compound : str
            Compound name
            
        Returns:
        --------
        Path or None : Path to reference POSCAR file
        """
        # Look for POSCAR files in the compound directory
        compound_dirs = list(self.base_dir.rglob(f'*/{compound}/V_*'))
        
        if not compound_dirs:
            return None
        
        # Prefer V_+00.0% or closest to 0%
        best_dir = None
        best_vol_pct = float('inf')
        
        for dir_path in compound_dirs:
            dir_name = dir_path.name
            vol_match = re.search(r'V_([+-]?[\d.]+)%', dir_name)
            if vol_match:
                vol_pct = abs(float(vol_match.group(1)))
                if vol_pct < best_vol_pct:
                    best_vol_pct = vol_pct
                    best_dir = dir_path
        
        if best_dir:
            poscar_path = best_dir / 'POSCAR'
            if poscar_path.exists():
                return poscar_path
        
        # Fallback: try any POSCAR in the compound directory
        for dir_path in compound_dirs:
            poscar_path = dir_path / 'POSCAR'
            if poscar_path.exists():
                return poscar_path
        
        return None
    
    def write_optimal_poscars(self):
        """Write POSCAR files with optimal volumes for all compounds."""
        print("Writing POSCAR files with optimal volumes...")
        print("="*100)
        
        written_files = []
        
        for compound, optimal_volume in sorted(self.optimal_volumes.items()):
            # Find reference POSCAR
            ref_poscar = self.find_reference_poscar(compound)
            
            if not ref_poscar:
                print(f"  {compound:<25} SKIPPED (no reference POSCAR found)")
                continue
            
            try:
                # Read POSCAR
                poscar_data = self.read_poscar(ref_poscar)
                
                # Calculate current volume
                current_volume = self.calculate_volume_from_lattice(
                    poscar_data['lattice'], 
                    poscar_data['scaling']
                )
                
                # Scale to optimal volume
                scaled_lattice, new_scaling = self.scale_lattice_to_volume(
                    poscar_data['lattice'],
                    poscar_data['scaling'],
                    optimal_volume
                )
                
                # Update POSCAR data
                poscar_data['lattice'] = scaled_lattice
                poscar_data['scaling'] = new_scaling
                
                # Verify new volume
                new_volume = self.calculate_volume_from_lattice(scaled_lattice, new_scaling)
                
                # Determine output path (overwrite original POSCAR)
                if self.output_dir != self.base_dir:
                    # If output directory is different, preserve directory structure there
                    rel_path = ref_poscar.relative_to(self.base_dir)
                    
                    # Check if output_dir already contains base_dir name as subdirectory
                    base_name = self.base_dir.name
                    potential_subdir = self.output_dir / base_name
                    if potential_subdir.exists() and potential_subdir.is_dir():
                        # Output dir already has base_dir structure, use it
                        output_path = potential_subdir / rel_path
                    else:
                        # Normal case: write directly to output_dir
                        output_path = self.output_dir / rel_path
                else:
                    # Overwrite original POSCAR
                    output_path = ref_poscar
                
                # Create output directory if needed
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write POSCAR (overwrites existing file)
                self.write_poscar(poscar_data, output_path)
                
                written_files.append(str(output_path))
                print(f"  {compound:<25} {current_volume:>8.3f} → {optimal_volume:>8.3f} Å³  ({output_path})")
                
            except Exception as e:
                print(f"  {compound:<25} ERROR: {e}")
        
        print("="*100)
        print(f"\nSuccessfully wrote {len(written_files)} POSCAR file(s)")
        
        return written_files


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Overwrite POSCAR files with optimal volumes from EOS fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Overwrite POSCARs in the same directory structure
  python write_optimal_volume_poscars.py /path/to/bulk_modulus_calculations/
  
  # Write to a different output directory (preserves structure, overwrites POSCARs there)
  python write_optimal_volume_poscars.py /path/to/bulk_modulus_calculations/ -o /path/to/output/
        """
    )
    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory containing bulk modulus calculations'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory for POSCAR files (default: same as base_dir)'
    )
    
    args = parser.parse_args()
    
    # Initialize writer
    writer = OptimalVolumePOSCARWriter(args.base_dir, args.output)
    
    # Parse directory structure
    writer.parse_directory_structure()
    
    if not writer.data:
        print("No data found! Check your directory structure.")
        return
    
    # Calculate optimal volumes
    writer.calculate_optimal_volumes()
    
    if not writer.optimal_volumes:
        print("No optimal volumes calculated!")
        return
    
    # Write POSCAR files
    writer.write_optimal_poscars()


if __name__ == '__main__':
    main()

