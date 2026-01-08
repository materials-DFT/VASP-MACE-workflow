#!/usr/bin/env python3
"""
Bulk Modulus Analysis Script for VASP Calculations (Per Atom Values)

This script analyzes bulk modulus calculations from VASP OUTCAR files.
It extracts volume and energy data, converts to per-atom values,
fits to Birch-Murnaghan EOS, and generates comprehensive plots.
All data is plotted without removing outliers.

Key Parameters (per atom):
- B0: Bulk modulus (GPa)
- V0: Equilibrium volume per atom (Å³/atom)
- B0': Pressure derivative of bulk modulus (dimensionless)
- E0: Equilibrium energy per atom (eV/atom)
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from collections import defaultdict


class BulkModulusAnalyzer:
    """Analyze bulk modulus calculations from VASP OUTCAR files (per atom)."""
    
    def __init__(self, base_dir):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing bulk modulus calculations
        """
        self.base_dir = Path(base_dir)
        self.data = defaultdict(dict)  # {compound: {volume: {energy, volume, num_atoms}}}
        
    def extract_volume_energy(self, outcar_path):
        """
        Extract volume and energy from OUTCAR file.
        
        Parameters:
        -----------
        outcar_path : str or Path
            Path to OUTCAR file
            
        Returns:
        --------
        tuple : (volume, energy) or (None, None) if extraction fails
        """
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
                
            # Extract volume - look for "volume of cell :      XXX.XX"
            vol_pattern = r'volume of cell\s*:\s*([\d.]+)'
            vol_match = re.findall(vol_pattern, content)
            if vol_match:
                volume = float(vol_match[-1])  # Take the last occurrence (final volume)
            else:
                return None, None
                
            # Extract energy - look for final "free energy    TOTEN  =     -XXXX.XXXXX eV"
            energy_pattern = r'free\s+energy\s+TOTEN\s+=\s+([\d.-]+)\s+eV'
            energy_matches = re.findall(energy_pattern, content)
            if energy_matches:
                energy = float(energy_matches[-1])  # Take the last occurrence (final energy)
            else:
                return None, None
                
            return volume, energy
            
        except Exception as e:
            print(f"Error reading {outcar_path}: {e}")
            return None, None
    
    def parse_poscar(self, poscar_path):
        """Parse POSCAR to get number of atoms."""
        try:
            with open(poscar_path, 'r') as f:
                lines = f.readlines()
            
            # Find line with element names (typically line 5 or 6)
            atom_line = None
            for i, line in enumerate(lines[5:8], start=5):
                if re.match(r'^\s*[A-Za-z]', line):
                    atom_line = i
                    break
            
            if atom_line is None:
                atom_line = 5
            
            # Get element names and counts
            counts = list(map(int, lines[atom_line + 1].split()))
            total_atoms = sum(counts)
            return total_atoms
        except Exception as e:
            return None
    
    def parse_directory_structure(self):
        """Parse directory structure and extract data from all OUTCAR files."""
        print("Scanning directory structure...")
        
        # Find all OUTCAR files
        outcar_files = list(self.base_dir.rglob('OUTCAR'))
        print(f"Found {len(outcar_files)} OUTCAR files")
        
        for outcar_path in outcar_files:
            # Extract compound name and volume from path
            # Path structure: base_dir/compound/.../V_XX.X%/OUTCAR
            parts = outcar_path.parts
            
            # Find volume directory (V_XX.X%)
            vol_dir = None
            compound = None
            
            for i, part in enumerate(parts):
                if part.startswith('V_'):
                    vol_dir = part
                    # Compound is typically 2-3 levels up
                    if i >= 2:
                        compound = parts[i-1]
                    elif i >= 1:
                        compound = parts[i-1]
                    break
            
            if not vol_dir or not compound:
                # Try alternative: look for compound name in path
                for part in parts:
                    if part not in ['optimization', 'neutral', 'monkhorst-pack_calculated', 
                                   'unitcells', 'isif3'] and not part.startswith('V_'):
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
                # Get number of atoms from POSCAR in same directory
                poscar_path = outcar_path.parent / 'POSCAR'
                num_atoms = self.parse_poscar(poscar_path) if poscar_path.exists() else None
                
                if compound not in self.data:
                    self.data[compound] = {}
                self.data[compound][vol_pct] = {
                    'volume': volume,
                    'energy': energy,
                    'num_atoms': num_atoms,
                    'path': str(outcar_path)
                }
        
        print(f"Extracted data for {len(self.data)} compounds")
        for compound, vols in self.data.items():
            print(f"  {compound}: {len(vols)} volume points")
    
    def birch_murnaghan_eos(self, V, E0, V0, B0, B0_prime):
        """
        Birch-Murnaghan equation of state.
        
        Parameters:
        -----------
        V : array-like
            Volume (Å³)
        E0 : float
            Equilibrium energy (eV)
        V0 : float
            Equilibrium volume (Å³)
        B0 : float
            Bulk modulus (eV/Å³)
        B0_prime : float
            Pressure derivative of bulk modulus (dimensionless)
            
        Returns:
        --------
        array-like : Energy (eV)
        """
        V = np.array(V)
        eta = (V0 / V) ** (1/3)
        
        # Birch-Murnaghan EOS (3rd order)
        E = E0 + (9 * V0 * B0 / 16) * (
            (eta**2 - 1)**3 * B0_prime +
            (eta**2 - 1)**2 * (6 - 4 * eta**2)
        )
        
        return E
    
    def fit_eos(self, volumes, energies):
        """
        Fit energy-volume data to Birch-Murnaghan EOS.
        
        Parameters:
        -----------
        volumes : array-like
            Volumes (Å³)
        energies : array-like
            Energies (eV)
            
        Returns:
        --------
        dict : Fitted parameters and statistics
        """
        volumes = np.array(volumes)
        energies = np.array(energies)
        
        # Need at least 4 points for 4-parameter fit
        if len(volumes) < 4:
            print(f"Warning: Only {len(volumes)} data points, need at least 4 for EOS fit")
            return None
        
        # Initial guesses
        E0_guess = np.min(energies)
        V0_guess = volumes[np.argmin(energies)]
        
        # Estimate B0 from curvature around minimum
        # Use polynomial fit to estimate second derivative
        if len(volumes) >= 5:
            # Fit a quadratic around the minimum
            idx_min = np.argmin(energies)
            idx_range = max(2, min(3, len(volumes) // 3))
            start_idx = max(0, idx_min - idx_range)
            end_idx = min(len(volumes), idx_min + idx_range + 1)
            V_local = volumes[start_idx:end_idx]
            E_local = energies[start_idx:end_idx]
            
            # Fit quadratic: E = a*V^2 + b*V + c
            coeffs = np.polyfit(V_local, E_local, 2)
            # B0 ≈ V0 * d²E/dV² at V0
            # d²E/dV² = 2*a
            # B0 = V0 * 2*a (in eV/Å³)
            B0_guess = abs(V0_guess * 2 * coeffs[0])
            B0_guess = max(0.1, min(B0_guess, 10.0))  # Reasonable bounds
        else:
            B0_guess = 1.0  # Default guess
        
        B0_prime_guess = 4.0  # Typical value
        
        # Set bounds for parameters
        # E0: within range of energies
        E0_bounds = (np.min(energies) - 100, np.max(energies) + 100)
        # V0: within range of volumes
        V0_bounds = (volumes.min() * 0.8, volumes.max() * 1.2)
        # B0: reasonable range (0.01 to 20 eV/Å³)
        B0_bounds = (0.01, 20.0)
        # B0_prime: typically 2-8
        B0_prime_bounds = (1.0, 10.0)
        
        bounds = ([E0_bounds[0], V0_bounds[0], B0_bounds[0], B0_prime_bounds[0]],
                  [E0_bounds[1], V0_bounds[1], B0_bounds[1], B0_prime_bounds[1]])
        
        # Fit
        try:
            popt, pcov = curve_fit(
                self.birch_murnaghan_eos,
                volumes,
                energies,
                p0=[E0_guess, V0_guess, B0_guess, B0_prime_guess],
                bounds=bounds,
                maxfev=20000,
                method='trf'  # Trust Region Reflective algorithm
            )
            
            E0, V0, B0, B0_prime = popt
            
            # Calculate errors
            perr = np.sqrt(np.diag(pcov))
            
            # Convert B0 to GPa
            B0_GPa = B0 * 160.21766208
            
            # Calculate R-squared
            energies_fit = self.birch_murnaghan_eos(volumes, *popt)
            ss_res = np.sum((energies - energies_fit)**2)
            ss_tot = np.sum((energies - np.mean(energies))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {
                'E0': E0,
                'V0': V0,
                'B0': B0,
                'B0_GPa': B0_GPa,
                'B0_prime': B0_prime,
                'E0_err': perr[0],
                'V0_err': perr[1],
                'B0_err': perr[2] * 160.21766208,  # Convert to GPa
                'B0_prime_err': perr[3],
                'r_squared': r_squared,
                'popt': popt
            }
            
        except Exception as e:
            print(f"Error fitting EOS: {e}")
            return None
    
    def plot_all_compounds(self):
        """Plot all compounds on a single Energy per atom vs Volume per atom plot with EOS fits.
        Energies are normalized (shifted) so each compound's minimum energy is at 0 eV/atom.
        All data is plotted without removing any outliers.
        """
        if not self.data:
            print("No data to plot")
            return
        
        n_compounds = len(self.data)
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Generate colors for each compound
        colors = plt.cm.tab20(np.linspace(0, 1, n_compounds))
        
        # Collect all normalized energies
        all_normalized_energies = []
        plot_data = []
        
        # Process all data: convert to per-atom, normalize and prepare for plotting
        for i, (compound, data) in enumerate(self.data.items()):
            vol_pcts = sorted(data.keys())
            
            # Get volumes and energies, convert to per-atom
            volumes_list = []
            energies_list = []
            num_atoms_list = []
            
            for v in vol_pcts:
                vol = data[v]['volume']
                en = data[v]['energy']
                natoms = data[v]['num_atoms']
                
                if natoms is not None and natoms > 0:
                    volumes_list.append(vol / natoms)
                    energies_list.append(en / natoms)
                    num_atoms_list.append(natoms)
            
            if len(volumes_list) == 0:
                continue
            
            volumes = np.array(volumes_list)
            energies = np.array(energies_list)
            
            # Fit EOS to get E0 (equilibrium energy)
            fit_result = self.fit_eos(volumes, energies)
            
            if fit_result:
                E0 = fit_result['E0']
                energies_normalized = energies - E0
                all_normalized_energies.extend(energies_normalized.tolist())
                
                # Store plot data
                V_fit = np.linspace(volumes.min() * 0.95, volumes.max() * 1.05, 200)
                E_fit = self.birch_murnaghan_eos(V_fit, *fit_result['popt'])
                E_fit_normalized = E_fit - E0
                
                plot_data.append({
                    'compound': compound,
                    'volumes': volumes,
                    'energies_norm': energies_normalized,
                    'V_fit': V_fit,
                    'E_fit_norm': E_fit_normalized,
                    'color': colors[i],
                    'fit_result': fit_result
                })
            else:
                # If fit failed, just normalize by minimum energy
                E0 = np.min(energies)
                energies_normalized = energies - E0
                all_normalized_energies.extend(energies_normalized.tolist())
                
                plot_data.append({
                    'compound': compound,
                    'volumes': volumes,
                    'energies_norm': energies_normalized,
                    'V_fit': None,
                    'E_fit_norm': None,
                    'color': colors[i],
                    'fit_result': None
                })
        
        # Calculate y-axis limits from all data (use full range to show all points)
        all_normalized_energies = np.array(all_normalized_energies)
        if len(all_normalized_energies) > 0:
            y_min = np.min(all_normalized_energies)
            y_max = np.max(all_normalized_energies)
            y_range = max(abs(y_min), abs(y_max))
            y_limit = y_range * 1.1  # Add 10% padding
        else:
            y_limit = 10.0  # Fallback
        
        # Plot each compound (all data, no filtering)
        print(f"\nPlotting {len(plot_data)} compound(s):")
        for data in plot_data:
            compound = data['compound']
            volumes = data['volumes']
            energies_norm = data['energies_norm']
            color = data['color']
            
            print(f"  '{compound}': {len(volumes)} points, energy range: [{np.min(energies_norm):.3f}, {np.max(energies_norm):.3f}] eV/atom")
            
            # Plot normalized data points
            # Use larger markers and different style for compounds without EOS fit to make them more visible
            if data['V_fit'] is not None and data['E_fit_norm'] is not None:
                # Standard markers for compounds with EOS fit
                ax.scatter(volumes, energies_norm, s=80, alpha=0.7, 
                          label=compound, color=color, zorder=3)
            else:
                # Larger, edge-highlighted markers for compounds without EOS fit
                ax.scatter(volumes, energies_norm, s=120, alpha=0.8, 
                          label=compound, color=color, zorder=3,
                          edgecolors='black', linewidths=1.5, marker='s')
            
            # Plot normalized fitted curve if available
            if data['V_fit'] is not None and data['E_fit_norm'] is not None:
                V_fit = data['V_fit']
                E_fit_norm = data['E_fit_norm']
                ax.plot(V_fit, E_fit_norm, '-', linewidth=2, color=color, 
                       alpha=0.7, zorder=2)
                print(f"    -> Fitted curve plotted")
            else:
                print(f"    -> No fitted curve (insufficient points or fit failed)")
        
        # Set y-axis limits (symmetric around zero)
        ax.set_ylim(-y_limit, y_limit)
        
        ax.set_xlabel('Volume per atom (Å³/atom)', fontsize=14)
        ax.set_ylabel('Energy per atom - E₀ (eV/atom)', fontsize=14)
        ax.set_title('Bulk Modulus: Normalized Energy per Atom vs Volume per Atom\n(All data included, no outliers removed)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=8, ncol=3, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add text annotation about y-axis limits
        annotation_text = f'Y-axis range: ±{y_limit:.1f} eV/atom\n(All {n_compounds} compound(s) plotted)'
        ax.text(0.02, 0.98, annotation_text, 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_table(self):
        """Generate a summary table of all fitted parameters (per atom)."""
        results = []
        
        for compound, data in self.data.items():
            vol_pcts = sorted(data.keys())
            
            # Convert to per-atom values
            volumes_list = []
            energies_list = []
            
            for v in vol_pcts:
                vol = data[v]['volume']
                en = data[v]['energy']
                natoms = data[v]['num_atoms']
                
                if natoms is not None and natoms > 0:
                    volumes_list.append(vol / natoms)
                    energies_list.append(en / natoms)
            
            if len(volumes_list) == 0:
                continue
            
            volumes = np.array(volumes_list)
            energies = np.array(energies_list)
            fit_result = self.fit_eos(volumes, energies)
            
            if fit_result:
                results.append({
                    'Compound': compound,
                    'B₀ (GPa)': f"{fit_result['B0_GPa']:.2f} ± {fit_result['B0_err']:.2f}",
                    'V₀ (Å³/atom)': f"{fit_result['V0']:.4f} ± {fit_result['V0_err']:.4f}",
                    "B₀'": f"{fit_result['B0_prime']:.2f} ± {fit_result['B0_prime_err']:.2f}",
                    'E₀ (eV/atom)': f"{fit_result['E0']:.4f} ± {fit_result['E0_err']:.4f}",
                    'R²': f"{fit_result['r_squared']:.4f}",
                    'N_points': len(volumes)
                })
        
        # Print table
        print("\n" + "="*100)
        print("BULK MODULUS ANALYSIS SUMMARY (Per Atom)")
        print("="*100)
        print(f"{'Compound':<20} {'B₀ (GPa)':<20} {'V₀ (Å³/atom)':<18} {'B₀\'':<15} {'E₀ (eV/atom)':<18} {'R²':<10} {'N':<5}")
        print("-"*100)
        
        for r in results:
            print(f"{r['Compound']:<20} {r['B₀ (GPa)']:<20} {r['V₀ (Å³/atom)']:<18} "
                  f"{r['B₀\'']:<15} {r['E₀ (eV/atom)']:<18} {r['R²']:<10} {r['N_points']:<5}")
        
        print("="*100 + "\n")
        
        return results


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze bulk modulus calculations from VASP OUTCAR files (per atom values, no outlier removal)'
    )
    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory containing bulk modulus calculations'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BulkModulusAnalyzer(args.base_dir)
    
    # Parse directory structure
    analyzer.parse_directory_structure()
    
    if not analyzer.data:
        print("No data found! Check your directory structure.")
        return
    
    # Generate summary
    analyzer.generate_summary_table()
    
    # Plot all compounds on single plot
    analyzer.plot_all_compounds()


if __name__ == '__main__':
    main()
