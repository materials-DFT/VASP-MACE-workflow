#!/usr/bin/env python3
"""
Bulk Modulus Analysis Script with ±20% Volume Filter

This script analyzes bulk modulus calculations from VASP OUTCAR files.
It:
1. Scans all directories recursively for OUTCAR files
2. Finds energy minima per structure
3. Filters data to only include points within ±20% expansion/compression range
4. Plots the filtered data with EOS fits

Key Parameters:
- B0: Bulk modulus (GPa)
- V0: Equilibrium volume (Å³) - determined from energy minimum
- B0': Pressure derivative of bulk modulus (dimensionless)
- E0: Equilibrium energy (eV) - energy at minimum
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for separate X11 windows
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from collections import defaultdict


class FilteredBulkModulusAnalyzer:
    """Analyze bulk modulus calculations with ±20% volume filtering."""
    
    def __init__(self, base_dir):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing bulk modulus calculations
        """
        self.base_dir = Path(base_dir)
        self.data = defaultdict(dict)  # {compound: {volume: {energy, volume, path}}}
        self.filtered_data = defaultdict(dict)  # {compound: {volume: {energy, volume, path}}}
        self.equilibrium = {}  # {compound: {V0, E0, vol_pct_min}}
        
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
    
    def parse_directory_structure(self):
        """Parse directory structure and extract data from all OUTCAR files."""
        print("Scanning directory structure recursively...")
        
        # Find all OUTCAR files recursively
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
                                   'unitcells', 'isif3', 'OUTCAR'] and not part.startswith('V_'):
                        if compound is None:
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
        
        print(f"Extracted data for {len(self.data)} structures")
        for compound, vols in self.data.items():
            print(f"  {compound}: {len(vols)} volume points")
    
    def find_energy_minima(self):
        """
        Find energy minima for each structure and determine equilibrium volume (V0).
        """
        print("\nFinding energy minima per structure...")
        
        for compound, data in self.data.items():
            vol_pcts = sorted(data.keys())
            volumes = np.array([data[v]['volume'] for v in vol_pcts])
            energies = np.array([data[v]['energy'] for v in vol_pcts])
            
            # Find index of minimum energy
            min_idx = np.argmin(energies)
            E0 = energies[min_idx]
            V0 = volumes[min_idx]
            vol_pct_min = vol_pcts[min_idx]
            
            self.equilibrium[compound] = {
                'V0': V0,
                'E0': E0,
                'vol_pct_min': vol_pct_min
            }
            
            print(f"  {compound}: E_min = {E0:.6f} eV at V = {V0:.2f} Å³ (V_{vol_pct_min:.1f}%)")
    
    def filter_by_volume_range(self, tolerance=0.20):
        """
        Filter data to only include points within ±tolerance (default ±20%) of equilibrium volume.
        
        Parameters:
        -----------
        tolerance : float
            Fractional tolerance (0.20 = ±20%)
        """
        print(f"\nFiltering data to ±{tolerance*100:.0f}% of equilibrium volume...")
        
        for compound, data in self.data.items():
            if compound not in self.equilibrium:
                continue
            
            V0 = self.equilibrium[compound]['V0']
            V_min = V0 * (1 - tolerance)  # -20% compression
            V_max = V0 * (1 + tolerance)  # +20% expansion
            
            filtered = {}
            for vol_pct, point_data in data.items():
                volume = point_data['volume']
                if V_min <= volume <= V_max:
                    filtered[vol_pct] = point_data
            
            self.filtered_data[compound] = filtered
            
            original_count = len(data)
            filtered_count = len(filtered)
            print(f"  {compound}: {filtered_count}/{original_count} points within ±{tolerance*100:.0f}% range")
    
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
    
    def get_target_compression_percentages(self):
        """
        Get target compression/expansion percentages for consistent annotation across structures.
        Returns evenly spaced percentages within ±20% range.
        
        Returns:
        --------
        list : Target compression/expansion percentages
        """
        # Evenly spaced from -20% to +20% (approximately 10 points)
        # Always include -20%, 0%, +20%, and evenly space the rest
        return [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    
    def plot_individual_compound(self, compound, data):
        """
        Plot a single compound with filtered data (±20% range) and EOS fit.
        Energies are normalized (shifted) so the minimum energy is at 0 eV.
        
        Parameters:
        -----------
        compound : str
            Name of the compound/phase
        data : dict
            Dictionary containing volume and energy data for the compound (filtered)
        """
        vol_pcts = sorted(data.keys())
        volumes = np.array([data[v]['volume'] for v in vol_pcts])
        energies = np.array([data[v]['energy'] for v in vol_pcts])
        
        # Get equilibrium values
        eq = self.equilibrium.get(compound, {})
        V0_eq = eq.get('V0', volumes[np.argmin(energies)])
        E0_eq = eq.get('E0', np.min(energies))
        
        # Create a new figure for this compound
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Fit EOS to filtered data
        fit_result = self.fit_eos(volumes, energies)
        
        if fit_result:
            E0 = fit_result['E0']
            energies_normalized = energies - E0
            
            # Generate fitted curve
            V_fit = np.linspace(volumes.min() * 0.95, volumes.max() * 1.05, 200)
            E_fit = self.birch_murnaghan_eos(V_fit, *fit_result['popt'])
            E_fit_normalized = E_fit - E0
            
            # Plot normalized fitted curve
            ax.plot(V_fit, E_fit_normalized, '-', linewidth=2.5, color='blue', 
                   alpha=0.8, zorder=2, label='Birch-Murnaghan EOS fit')
            
            # Calculate y-axis limits
            y_min = np.min(energies_normalized)
            y_max = np.max(energies_normalized)
            y_range = max(abs(y_min), abs(y_max))
            y_limit = y_range * 1.15  # Add 15% padding
            
            # Plot normalized data points
            ax.scatter(volumes, energies_normalized, s=100, alpha=0.8, 
                      color='red', zorder=3, edgecolors='black', 
                      linewidths=1.5, label=f'Data points (±20% range, N={len(volumes)})')
            
            # Calculate compression/expansion percentages and select points to annotate
            compression_expansion = ((volumes - V0_eq) / V0_eq) * 100  # Percentage relative to equilibrium
            
            # Get target compression percentages (consistent across all structures)
            target_percentages = self.get_target_compression_percentages()
            
            # Find points closest to target percentages
            annotation_indices = []
            for target_pct in target_percentages:
                # Find index of point closest to target percentage
                distances = np.abs(compression_expansion - target_pct)
                closest_idx = np.argmin(distances)
                annotation_indices.append(closest_idx)
            
            # Remove duplicates while preserving order
            annotation_indices = sorted(list(set(annotation_indices)))
            
            # Annotate selected points with compression/expansion percentage
            for idx in annotation_indices:
                vol = volumes[idx]
                en = energies_normalized[idx]
                comp_exp_pct = compression_expansion[idx]
                
                # Add text annotation with compression/expansion percentage
                if comp_exp_pct < 0:
                    label = f'{comp_exp_pct:.1f}%'
                    ha = 'right'
                    offset_x = -0.01 * (volumes.max() - volumes.min())
                else:
                    label = f'+{comp_exp_pct:.1f}%'
                    ha = 'left'
                    offset_x = 0.01 * (volumes.max() - volumes.min())
                
                ax.annotate(label, xy=(vol, en), xytext=(vol + offset_x, en + 0.05 * y_limit),
                           fontsize=9, ha=ha, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='blue'),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                                         color='blue', lw=1.5),
                           zorder=5)
            
            # Add vertical lines showing ±20% range
            ax.axvline(x=V0_eq * 0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='±20% range')
            ax.axvline(x=V0_eq * 1.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=V0_eq, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Equilibrium (min energy)')
            
            print(f"  '{compound}': {len(volumes)} filtered points, B₀ = {fit_result['B0_GPa']:.2f} ± {fit_result['B0_err']:.2f} GPa, R² = {fit_result['r_squared']:.4f}")
        else:
            # If fit failed, just normalize by minimum energy
            E0 = np.min(energies)
            energies_normalized = energies - E0
            
            # Calculate y-axis limits
            y_min = np.min(energies_normalized)
            y_max = np.max(energies_normalized)
            y_range = max(abs(y_min), abs(y_max))
            y_limit = y_range * 1.15  # Add 15% padding
            
            # Plot normalized data points only
            ax.scatter(volumes, energies_normalized, s=120, alpha=0.8, 
                      color='red', zorder=3, edgecolors='black', 
                      linewidths=1.5, marker='s', label=f'Data points (±20% range, N={len(volumes)})')
            
            # Calculate compression/expansion percentages and select points to annotate
            compression_expansion = ((volumes - V0_eq) / V0_eq) * 100  # Percentage relative to equilibrium
            
            # Get target compression percentages (consistent across all structures)
            target_percentages = self.get_target_compression_percentages()
            
            # Find points closest to target percentages
            annotation_indices = []
            for target_pct in target_percentages:
                # Find index of point closest to target percentage
                distances = np.abs(compression_expansion - target_pct)
                closest_idx = np.argmin(distances)
                annotation_indices.append(closest_idx)
            
            # Remove duplicates while preserving order
            annotation_indices = sorted(list(set(annotation_indices)))
            
            # Annotate selected points with compression/expansion percentage
            for idx in annotation_indices:
                vol = volumes[idx]
                en = energies_normalized[idx]
                comp_exp_pct = compression_expansion[idx]
                
                # Add text annotation with compression/expansion percentage
                if comp_exp_pct < 0:
                    label = f'{comp_exp_pct:.1f}%'
                    ha = 'right'
                    offset_x = -0.01 * (volumes.max() - volumes.min())
                else:
                    label = f'+{comp_exp_pct:.1f}%'
                    ha = 'left'
                    offset_x = 0.01 * (volumes.max() - volumes.min())
                
                ax.annotate(label, xy=(vol, en), xytext=(vol + offset_x, en + 0.05 * y_limit),
                           fontsize=9, ha=ha, va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='blue'),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                                         color='blue', lw=1.5),
                           zorder=5)
            
            # Add vertical lines showing ±20% range
            ax.axvline(x=V0_eq * 0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='±20% range')
            ax.axvline(x=V0_eq * 1.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.axvline(x=V0_eq, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Equilibrium (min energy)')
            
            print(f"  '{compound}': {len(volumes)} filtered points, EOS fit failed (insufficient points or fit error)")
        
        # Set y-axis limits (symmetric around zero)
        ax.set_ylim(-y_limit, y_limit)
        
        # Set labels and title
        ax.set_xlabel('Volume (Å³)', fontsize=14)
        ax.set_ylabel('Energy - E₀ (eV)', fontsize=14)
        ax.set_title(f'Bulk Modulus: {compound}\nFiltered Data (±20% range) - Normalized Energy vs Volume', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        
        # Set window title
        fig.canvas.manager.set_window_title(f'Bulk Modulus (Filtered): {compound}')
        
        # Show the plot in a separate window (non-blocking)
        plt.show(block=False)
        
        return fig
    
    def plot_all_compounds_individual(self):
        """
        Plot each compound individually in separate X11 windows.
        Only filtered data (±20% range) is plotted.
        """
        if not self.filtered_data:
            print("No filtered data to plot")
            return
        
        print(f"\nPlotting {len(self.filtered_data)} structure(s) with filtered data in separate windows:")
        
        # Plot each compound in its own window
        for compound, data in self.filtered_data.items():
            if len(data) > 0:  # Only plot if there's filtered data
                self.plot_individual_compound(compound, data)
        
        # Keep all windows open
        print("\nAll plots are displayed in separate windows. Close windows manually or press Ctrl+C to exit.")
        try:
            plt.show(block=True)  # Block to keep windows open
        except KeyboardInterrupt:
            print("\nInterrupted by user. Closing plots...")
            plt.close('all')
    
    def generate_summary_table(self):
        """Generate a summary table of all fitted parameters for filtered data."""
        results = []
        
        for compound, data in self.filtered_data.items():
            if len(data) == 0:
                continue
                
            vol_pcts = sorted(data.keys())
            volumes = np.array([data[v]['volume'] for v in vol_pcts])
            energies = np.array([data[v]['energy'] for v in vol_pcts])
            fit_result = self.fit_eos(volumes, energies)
            
            eq = self.equilibrium.get(compound, {})
            V0_min = eq.get('V0', volumes[np.argmin(energies)])
            
            if fit_result:
                results.append({
                    'Compound': compound,
                    'B₀ (GPa)': f"{fit_result['B0_GPa']:.2f} ± {fit_result['B0_err']:.2f}",
                    'V₀ (Å³)': f"{fit_result['V0']:.2f} ± {fit_result['V0_err']:.2f}",
                    'V₀ (min)': f"{V0_min:.2f}",
                    "B₀'": f"{fit_result['B0_prime']:.2f} ± {fit_result['B0_prime_err']:.2f}",
                    'E₀ (eV)': f"{fit_result['E0']:.2f} ± {fit_result['E0_err']:.2f}",
                    'R²': f"{fit_result['r_squared']:.4f}",
                    'N_points': len(volumes)
                })
        
        # Print table
        print("\n" + "="*120)
        print("BULK MODULUS ANALYSIS SUMMARY (Filtered: ±20% range)")
        print("="*120)
        print(f"{'Compound':<20} {'B₀ (GPa)':<20} {'V₀ (Å³)':<15} {'V₀ (min)':<15} {'B₀\'':<15} {'E₀ (eV)':<15} {'R²':<10} {'N':<5}")
        print("-"*120)
        
        for r in results:
            print(f"{r['Compound']:<20} {r['B₀ (GPa)']:<20} {r['V₀ (Å³)']:<15} {r['V₀ (min)']:<15} "
                  f"{r['B₀\'']:<15} {r['E₀ (eV)']:<15} {r['R²']:<10} {r['N_points']:<5}")
        
        print("="*120 + "\n")
        
        return results


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze bulk modulus calculations with ±20% volume filtering. Finds energy minima per structure and plots only filtered data.'
    )
    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory containing bulk modulus calculations (will be scanned recursively)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.20,
        help='Volume tolerance as fraction (default: 0.20 = ±20%%)'
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FilteredBulkModulusAnalyzer(args.base_dir)
    
    # Parse directory structure
    analyzer.parse_directory_structure()
    
    if not analyzer.data:
        print("No data found! Check your directory structure.")
        return
    
    # Find energy minima per structure
    analyzer.find_energy_minima()
    
    # Filter data to ±20% range
    analyzer.filter_by_volume_range(tolerance=args.tolerance)
    
    if not analyzer.filtered_data:
        print("No data within the specified range! Try adjusting the tolerance.")
        return
    
    # Generate summary
    analyzer.generate_summary_table()
    
    # Plot each compound individually in separate windows
    analyzer.plot_all_compounds_individual()


if __name__ == '__main__':
    main()
