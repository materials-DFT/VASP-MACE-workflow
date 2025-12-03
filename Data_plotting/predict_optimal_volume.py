#!/usr/bin/env python3
"""
Optimal Volume Prediction Script for VASP Calculations

This script analyzes bulk modulus calculations from VASP OUTCAR files.
It extracts volume and energy data, fits to Birch-Murnaghan EOS,
and predicts the optimal cell volume (V0) that corresponds to the energy minimum.

Key Output:
- V0: Predicted optimal equilibrium volume (Å³) - the volume at energy minimum
- E0: Predicted equilibrium energy (eV) at V0
- B0: Bulk modulus (GPa)
- Quality metrics: R², number of data points
"""

import os
import re
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from collections import defaultdict


class OptimalVolumePredictor:
    """Predict optimal cell volumes from VASP bulk modulus calculations."""
    
    def __init__(self, base_dir):
        """
        Initialize the predictor.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing bulk modulus calculations
        """
        self.base_dir = Path(base_dir)
        self.data = defaultdict(dict)  # {compound: {volume: {energy, volume}}}
        
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
        print("Scanning directory structure...")
        
        # Find all OUTCAR files
        outcar_files = list(self.base_dir.rglob('OUTCAR'))
        print(f"Found {len(outcar_files)} OUTCAR files")
        
        for outcar_path in outcar_files:
            # Extract compound name and volume from path
            # Path structure: base_dir/compound/.../V_XX.X%/OUTCAR
            # Get relative path from base_dir to understand structure better
            try:
                rel_path = outcar_path.relative_to(self.base_dir)
                parts = rel_path.parts
            except ValueError:
                # If path is not relative to base_dir, use absolute path parts
                parts = outcar_path.parts
            
            # Find volume directory (V_XX.X%)
            vol_dir = None
            compound = None
            
            for i, part in enumerate(parts):
                if part.startswith('V_'):
                    vol_dir = part
                    # Compound is typically 1 level up from V_ directory
                    if i >= 1:
                        compound = parts[i-1]
                    break
            
            if not vol_dir or not compound:
                # Try alternative: look for compound name in path
                # Skip common directory names
                skip_dirs = {'optimization', 'neutral', 'monkhorst-pack_calculated', 
                            'unitcells', 'isif3', 'bulk_modulus_calculations'}
                for part in parts:
                    if part not in skip_dirs and not part.startswith('V_') and part != 'OUTCAR':
                        # Check if this looks like a compound name (not a number, not a common dir)
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
        dict : Fitted parameters and statistics, or None if fit fails
        """
        volumes = np.array(volumes)
        energies = np.array(energies)
        
        # Need at least 4 points for 4-parameter fit
        if len(volumes) < 4:
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
            return None
    
    def predict_optimal_volumes(self):
        """
        Predict optimal volumes for all compounds.
        
        Returns:
        --------
        list : List of dictionaries with prediction results
        """
        results = []
        
        print("\n" + "="*100)
        print("OPTIMAL VOLUME PREDICTIONS")
        print("="*100)
        print(f"{'Compound':<25} {'V₀ (Å³)':<20} {'V₀ Error':<15} {'E₀ (eV)':<15} {'B₀ (GPa)':<15} {'R²':<10} {'N':<5}")
        print("-"*100)
        
        for compound, data in sorted(self.data.items()):
            vol_pcts = sorted(data.keys())
            volumes = np.array([data[v]['volume'] for v in vol_pcts])
            energies = np.array([data[v]['energy'] for v in vol_pcts])
            
            # Fit EOS to predict optimal volume
            fit_result = self.fit_eos(volumes, energies)
            
            if fit_result:
                V0 = fit_result['V0']
                V0_err = fit_result['V0_err']
                E0 = fit_result['E0']
                B0_GPa = fit_result['B0_GPa']
                r_squared = fit_result['r_squared']
                n_points = len(volumes)
                
                results.append({
                    'compound': compound,
                    'V0': V0,
                    'V0_err': V0_err,
                    'E0': E0,
                    'E0_err': fit_result['E0_err'],
                    'B0_GPa': B0_GPa,
                    'B0_err': fit_result['B0_err'],
                    'B0_prime': fit_result['B0_prime'],
                    'r_squared': r_squared,
                    'n_points': n_points,
                    'volumes': volumes,
                    'energies': energies
                })
                
                print(f"{compound:<25} {V0:>10.3f} ± {V0_err:>6.3f}     {E0:>10.3f}     {B0_GPa:>10.2f}     {r_squared:>6.4f}  {n_points:>3}")
            else:
                # If fit failed, use minimum energy point as estimate
                idx_min = np.argmin(energies)
                V0_approx = volumes[idx_min]
                E0_approx = energies[idx_min]
                
                results.append({
                    'compound': compound,
                    'V0': V0_approx,
                    'V0_err': None,
                    'E0': E0_approx,
                    'E0_err': None,
                    'B0_GPa': None,
                    'B0_err': None,
                    'B0_prime': None,
                    'r_squared': None,
                    'n_points': len(volumes),
                    'volumes': volumes,
                    'energies': energies,
                    'note': 'Insufficient data for EOS fit, using minimum energy point'
                })
                
                print(f"{compound:<25} {V0_approx:>10.3f} (approx)  {E0_approx:>10.3f}     {'N/A':>10}     {'N/A':>6}  {len(volumes):>3}  [*]")
        
        print("="*100)
        print("\n[*] Compounds marked with [*] have insufficient data points (<4) for EOS fitting.")
        print("    The reported volume is the volume at the minimum energy point in the data.\n")
        
        return results


def main():
    """Main function to run the analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Predict optimal cell volumes from VASP bulk modulus calculations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict optimal volumes for all compounds in a directory
  python predict_optimal_volume.py /path/to/bulk_modulus_calculations/
        """
    )
    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory containing bulk modulus calculations'
    )
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = OptimalVolumePredictor(args.base_dir)
    
    # Parse directory structure
    predictor.parse_directory_structure()
    
    if not predictor.data:
        print("No data found! Check your directory structure.")
        return
    
    # Predict optimal volumes
    results = predictor.predict_optimal_volumes()
    
    print(f"\nSummary: Analyzed {len(results)} compound(s)")
    successful_fits = sum(1 for r in results if r['V0_err'] is not None)
    print(f"  - Successful EOS fits: {successful_fits}")
    print(f"  - Approximate estimates: {len(results) - successful_fits}")


if __name__ == '__main__':
    main()

