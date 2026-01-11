

#!/usr/bin/env python3
"""
Bulk Modulus Analysis Script for VASP Calculations (No Outlier Removal)

Enhancements:
- Representative EOS point selection (min, endpoints, curve points)
- Copy selected structures
- Annotate selected points on E–V plots
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from collections import defaultdict
import shutil


class BulkModulusAnalyzer:
    """Analyze bulk modulus calculations from VASP OUTCAR files."""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data = defaultdict(dict)
        self.selected_eos_points = defaultdict(list)  # NEW

    # -------------------- Parsing --------------------

    def extract_volume_energy(self, outcar_path):
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()

            vol_match = re.findall(r'volume of cell\s*:\s*([\d.]+)', content)
            energy_match = re.findall(
                r'free\s+energy\s+TOTEN\s+=\s+([\d.-]+)\s+eV', content
            )

            if not vol_match or not energy_match:
                return None, None

            return float(vol_match[-1]), float(energy_match[-1])

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
            parts = outcar_path.parts
            
            # Find volume directory (V_XX.X%)
            vol_dir = None
            compound = None
            
            for i, part in enumerate(parts):
                if "V_" in part:
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

    # -------------------- EOS --------------------

    def birch_murnaghan_eos(self, V, E0, V0, B0, B0p):
        eta = (V0 / V) ** (1 / 3)
        return E0 + (9 * V0 * B0 / 16) * (
            (eta**2 - 1) ** 3 * B0p +
            (eta**2 - 1) ** 2 * (6 - 4 * eta**2)
        )

    def fit_eos(self, volumes, energies):
        if len(volumes) < 4:
            return None

        try:
            p0 = [energies.min(), volumes[np.argmin(energies)], 1.0, 4.0]
            popt, _ = curve_fit(
                self.birch_murnaghan_eos,
                volumes,
                energies,
                p0=p0,
                maxfev=20000
            )
            return {"E0": popt[0], "popt": popt}
        except Exception:
            return None

    # -------------------- Representative Selection --------------------

    def collect_representative_eos_structures(self, dest_dir,min_strain=30.0,max_strain=50.0, max_strain_2 =10, min_strain_2=5,curve_start=37,curve_end =75,n_side_points=5,n_pre_left_points=10):
        """
        Select representative EOS points:
        - Endpoints
        - Minimum-energy point
        - n_side_points on either side of minimum (within min_strain to max_strain)
        - n_pre_left_points leading up to the left side of the minimum
        """
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for compound, data in self.data.items():
            vol_pcts = np.array(sorted(data.keys()))
            volumes = np.array([data[v]["volume"] for v in vol_pcts])
            energies = np.array([data[v]["energy"] for v in vol_pcts])

            fit = self.fit_eos(volumes, energies)
            E0 = fit["E0"] if fit else energies.min()
            energies_norm = energies - E0

            # Sort by volume
            order = np.argsort(vol_pcts)
            vol_pcts = vol_pcts[order]
            energies_norm = energies_norm[order]

            # Minimum-energy point
            idx_min = np.argmin(energies_norm)
            min_vol_pct = vol_pcts[idx_min]

            # Endpoints
            selected = set([0, len(vol_pcts) - 1, idx_min])

            # Left side: to 40% below minimum
            left_mask = (vol_pcts >= min_vol_pct - max_strain) & (vol_pcts <= min_vol_pct - min_strain)
            left_indices = np.where(left_mask)[0]
            if len(left_indices) > 0:
                left_selected = np.linspace(left_indices[0], left_indices[-1],
                                            min(n_side_points, len(left_indices)), dtype=int)
                selected.update(left_selected)
            
             # Left side: to 10% below minimum
            left_mask2 = (vol_pcts >= min_vol_pct -  max_strain_2) & (vol_pcts <= min_vol_pct - min_strain_2)
            left_indices2 = np.where(left_mask2)[0]
            if len(left_indices2) > 0:
                left_selected2 = np.linspace(left_indices2[0], left_indices2[-1],
                                            min(2, len(left_indices2)), dtype=int)
                selected.update(left_selected2)

           # ---- PRE-LEFT points: spread across a specific range (curve_start → curve_end) ----
            pre_left_mask = (vol_pcts <= curve_start) & (vol_pcts >= curve_end)
            pre_left_indices = np.where(pre_left_mask)[0]
            if len(pre_left_indices) > 0:
                # Evenly pick up to n_pre_left_points
                n_pick = min(n_pre_left_points, len(pre_left_indices))
                pre_left_selected = np.linspace(pre_left_indices[0], pre_left_indices[-1], n_pick, dtype=int)
                selected.update(pre_left_selected)

            # Right side: 30–50% above minimum
            right_mask = (vol_pcts >= min_vol_pct + min_strain) & (vol_pcts <= min_vol_pct + max_strain)
            right_indices = np.where(right_mask)[0]
            if len(right_indices) > 0:
                right_selected = np.linspace(right_indices[0], right_indices[-1],
                                            min(n_side_points, len(right_indices)), dtype=int)
                selected.update(right_selected)
             # Right side: 5–50% above minimum
            right_mask2= (vol_pcts >= min_vol_pct + min_strain_2) & (vol_pcts <= min_vol_pct + max_strain_2)
            right_indices2 = np.where(right_mask2)[0]
            if len(right_indices2) > 0:
                right_selected2 = np.linspace(right_indices2[0], right_indices2[-1],
                                            min(2, len(right_indices2)), dtype=int)
                selected.update(right_selected2)

            # Store (vol_pct, normalized_energy)
            self.selected_eos_points[compound] = [
                (vol_pcts[i], energies_norm[i]) for i in sorted(selected)
            ]

            # Print selected points for verification
            print(f"\n{compound}: Selected EOS points:")
            for v, e in self.selected_eos_points[compound]:
                min_marker = " <-- MIN" if v == min_vol_pct else ""
                print(f"  V={v:+.1f}%  E−E0={e:.4f} eV{min_marker}")

            # Copy structures
            for v, _ in self.selected_eos_points[compound]:
                src = Path(data[v]["path"]).parent
                rel = src.relative_to(self.base_dir)
                dst = dest_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copytree(src, dst)

        print(f"\nSaved EOS representative structures to: {dest_dir}")

   
        # -------------------- Plotting --------------------

    def plot_all_compounds(self):
        """Plot all compounds on a single Energy vs Volume plot with EOS fits.
        Energies are normalized (shifted) so each compound's minimum energy is at 0 eV.
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
        
        # Process all data: normalize and prepare for plotting
        for i, (compound, data) in enumerate(self.data.items()):
            vol_pcts = sorted(data.keys())
            volumes = np.array([data[v]['volume'] for v in vol_pcts])
            energies = np.array([data[v]['energy'] for v in vol_pcts])
            
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
            
            print(f"  '{compound}': {len(volumes)} points, energy range: [{np.min(energies_norm):.3f}, {np.max(energies_norm):.3f}] eV")
            
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
        
        ax.set_xlabel('Volume (Å³)', fontsize=14)
        ax.set_ylabel('Energy - E₀ (eV)', fontsize=14)
        ax.set_title('Bulk Modulus: Normalized Energy vs Volume\n(All data included, no outliers removed)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=8, ncol=3, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add text annotation about y-axis limits
        annotation_text = f'Y-axis range: ±{y_limit:.1f} eV\n(All {n_compounds} compound(s) plotted)'
        ax.text(0.02, 0.98, annotation_text, 
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_table(self):
        """Generate a summary table of all fitted parameters."""
        results = []
        
        for compound, data in self.data.items():
            vol_pcts = sorted(data.keys())
            volumes = np.array([data[v]['volume'] for v in vol_pcts])
            energies = np.array([data[v]['energy'] for v in vol_pcts])
            fit_result = self.fit_eos(volumes, energies)
            
            if fit_result:
                results.append({
                    'Compound': compound,
                    'B₀ (GPa)': f"{fit_result['B0_GPa']:.2f} ± {fit_result['B0_err']:.2f}",
                    'V₀ (Å³)': f"{fit_result['V0']:.2f} ± {fit_result['V0_err']:.2f}",
                    "B₀'": f"{fit_result['B0_prime']:.2f} ± {fit_result['B0_prime_err']:.2f}",
                    'E₀ (eV)': f"{fit_result['E0']:.2f} ± {fit_result['E0_err']:.2f}",
                    'R²': f"{fit_result['r_squared']:.4f}",
                    'N_points': len(volumes)
                })
        
        # Print table
        print("\n" + "="*100)
        print("BULK MODULUS ANALYSIS SUMMARY")
        print("="*100)
        # print(f"{'Compound':<20} {'B₀ (GPa)':<20} {'V₀ (Å³)':<15} {'B₀\'':<15} {'E₀ (eV)':<15} {'R²':<10} {'N':<5}")
        print("-"*100)
        
        # for r in results:
        #     print(f"{r['Compound']:<20} {r['B₀ (GPa)']:<20} {r['V₀ (Å³)']:<15} "
                #   f"{r['B₀\'']:<15} {r['E₀ (eV)']:<15} {r['R²']:<10} {r['N_points']:<5}")
        
        print("="*100 + "\n")
        
        return results

    def plot_per_phase_annotated(self, selected_points_per_phase, output_dir):
        """
        Plot energy vs volume for each compound separately,
        annotate selected points (min, endpoints, curve points),
        and save plots to output_dir.

        Parameters
        ----------
        selected_points_per_phase : dict
            {compound: [vol_pct, ...]} or {compound: [(vol_pct, E_norm), ...]}
            If only vol_pct is passed, E_norm is computed from data.
        output_dir : str or Path
            Directory to save plots
        """
  
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for compound, data in self.data.items():
            vol_pcts = sorted(data.keys())
            volumes = np.array([data[v]['volume'] for v in vol_pcts])
            energies = np.array([data[v]['energy'] for v in vol_pcts])

            # Fit EOS to normalize energies
            fit_result = self.fit_eos(volumes, energies)
            E0 = fit_result['E0'] if fit_result else np.min(energies)
            energies_norm = energies - E0

            # EOS curve for plotting
            if fit_result:
                V_fit = np.linspace(volumes.min()*0.95, volumes.max()*1.05, 200)
                E_fit = self.birch_murnaghan_eos(V_fit, *fit_result['popt']) - E0
            else:
                V_fit = E_fit = None

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(volumes, energies_norm, s=80, color='blue', alpha=0.7, label='All data')
            if V_fit is not None and E_fit is not None:
                ax.plot(V_fit, E_fit, '-', color='red', label='EOS Fit')
            
            selected_points = self.selected_eos_points.get(compound, [])
            if not selected_points:
                continue

            emin = energies_norm.min()
            y_span = energies_norm.max() - energies_norm.min()
            dy = 0.035 * y_span if y_span > 0 else 0.02  # modest offset
            for vol_pct, E_norm in selected_points:
                vol = data[vol_pct]["volume"]

                # highlight selected point
                ax.scatter(
                    vol, E_norm,
                    s=120,
                    facecolors="none",
                    edgecolors="green",
                    linewidths=2,
                    zorder=4
                )
                   # Determine if this is the minimum
                if E_norm == emin:
                    label = f"MIN\n{vol_pct:+.1f}%"
                    text_color= "red"
                    weight = "bold"
                else:
                    label = f"{vol_pct:+.1f}%"
                    text_color = "green"
                    weight = "normal"

                # label (small + offset)
                ax.annotate(
                    label, 
                    xy=(vol, E_norm),
                    xytext=(11.2, 4.0),                 # ← small vertical offset
                    textcoords="offset points",     # ← screen-space offset
                    ha="center",
                    va="bottom",
                    rotation=50, 
                    fontsize=5,
                    color=text_color,                         
                    zorder=5
                )


            ax.set_xlabel('Volume (Å³)')
            ax.set_ylabel('Energy - E₀ (eV)')
            ax.set_title(f'Phase: {compound}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='k', linestyle='--', alpha=0.5)

            # Save figure
            plot_file = output_dir / f'phase_{compound}.png'
            fig.savefig(plot_file, dpi=200)
            plt.close(fig)
            print(f"Saved plot for {compound} -> {plot_file}")




# -------------------- Main --------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    args = parser.parse_args()

    analyzer = BulkModulusAnalyzer(args.base_dir)
    analyzer.parse_directory_structure()

    eos_dir = Path(args.base_dir) / "EOS_REPRESENTATIVE_SET"
    analyzer.collect_representative_eos_structures(
        eos_dir,min_strain=20.0,max_strain=40.0,min_strain_2=5.0,max_strain_2=10.0,curve_start=-37,curve_end =-75, n_side_points=3,n_pre_left_points=10
    )
        # Plot all compounds together
    analyzer.plot_all_compounds()

    per_phase_dir = Path(args.base_dir) / "PER_PHASE_PLOTS"
    analyzer.plot_per_phase_annotated(
        selected_points_per_phase=analyzer.selected_eos_points,
        output_dir=per_phase_dir
    )




if __name__ == "__main__":
    main()
