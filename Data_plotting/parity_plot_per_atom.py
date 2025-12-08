import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import argparse
import sys
from sklearn.metrics import mean_squared_error

# --- Argument parser ---
parser = argparse.ArgumentParser(description="Plot MACE vs REF parity plots from an XYZ file.")
parser.add_argument("xyz_file", type=str, help="Path to the input .xyz file")
args = parser.parse_args()

# --- Read file ---
try:
    frames = read(args.xyz_file, index=":")
except Exception as e:
    print(f"Error reading file {args.xyz_file}: {e}")
    sys.exit(1)

# --- Data extraction ---
mace_energies = []
ref_energies = []
mace_forces = []
ref_forces = []
n_atoms_list = []
k_content_list = []
k_content_forces = []  # For coloring force points

for frame in frames:
    mace_energies.append(frame.info["MACE_energy"])
    ref_energies.append(frame.info["REF_energy"])
    n_atoms_list.append(len(frame))
    
    # Count K atoms in this frame
    k_count = sum(1 for atom in frame if atom.symbol == 'K')
    k_content_list.append(k_count)
    
    # For forces, we need to repeat the K content for each force component
    frame_forces = frame.arrays["MACE_forces"].flatten()
    k_content_forces.extend([k_count] * len(frame_forces))
    
    mace_forces.extend(frame_forces)
    ref_forces.extend(frame.arrays["REF_forces"].flatten())

mace_energies = np.array(mace_energies)
ref_energies = np.array(ref_energies)
mace_forces = np.array(mace_forces)
ref_forces = np.array(ref_forces)
n_atoms_list = np.array(n_atoms_list)
k_content_list = np.array(k_content_list)
k_content_forces = np.array(k_content_forces)

# --- Per-atom energies ---
ref_per_atom = ref_energies / n_atoms_list
mace_per_atom = mace_energies / n_atoms_list

# --- RMSE calculations ---
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_total_energy = calculate_rmse(ref_energies, mace_energies)
rmse_per_atom_energy = calculate_rmse(ref_per_atom, mace_per_atom)
rmse_forces = calculate_rmse(ref_forces, mace_forces)

# Print summary statistics
print(f"Dataset Summary:")
print(f"Number of frames: {len(frames)}")
print(f"Total atoms: {n_atoms_list.sum()}")
print(f"K content range: {k_content_list.min()} - {k_content_list.max()} atoms per frame")
print(f"RMSE - Total Energy: {rmse_total_energy:.4f} eV")
print(f"RMSE - Per-Atom Energy: {rmse_per_atom_energy:.4f} eV")
print(f"RMSE - Forces: {rmse_forces:.4f} eV/Å")

# --- Plotting ---
# Create figure with dedicated space for colorbar
fig = plt.figure(figsize=(24, 6))

# Create a colormap for K content
cmap = plt.cm.viridis
norm = plt.Normalize(vmin=k_content_list.min(), vmax=k_content_list.max())

# Define grid layout: 3 plots + 1 colorbar column
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.2], hspace=0.3, wspace=0.4)

# Create individual axes
ax1 = fig.add_subplot(gs[0, 0])  # Total Energy
ax2 = fig.add_subplot(gs[0, 1])  # Forces  
ax3 = fig.add_subplot(gs[0, 2])  # Per-atom Energy
ax_cbar = fig.add_subplot(gs[0, 3])  # Colorbar

# Total Energy Plot
scatter1 = ax1.scatter(ref_energies, mace_energies, c=k_content_list, cmap=cmap, norm=norm, alpha=0.7, s=50)
lims = [min(ref_energies.min(), mace_energies.min()), max(ref_energies.max(), mace_energies.max())]
ax1.plot(lims, lims, 'k--', linewidth=2, alpha=0.8)
ax1.set_xlabel("REF Energy (eV)", fontsize=12)
ax1.set_ylabel("MACE Energy (eV)", fontsize=12)
ax1.set_title(f"Total Energy Parity Plot\nRMSE = {rmse_total_energy:.4f} eV", fontsize=12)
ax1.axis("equal")
ax1.grid(True, alpha=0.3)

# Force Parity Plot
scatter2 = ax2.scatter(ref_forces, mace_forces, c=k_content_forces, cmap=cmap, norm=norm, alpha=0.7, s=50)
lims = [min(ref_forces.min(), mace_forces.min()), max(ref_forces.max(), mace_forces.max())]
ax2.plot(lims, lims, 'k--', linewidth=2, alpha=0.8)
ax2.set_xlabel("REF Force (eV/Å)", fontsize=12)
ax2.set_ylabel("MACE Force (eV/Å)", fontsize=12)
ax2.set_title(f"Force Parity Plot\nRMSE = {rmse_forces:.4f} eV/Å", fontsize=12)
ax2.axis("equal")
ax2.grid(True, alpha=0.3)

# Per-Atom Energy Plot
scatter3 = ax3.scatter(ref_per_atom, mace_per_atom, c=k_content_list, cmap=cmap, norm=norm, alpha=0.7, s=50)
lims = [min(ref_per_atom.min(), mace_per_atom.min()), max(ref_per_atom.max(), mace_per_atom.max())]
ax3.plot(lims, lims, 'k--', linewidth=2, alpha=0.8)
ax3.set_xlabel("REF Energy per Atom (eV)", fontsize=12)
ax3.set_ylabel("MACE Energy per Atom (eV)", fontsize=12)
ax3.set_title(f"Per-Atom Energy Parity Plot\nRMSE = {rmse_per_atom_energy:.4f} eV", fontsize=12)
ax3.axis("equal")
ax3.grid(True, alpha=0.3)

# Create colorbar in dedicated subplot
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=ax_cbar)
cbar.set_label('K Content (number of K atoms)', fontsize=12, rotation=270, labelpad=20)

plt.show()
