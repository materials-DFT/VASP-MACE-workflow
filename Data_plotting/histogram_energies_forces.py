import matplotlib.pyplot as plt
import numpy as np
import argparse
from ase.io import read

def parse_xyz(filename):
    atoms_list = read(filename, index=":")
    
    frame_avg_force_magnitudes = []
    energies = []
    num_atoms_list = []

    for atoms in atoms_list:
        forces = atoms.arrays["REF_forces"]
        energy = atoms.info["REF_energy"]
        avg_force = np.mean(np.linalg.norm(forces, axis=1))  # average per structure
        frame_avg_force_magnitudes.append(avg_force)
        energies.append(energy)
        num_atoms_list.append(len(atoms))

    avg_atoms = np.mean(num_atoms_list)
    return frame_avg_force_magnitudes, energies, avg_atoms

parser = argparse.ArgumentParser(description="Plot histograms of forces and energies from XYZ file.")
parser.add_argument("filename", help="Input XYZ file with energy and force information.")
args = parser.parse_args()

print(f"Parsing file: {args.filename}")
force_magnitudes, energies, avg_atoms = parse_xyz(args.filename)
print(f"Found {len(force_magnitudes)} force magnitudes and {len(energies)} energy values")
print(f"Average number of atoms per structure: {avg_atoms}")

energies_array = np.array(energies, dtype=np.float64) / avg_atoms
energies_array = energies_array[np.isfinite(energies_array)]

force_cutoff = 20
filtered_forces = [f for f in force_magnitudes if f <= force_cutoff]
print(f"\nFiltered {len(force_magnitudes) - len(filtered_forces)} force values > {force_cutoff}")

force_counts, force_bins = np.histogram(filtered_forces, bins=60, range=(0, force_cutoff))
print("\nForce Magnitude Histogram (≤ 20):")
for count, edge in zip(force_counts, force_bins[:-1]):
    print(f"Bin edge: {edge:.2f}, Count: {count}")

energy_counts, energy_bins = np.histogram(energies_array, bins=60)
print("\nEnergy per Atom Histogram (Unfiltered):")
for count, edge in zip(energy_counts, energy_bins[:-1]):
    print(f"Bin edge: {edge:.2f}, Count: {count}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.hist(filtered_forces, bins=60, range=(0, force_cutoff), edgecolor='black', alpha=0.7, color='skyblue')
ax1.set_xlabel("Force Magnitude (eV/Å)")
ax1.set_ylabel("Frequency")
ax1.set_title("Histogram of Atomic Forces (≤ 20)")
ax1.grid(True, alpha=0.3)

ax2.hist(energies_array, bins=60, edgecolor='black', alpha=0.7, color='lightcoral')
ax2.set_xlabel("Energy per Atom (eV)")
ax2.set_ylabel("Frequency")
ax2.set_title("Histogram of Energies (Normalized, Unfiltered)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
