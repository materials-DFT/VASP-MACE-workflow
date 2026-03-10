import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot relative per-atom energy parity (MLIP vs REF) from an XYZ file.\n"
            "Energies are shifted so that a chosen reference structure has zero energy."
        )
    )
    parser.add_argument("xyz_file", type=str, help="Path to the input .xyz file")
    parser.add_argument(
        "--ref-index",
        type=int,
        default=None,
        help=(
            "Index (0-based) of the reference structure. "
            "If not provided, the structure with the lowest REF energy per atom is used."
        ),
    )
    args = parser.parse_args()

    # --- Read file ---
    try:
        frames = read(args.xyz_file, index=":")
    except Exception as e:
        print(f"Error reading file {args.xyz_file}: {e}")
        sys.exit(1)

    if len(frames) == 0:
        print(f"No frames found in file {args.xyz_file}")
        sys.exit(1)

    # --- Detect ML energy key (MACE or ALLEGRO) ---
    first_info = frames[0].info
    if "MACE_energy" in first_info:
        ml_key = "MACE_energy"
        ml_label = "MACE"
    elif "ALLEGRO_energy" in first_info:
        ml_key = "ALLEGRO_energy"
        ml_label = "ALLEGRO"
    else:
        print(
            "No 'MACE_energy' or 'ALLEGRO_energy' in frame.info. "
            "Make sure the XYZ file was written with one of these fields."
        )
        sys.exit(1)
    if "REF_energy" not in first_info:
        print("No 'REF_energy' in frame.info.")
        sys.exit(1)

    # --- Extract energies and atom counts ---
    ml_energies = []
    ref_energies = []
    n_atoms_list = []

    for i, frame in enumerate(frames):
        if ml_key not in frame.info or "REF_energy" not in frame.info:
            print(
                f"Frame {i} is missing '{ml_key}' or 'REF_energy' in frame.info. "
                "Make sure the XYZ file was written with these fields."
            )
            sys.exit(1)

        ml_energies.append(frame.info[ml_key])
        ref_energies.append(frame.info["REF_energy"])
        n_atoms_list.append(len(frame))

    ml_energies = np.array(ml_energies, dtype=float)
    ref_energies = np.array(ref_energies, dtype=float)
    n_atoms_list = np.array(n_atoms_list, dtype=float)

    # --- Per-atom energies ---
    ref_per_atom = ref_energies / n_atoms_list
    ml_per_atom = ml_energies / n_atoms_list

    # --- Choose reference structure ---
    if args.ref_index is not None:
        if args.ref_index < 0 or args.ref_index >= len(ref_per_atom):
            print(
                f"Invalid ref-index {args.ref_index}. "
                f"Must be between 0 and {len(ref_per_atom) - 1}."
            )
            sys.exit(1)
        ref_idx = args.ref_index
    else:
        ref_idx = int(np.argmin(ref_per_atom))

    ref_ref_energy = ref_per_atom[ref_idx]
    ref_ml_energy = ml_per_atom[ref_idx]

    # --- Relative energies (per atom) ---
    delta_ref = ref_per_atom - ref_ref_energy
    delta_ml = ml_per_atom - ref_ml_energy

    # --- Error metric (RMSE of relative energies) ---
    diff = delta_ml - delta_ref
    rmse_eV = float(np.sqrt(np.mean(diff**2)))

    print(f"Using frame {ref_idx} as reference.")
    print(f"  REF per-atom energy of reference:  {ref_ref_energy:.6f} eV")
    print(f"  {ml_label} per-atom energy of reference: {ref_ml_energy:.6f} eV")
    print(f"  RMSE of ΔE per atom: {rmse_eV:.6f} eV")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(delta_ref, delta_ml, alpha=0.6)

    # Parity line y = x
    all_vals = np.concatenate([delta_ref, delta_ml])
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    lims = [vmin, vmax]
    ax.plot(lims, lims, "k--", label="y = x")

    ax.set_xlabel(r"$\Delta E_{\mathrm{REF}}$ per atom (eV)")
    ax.set_ylabel(r"$\Delta E_{\mathrm{" + ml_label + r"}}$ per atom (eV)")
    ax.set_title(
        f"Relative Per-Atom Energy Parity Plot\n"
        f"reference frame = {ref_idx}, RMSE = {rmse_eV:.6f} eV/atom"
    )
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

