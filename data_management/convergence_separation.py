#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil


def read_nelm_from_incar(incar_path: Path) -> int | None:
    """Parse NELM value from INCAR if present; return int or None."""
    if not incar_path.is_file():
        return None

    nelm = None
    with incar_path.open("r") as f:
        for line in f:
            # Strip comments starting with ! or #
            line = line.split("!")[0].split("#")[0].strip()
            if not line:
                continue
            if "NELM" in line.upper():
                parts = line.replace("=", " ").split()
                for i, token in enumerate(parts):
                    if token.upper() == "NELM" and i + 1 < len(parts):
                        try:
                            nelm = int(parts[i + 1])
                        except ValueError:
                            pass
                        break
    return nelm


def max_iter_in_oszicar(oszicar_path: Path) -> int | None:
    """
    Return the maximum electronic iteration number seen in OSZICAR
    (the 'N' column for DAV/CGA/SDA/.. lines). If none found, return None.
    """
    if not oszicar_path.is_file():
        return None

    max_iter = None
    with oszicar_path.open("r", errors="ignore") as f:
        for line in f:
            # Typical lines start with 'DAV:', 'CGA:', 'SDA:', 'RMM:' etc.
            if ":" not in line:
                continue
            head, *rest = line.split(":")
            head = head.strip()
            if head not in {"DAV", "CGA", "SDA", "RMM"}:
                continue
            try:
                # After the colon, first integer is iteration index
                tokens = rest[0].split()
                if not tokens:
                    continue
                n = int(tokens[0])
            except (ValueError, IndexError):
                continue
            if max_iter is None or n > max_iter:
                max_iter = n
    return max_iter


def is_unconverged(incar_path: Path, oszicar_path: Path) -> bool | None:
    """
    Determine whether NELM was hit.

    Returns:
    - True  -> unconverged (max iter == NELM)
    - False -> converged  (max iter < NELM)
    - None  -> cannot decide (missing data)
    """
    nelm = read_nelm_from_incar(incar_path)
    max_iter = max_iter_in_oszicar(oszicar_path)

    if nelm is None or max_iter is None:
        return None

    # If the maximum electronic iteration equals NELM, we assume NELM was hit
    if max_iter >= nelm:
        return True
    return False


def classify_simulation(sim_dir: Path) -> str | None:
    """
    Classify a simulation directory as 'converged', 'unconverged', or None
    if it doesn't look like a VASP run (no INCAR/OSZICAR or cannot decide).
    """
    incar = sim_dir / "INCAR"
    oszicar = sim_dir / "OSZICAR"

    if not incar.exists() or not oszicar.exists():
        return None

    flag = is_unconverged(incar, oszicar)
    if flag is None:
        return None
    return "unconverged" if flag else "converged"


def move_simulation(sim_dir: Path, dest_root: Path, label: str) -> None:
    target_dir = dest_root / label
    target_dir.mkdir(exist_ok=True)
    dest = target_dir / sim_dir.name
    if dest.exists():
        i = 1
        while True:
            alt = target_dir / f"{sim_dir.name}_{i}"
            if not alt.exists():
                dest = alt
                break
            i += 1
    shutil.move(str(sim_dir), str(dest))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check DFT convergence in VASP runs in a directory by comparing "
            "NELM from INCAR with the maximum electronic iteration in OSZICAR. "
            "Simulation subdirectories are moved into 'converged' and "
            "'unconverged' folders."
        )
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Path to directory containing VASP simulation subdirectories",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Provided path is not a directory: {root}")

    converged = 0
    unconverged = 0
    skipped = 0

    for entry in sorted(root.iterdir()):
        if entry.name in {"converged", "unconverged"}:
            continue
        if not entry.is_dir():
            continue

        status = classify_simulation(entry)
        if status is None:
            skipped += 1
            continue
        if status == "converged":
            move_simulation(entry, root, "converged")
            converged += 1
        else:
            move_simulation(entry, root, "unconverged")
            unconverged += 1

    print(f"Converged runs moved:   {converged}")
    print(f"Unconverged runs moved: {unconverged}")
    print(f"Skipped (undetermined/no INCAR/OSZICAR): {skipped}")


if __name__ == "__main__":
    main()

