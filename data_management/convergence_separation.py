#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil


# Default NELM when not set in INCAR (VASP default)
DEFAULT_NELM = 60

# Substrings that indicate electronic/ionic convergence failure in OUTCAR.
# Do not use "aborting" alone: VASP writes "aborting loop because EDIFF is reached" on success.
OUTCAR_FAILURE_PATTERNS = (
    "reached maximum number",
    "NOT CONVERGED",
    "EDDDAV",
)


def read_nelm_from_incar(incar_path: Path) -> int | None:
    """Parse NELM value from INCAR if present; return int or None (then use DEFAULT_NELM)."""
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


# Algorithm tags that start electronic iteration lines in OSZICAR (VASP)
OSZICAR_ALGORITHM_TAGS = {"DAV", "CGA", "SDA", "RMM", "CG", "trl"}


def max_iter_in_oszicar(oszicar_path: Path) -> int | None:
    """
    Return the maximum electronic iteration number seen in OSZICAR
    (the 'N' column for DAV/CGA/SDA/RMM/.. lines). If none found, return None.
    """
    if not oszicar_path.is_file():
        return None

    max_iter = None
    with oszicar_path.open("r", errors="ignore") as f:
        for line in f:
            # Typical lines start with 'DAV:', 'CGA:', 'SDA:', 'RMM:' etc.
            if ":" not in line:
                continue
            head, *rest = line.split(":", 1)
            head = head.strip()
            if head not in OSZICAR_ALGORITHM_TAGS:
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


def oszicar_has_final_energy(oszicar_path: Path) -> bool:
    """
    Return True if OSZICAR contains a completed ionic step summary line
    (contains " F= " or " E0="). Such a line is written when the electronic
    step has finished for that ionic step; absence indicates the run did not
    complete (e.g. hit NELM or was killed).
    """
    if not oszicar_path.is_file():
        return False
    with oszicar_path.open("r", errors="ignore") as f:
        for line in f:
            if " F= " in line or " E0=" in line:
                return True
    return False


def outcar_has_failure(outcar_path: Path) -> bool:
    """Return True if OUTCAR contains known convergence/failure messages."""
    if not outcar_path.is_file():
        return False
    try:
        text = outcar_path.read_text(errors="ignore")
        return any(p in text for p in OUTCAR_FAILURE_PATTERNS)
    except OSError:
        return False


def is_unconverged(
    incar_path: Path, oszicar_path: Path, outcar_path: Path | None = None
) -> bool | None:
    """
    Determine whether the run converged.

    Returns:
    - True  -> unconverged (hit NELM, no F=/E0= summary, or OUTCAR failure)
    - False -> converged  (max iter < NELM, has F=/E0=, no OUTCAR failure)
    - None  -> cannot decide (missing OSZICAR/INCAR or no iteration data)
    """
    nelm = read_nelm_from_incar(incar_path)
    if nelm is None:
        nelm = DEFAULT_NELM
    max_iter = max_iter_in_oszicar(oszicar_path)
    if max_iter is None:
        return None

    # OUTCAR reports convergence failure
    if outcar_path and outcar_has_failure(outcar_path):
        return True

    # No final energy line: run did not complete an ionic step (e.g. hit NELM or killed)
    if not oszicar_has_final_energy(oszicar_path):
        return True

    # Hit or exceeded NELM => electronic convergence not reached
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
    outcar = sim_dir / "OUTCAR"

    if not incar.exists() or not oszicar.exists():
        return None

    flag = is_unconverged(incar, oszicar, outcar if outcar.exists() else None)
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

    for entry in sorted(root.iterdir()):
        if entry.name in {"converged", "unconverged"}:
            continue
        if not entry.is_dir():
            continue

        status = classify_simulation(entry)
        if status is None:
            # Cannot decide (no INCAR/OSZICAR or no iteration data) -> treat as unconverged
            move_simulation(entry, root, "unconverged")
            unconverged += 1
            continue
        if status == "converged":
            move_simulation(entry, root, "converged")
            converged += 1
        else:
            move_simulation(entry, root, "unconverged")
            unconverged += 1

    print(f"Converged runs moved:   {converged}")
    print(f"Unconverged runs moved: {unconverged}")


if __name__ == "__main__":
    main()

