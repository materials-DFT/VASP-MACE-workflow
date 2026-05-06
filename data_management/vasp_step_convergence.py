"""
Per-ion-step electronic convergence heuristics for VASP runs.

Used by extract_* scripts: OUTCAR chunk text (SCF failure substrings) plus
OSZICAR/NELM when step counts match OUTCAR frame counts.
"""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import ParseError
from ase.io.vasp_parsers.vasp_outcar_parsers import outcarchunks

try:
    from convergence_separation import DEFAULT_NELM, read_nelm_from_incar
except ImportError:
    DEFAULT_NELM = 60

    def read_nelm_from_incar(incar_path: Path) -> int | None:
        if not incar_path.is_file():
            return None
        nelm = None
        with incar_path.open("r", errors="ignore") as f:
            for line in f:
                line = line.split("!")[0].split("#")[0].strip()
                if not line or "NELM" not in line.upper():
                    continue
                parts = line.replace("=", " ").split()
                for i, token in enumerate(parts):
                    if token.upper() == "NELM" and i + 1 < len(parts):
                        try:
                            nelm = int(parts[i + 1])
                        except ValueError:
                            pass
                        break
        return nelm


# Substrings in one OUTCAR ionic/MD chunk that indicate SCF did not finish well.
OUTCAR_CHUNK_UNCONVERGED = (
    "reached maximum number of electronic",
    "BRMIX: very serious",
    "NOT CONVERGED",
)

_OSZICAR_ELEC_TAGS = {"DAV", "CGA", "SDA", "RMM", "CG", "trl"}


def outcar_per_step_electronic_ok(outcar: Path) -> list[bool] | None:
    """
    One bool per OUTCAR image (ASE chunk): False if the chunk text matches
    known SCF/ionic failure substrings, else True.
    """
    try:
        flags: list[bool] = []
        with outcar.open(errors="ignore") as fd:
            for chunk in outcarchunks(fd):
                t = "".join(chunk.lines)
                bad = any(p in t for p in OUTCAR_CHUNK_UNCONVERGED)
                flags.append(not bad)
    except (OSError, ParseError):
        return None
    if not flags:
        return None
    return flags


def per_step_nelm_satisfied(oszicar: Path, nelm: int) -> list[bool] | None:
    """
    For each ionic/MD block in OSZICAR, return True if max SCF iteration index
    in that block is strictly less than NELM. Blocks end on F= / E0= lines.
    """
    try:
        with oszicar.open("r", errors="ignore") as f:
            lines = f.readlines()
    except OSError:
        return None
    block_max = 0
    out: list[bool] = []
    for line in lines:
        if ":" in line:
            head, _, rest_ = line.partition(":")
            head = head.strip()
            if head in _OSZICAR_ELEC_TAGS:
                rest = rest_.strip().split()
                if rest:
                    try:
                        n = int(rest[0])
                    except ValueError:
                        pass
                    else:
                        block_max = max(block_max, n)
        if " F= " in line or " E0=" in line:
            out.append(block_max < nelm)
            block_max = 0
    return out or None


def per_step_electronically_converged(
    outcar: Path, incar: Path, oszicar: Path
) -> list[bool] | None:
    """
    Combine OSZICAR/NELM with OUTCAR chunk heuristics. Length matches the
    number of ionic images in outcar. None if OUTCAR chunk list cannot be built.
    """
    o_list = outcar_per_step_electronic_ok(outcar)
    if o_list is None:
        return None

    nelm = read_nelm_from_incar(incar)
    if nelm is None:
        nelm = DEFAULT_NELM

    z_list = per_step_nelm_satisfied(oszicar, nelm) if oszicar.is_file() else None
    n = len(o_list)

    if z_list is None or len(z_list) != n:
        if z_list is not None and len(z_list) != n:
            print(
                f"  [convergence] OSZICAR step count {len(z_list)} != "
                f"OUTCAR frames {n}; using OUTCAR per-step checks only"
            )
        return o_list
    return [a and b for a, b in zip(o_list, z_list)]


def filter_images_converged_only(
    outcar: Path, images: list[Atoms]
) -> tuple[list[Atoms] | None, str | None]:
    """
    Keep only frames whose ionic step passed convergence checks.
    Returns (filtered, None) on success, or (None, error_message) on failure.
    """
    sim = outcar.parent
    ok = per_step_electronically_converged(outcar, sim / "INCAR", sim / "OSZICAR")
    if ok is None:
        return None, "could not parse per-step convergence (OUTCAR chunk read failed)"
    if len(ok) != len(images):
        return (
            None,
            f"convergence list length {len(ok)} != frame count {len(images)}",
        )
    return [a for k, a in enumerate(images) if ok[k]], None


def lowest_converged_frame(
    outcar: Path, images: list[Atoms]
) -> tuple[Atoms | None, str | None]:
    """
    Of all ionic steps that pass checks, return the one with minimum energy.
    Returns (None, err) if parsing fails or no converged frame exists.
    """
    filtered, err = filter_images_converged_only(outcar, images)
    if err is not None:
        return None, err
    assert filtered is not None
    if not filtered:
        return None, "no electronically converged ionic steps in this OUTCAR"
    return min(filtered, key=lambda a: a.get_potential_energy()), None
