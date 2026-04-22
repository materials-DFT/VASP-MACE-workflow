#!/usr/bin/env python3
"""Print element counts and concentrations from XYZ or VASP POSCAR/CONTCAR files."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# For inferring species labels in VASP4-style POSCARs (counts only, no symbol line).
_KNOWN_ELEMENTS = frozenset(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co
    Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I
    Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au
    Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db
    Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"""
    .split()
)


def _strip_comment(line: str) -> str:
    return line.split("!")[0].strip()


def normalize_symbol(raw: str) -> str:
    """First column may be 'Fe', 'Fe12', 'Fe_0', etc.; keep element stem."""
    s = raw.strip()
    m = re.match(r"^([A-Za-z]+)", s)
    if not m:
        return s
    sym = m.group(1)
    return sym[0].upper() + sym[1:].lower() if len(sym) > 1 else sym.upper()


def parse_xyz(path: Path) -> tuple[int, Counter[str]]:
    """Return (number_of_frames, element_counts_over_all_frames)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    i = 0
    n_frames = 0
    total: Counter[str] = Counter()

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n_atoms = int(line)
        except ValueError:
            i += 1
            continue
        if n_atoms < 0:
            raise ValueError(f"Negative atom count at line {i + 1}: {n_atoms}")
        if i + 1 + n_atoms > len(lines):
            raise ValueError(
                f"Incomplete frame at line {i + 1}: need {n_atoms} atom rows, "
                f"file ends early."
            )
        for j in range(i + 2, i + 2 + n_atoms):
            parts = lines[j].split()
            if not parts:
                continue
            total[normalize_symbol(parts[0])] += 1
        n_frames += 1
        i += 2 + n_atoms

    return n_frames, total


def _nonempty_lines(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = _strip_comment(line).strip()
        if s:
            out.append(s)
    return out


def _line_is_all_positive_ints(line: str) -> bool:
    parts = line.split()
    if not parts:
        return False
    try:
        return all(int(p) > 0 for p in parts)
    except ValueError:
        return False


def _infer_symbols_vasp4(comment: str, n_species: int) -> list[str]:
    """Map VASP4 comment tokens to element symbols when possible."""
    tokens = comment.split()
    if len(tokens) == n_species and all(
        normalize_symbol(t) in _KNOWN_ELEMENTS for t in tokens
    ):
        return [normalize_symbol(t) for t in tokens]
    return [f"Species_{i + 1}" for i in range(n_species)]


def parse_poscar(path: Path) -> tuple[int, Counter[str]]:
    """Return (1, element_counts) for a single-frame POSCAR/CONTCAR.

    Uses only the header (comment, scaling, lattice, species/counts); coordinate
    blocks are not required, so trailing velocities in CONTCAR are ignored.
    """
    lines = _nonempty_lines(path)
    if len(lines) < 6:
        raise ValueError(f"POSCAR too short (need at least 6 non-empty lines): {path}")

    # 0: comment, 1: scale, 2–4: lattice, 5: either VASP5 symbols or VASP4 counts
    if _line_is_all_positive_ints(lines[5]):
        # VASP4: counts only; try to recover symbols from line 1.
        counts = [int(x) for x in lines[5].split()]
        symbols = _infer_symbols_vasp4(lines[0], len(counts))
    else:
        symbols = [normalize_symbol(t) for t in lines[5].split()]
        if len(lines) < 7:
            raise ValueError(f"POSCAR missing species counts line after symbols: {path}")
        if not _line_is_all_positive_ints(lines[6]):
            raise ValueError(
                f"POSCAR species counts line must be positive integers: {lines[6]!r}"
            )
        counts = [int(x) for x in lines[6].split()]
        if len(symbols) != len(counts):
            raise ValueError(
                f"POSCAR: {len(symbols)} symbols but {len(counts)} count fields"
            )

    total: Counter[str] = Counter()
    for sym, c in zip(symbols, counts):
        total[sym] += c
    return 1, total


def _detect_structure_format(path: Path) -> str:
    """Return 'poscar' or 'xyz' using name, extension, and first non-empty line."""
    upper_name = path.name.upper()
    if upper_name in {"POSCAR", "CONTCAR", "REFCAR"}:
        return "poscar"
    suf = path.suffix.lower()
    if suf in {".xyz", ".extxyz"}:
        return "xyz"

    first: str | None = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = _strip_comment(line).strip()
            if s:
                first = s
                break
    if first is None:
        return "xyz"
    try:
        n = int(first)
    except ValueError:
        return "poscar"
    if n < 0:
        return "poscar"
    return "xyz"


def parse_structure(path: Path) -> tuple[int, Counter[str]]:
    """Parse XYZ or POSCAR; prefer format from `_detect_structure_format`, then fallback."""
    fmt = _detect_structure_format(path)
    order = (parse_poscar, parse_xyz) if fmt == "poscar" else (parse_xyz, parse_poscar)
    errs: list[str] = []
    for parser in order:
        try:
            n, c = parser(path)
            if sum(c.values()) > 0:
                return n, c
        except (ValueError, OSError) as e:
            errs.append(f"{parser.__name__}: {e}")
    if errs:
        raise ValueError(" | ".join(errs))
    return 0, Counter()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute element concentrations from an XYZ / extended XYZ "
        "or a VASP POSCAR / CONTCAR."
    )
    p.add_argument(
        "structure_path",
        type=Path,
        help="Path to .xyz / .extxyz or POSCAR / CONTCAR",
    )
    args = p.parse_args()
    path: Path = args.structure_path.expanduser()
    if not path.is_file():
        print(f"Error: not a file: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        n_frames, counts = parse_structure(path)
    except (ValueError, OSError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    n_total = sum(counts.values())
    if n_total == 0:
        print(f"File: {path}")
        print("No atoms parsed (empty or unrecognized format).")
        sys.exit(0)

    print(f"File: {path}")
    print(f"Frames: {n_frames}")
    print(f"Total atoms: {n_total}")
    print()

    ordered = sorted(counts.items(), key=lambda x: (-x[1], x[0]))

    print(f"{'Element':<10} {'Count':>10} {'atomic %':>12}")

    for sym, c in ordered:
        at_pct = 100.0 * c / n_total
        print(f"{sym:<10} {c:10d} {at_pct:12.4f}")


if __name__ == "__main__":
    main()
