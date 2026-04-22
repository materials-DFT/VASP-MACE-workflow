#!/usr/bin/env python3
"""Extract Cartesian geometry from a Gaussian log or .com/.gjf input and write VASP 5 POSCAR."""

from __future__ import annotations

import re
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Periodic table (1-based atomic numbers)
# -----------------------------------------------------------------------------
PERIODIC = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)


def z_to_symbol(z: int) -> str:
    if z < 1 or z > len(PERIODIC):
        raise ValueError(f"Unsupported atomic number: {z}")
    return PERIODIC[z - 1]


ROW_RE = re.compile(
    r"^\s*\d+\s+(\d+)\s+\d+\s+"
    r"([-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][-+]?\d+)?)\s*$"
)

DASH_ROW_RE = re.compile(r"^\s*-+\s*$")

# Archive: \Element,x,y,z repeated (Gaussian uses backslash as separator)
ARCHIVE_ATOM_RE = re.compile(
    r"\\([A-Z][a-z]?),([-+0-9.eE]+),([-+0-9.eE]+),([-+0-9.eE]+)"
)


def _collapse_archive_whitespace(blob: str) -> str:
    """Join wrapped Gaussian archive lines so \\El,x,y,z tokens are contiguous."""
    return re.sub(r"\s+", "", blob)


def _truncate_before_normal_termination(text: str) -> str:
    marker = "Normal termination of Gaussian"
    idx = text.rfind(marker)
    if idx == -1:
        return text
    return text[:idx]


def parse_orientation_tables(text: str, kind: str) -> list[list[tuple[str, float, float, float]]]:
    """Return all frames for the given orientation kind ('standard' or 'input')."""
    header = "Standard orientation:" if kind == "standard" else "Input orientation:"
    lines = text.splitlines()
    frames: list[list[tuple[str, float, float, float]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() != header.strip():
            i += 1
            continue
        # Found orientation block; locate coordinate table
        j = i + 1
        while j < len(lines) and "Coordinates (Angstroms)" not in lines[j]:
            j += 1
        if j >= len(lines):
            break
        # Skip header lines until dashed separator after "Z" header
        j += 1
        while j < len(lines) and not DASH_ROW_RE.match(lines[j]):
            j += 1
        if j >= len(lines):
            break
        j += 1  # past first ------
        atoms: list[tuple[str, float, float, float]] = []
        while j < len(lines):
            row = lines[j]
            if DASH_ROW_RE.match(row):
                break
            m = ROW_RE.match(row)
            if m:
                z = int(m.group(1))
                x, y, zc = float(m.group(2)), float(m.group(3)), float(m.group(4))
                atoms.append((z_to_symbol(z), x, y, zc))
            j += 1
        if atoms:
            frames.append(atoms)
        i = j
    return frames


CHARGE_MULT_RE = re.compile(r"^\s*([+-]?\d+)\s+(\d+)\s*$")


def _symbol_from_token(tok: str) -> str | None:
    """Map first field to a periodic symbol (supports isotope prefixes like 16O)."""
    s = tok.lstrip("0123456789")
    if not s:
        return None
    key = s.lower()
    for sym in PERIODIC:
        if sym.lower() == key:
            return sym
    return None


def parse_com_atom_line(line: str) -> tuple[str, float, float, float] | None:
    """One Cartesian line from a Gaussian input: Element x y z (Å) or Z x y z."""
    raw = line.split("!")[0].strip()
    if not raw or raw.startswith("--"):
        return None
    parts = raw.split()
    if len(parts) < 4:
        return None
    x_str, y_str, z_str = parts[-3], parts[-2], parts[-1]
    try:
        x, y, z = float(x_str), float(y_str), float(z_str)
    except ValueError:
        return None
    elem_tok = parts[0]
    if elem_tok.isdigit():
        znum = int(elem_tok)
        if 1 <= znum <= len(PERIODIC):
            return (z_to_symbol(znum), x, y, z)
        return None
    sym = _symbol_from_token(elem_tok)
    if sym is None:
        return None
    low = sym.lower()
    if low in ("bq", "x"):  # ghost / placeholder centers
        return None
    if low == "tv":  # periodic translation vectors
        return None
    try:
        float(parts[1])
    except (ValueError, IndexError):
        return None
    return (sym, x, y, z)


def parse_com_geometry(text: str) -> list[tuple[str, float, float, float]] | None:
    """Parse all Cartesian blocks after charge/multiplicity lines; return the last non-empty block."""
    lines = text.splitlines()
    blocks: list[list[tuple[str, float, float, float]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not CHARGE_MULT_RE.match(line.split("!")[0]):
            i += 1
            continue
        atoms: list[tuple[str, float, float, float]] = []
        j = i + 1
        while j < len(lines):
            stripped = lines[j].split("!")[0].strip()
            if not stripped:
                j += 1
                break
            if stripped.startswith("--"):
                break
            atom = parse_com_atom_line(lines[j])
            if atom is None:
                break
            atoms.append(atom)
            j += 1
        if atoms:
            blocks.append(atoms)
            i = j
        else:
            i += 1
    if not blocks:
        return None
    return blocks[-1]


def parse_archive_geometry(text: str) -> list[tuple[str, float, float, float]] | None:
    """Parse Cartesian coordinates from the last Gaussian archive block (line-wrapped safe)."""
    truncated = _truncate_before_normal_termination(text)
    marker = "Unable to Open any file for archive entry."
    uo = truncated.rfind(marker)
    if uo == -1:
        blob = truncated
    else:
        blob = truncated[uo + len(marker) :]
    ver = blob.rfind("\\Version=")
    if ver == -1:
        return None
    blob = blob[:ver]
    one = _collapse_archive_whitespace(blob)
    # Cartesians start after charge/mult 0,1\ — avoid matching \\F (route) earlier in the chunk
    m0 = re.search(r"0,1\\", one)
    if not m0:
        return None
    geom = one[m0.end() :]
    # First center often appears as C,... ; later centers use \Cl,... \H,... etc.
    if geom and geom[0].isalpha():
        geom = "\\" + geom
    tuples = ARCHIVE_ATOM_RE.findall(geom)
    if not tuples:
        return None
    out: list[tuple[str, float, float, float]] = []
    for sym, xs, ys, zs in tuples:
        out.append((sym, float(xs), float(ys), float(zs)))
    return out


def tight_orthorhombic_cell(
    atoms: list[tuple[str, float, float, float]],
    vacuum_per_side_a: float = 0.0,
) -> tuple[tuple[float, float, float], list[tuple[str, float, float, float]]]:
    """Axis-aligned box: molecular span plus ``vacuum_per_side_a`` empty space on each face (Å)."""
    if vacuum_per_side_a < 0:
        raise ValueError("vacuum per side must be >= 0")
    xs = [a[1] for a in atoms]
    ys = [a[2] for a in atoms]
    zs = [a[3] for a in atoms]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    zmin, zmax = min(zs), max(zs)
    eps = 1e-10
    span_x = max(xmax - xmin, eps)
    span_y = max(ymax - ymin, eps)
    span_z = max(zmax - zmin, eps)
    pad = vacuum_per_side_a
    bx = span_x + 2 * pad
    by = span_y + 2 * pad
    bz = span_z + 2 * pad
    shifted = [
        (s, x - xmin + pad, y - ymin + pad, z - zmin + pad) for s, x, y, z in atoms
    ]
    return (bx, by, bz), shifted


def reorder_species_alphabetical(
    atoms: list[tuple[str, float, float, float]],
) -> tuple[list[str], list[int], list[tuple[float, float, float]]]:
    species = sorted({a[0] for a in atoms})
    counts = [sum(1 for a in atoms if a[0] == s) for s in species]
    coords: list[tuple[float, float, float]] = []
    for s in species:
        for sym, x, y, z in atoms:
            if sym == s:
                coords.append((x, y, z))
    return species, counts, coords


def write_poscar(
    path: Path,
    comment: str,
    lattice: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    species: list[str],
    counts: list[int],
    coords: list[tuple[float, float, float]],
) -> None:
    ax, ay, az = lattice[0]
    bx, by, bz = lattice[1]
    cx, cy, cz = lattice[2]
    lines = [
        comment[:80],
        "1.0",
        f"{ax:20.16f} {ay:20.16f} {az:20.16f}",
        f"{bx:20.16f} {by:20.16f} {bz:20.16f}",
        f"{cx:20.16f} {cy:20.16f} {cz:20.16f}",
        " ".join(species),
        " ".join(str(c) for c in counts),
        "Cartesian",
    ]
    for x, y, z in coords:
        lines.append(f"{x:20.16f} {y:20.16f} {z:20.16f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(f"Usage: {sys.argv[0]} Gaussian.log|.com|.gjf [vacuum_angstrom]", file=sys.stderr)
        print(
            "  vacuum_angstrom  optional; empty space on each side of the bounding box (Å).",
            file=sys.stderr,
        )
        print("  Omit it for a tight cell (no extra vacuum). Writes ./POSCAR", file=sys.stderr)
        return 2

    log_path = Path(sys.argv[1])
    vacuum = 0.0
    if len(sys.argv) == 3:
        try:
            vacuum = float(sys.argv[2])
        except ValueError:
            print(f"Invalid vacuum value: {sys.argv[2]!r} (need a number, Å)", file=sys.stderr)
            return 2
        if vacuum < 0:
            print("Vacuum must be >= 0.", file=sys.stderr)
            return 2
    text = log_path.read_text(encoding="utf-8", errors="replace")
    suffix = log_path.suffix.lower()
    atoms: list[tuple[str, float, float, float]] | None

    if suffix in (".com", ".gjf"):
        atoms = parse_com_geometry(text)
        if not atoms:
            print(
                "No Cartesian coordinates found in input file "
                "(expected charge/mult line then Element x y z lines).",
                file=sys.stderr,
            )
            return 1
    else:
        truncated = _truncate_before_normal_termination(text)
        frames = parse_orientation_tables(truncated, "standard")
        if frames:
            atoms = frames[-1]
        else:
            atoms = parse_archive_geometry(text)

        if not atoms:
            print("No coordinates found (orientation tables and archive parse failed).", file=sys.stderr)
            return 1

    (bx, by, bz), shifted = tight_orthorhombic_cell(atoms, vacuum_per_side_a=vacuum)

    species, counts, coords = reorder_species_alphabetical(shifted)
    lattice = ((bx, 0.0, 0.0), (0.0, by, 0.0), (0.0, 0.0, bz))
    vac_note = f"vacuum={vacuum:g}A/side" if vacuum else "tight box"
    species_hint = " ".join(f"{s}*{c}" for s, c in zip(species, counts))
    comment = f"{log_path.name} | Gaussian -> POSCAR | {vac_note} | {species_hint}"
    write_poscar(Path("POSCAR"), comment, lattice, species, counts, coords)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
