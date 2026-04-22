#!/usr/bin/env python3
"""
Convert a LAMMPS trajectory dump (lammpstrj) to extxyz.

Example:
  python convert_trajectory_to_extxyz.py trajectory.lammpstrj
"""

from __future__ import annotations

import argparse
import os
import shlex
from typing import Dict, List, Set

from ase.io import iread, write
from ase.data import atomic_masses, chemical_symbols


def resolve_input_path(path: str) -> str:
    """
    Resolve input as either:
      - a direct trajectory file path, or
      - a directory containing one or more *.lammpstrj files.
    """
    apath = os.path.abspath(path)
    if os.path.isfile(apath):
        return apath

    if not os.path.isdir(apath):
        raise FileNotFoundError(f"Input path not found: {path}")

    candidates = sorted(
        os.path.join(apath, name)
        for name in os.listdir(apath)
        if name.lower().endswith(".lammpstrj")
        and os.path.isfile(os.path.join(apath, name))
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .lammpstrj file found in directory: {apath}"
        )

    if len(candidates) > 1:
        print(
            "Multiple .lammpstrj files found; using first alphabetically: "
            f"{os.path.basename(candidates[0])}"
        )
    return candidates[0]


def parse_type_map(raw: str) -> Dict[int, str]:
    if not raw or not raw.strip():
        return {}
    mapping: Dict[int, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                f"Invalid mapping entry '{item}'. Expected format like '1:K'."
            )
        key_str, symbol = item.split(":", 1)
        key = int(key_str.strip())
        symbol = symbol.strip()
        if not symbol:
            raise ValueError(f"Empty chemical symbol in mapping entry '{item}'.")
        mapping[key] = symbol
    return mapping


def map_symbols(atoms, type_map: Dict[int, str]) -> None:
    if "type" not in atoms.arrays:
        raise KeyError(
            "No 'type' array found in trajectory frame. "
            "Ensure input is a LAMMPS dump with a 'type' column."
        )
    atom_types = atoms.arrays["type"]
    symbols = []
    for t in atom_types:
        it = int(t)
        if it not in type_map:
            raise KeyError(
                f"Atom type {it} not found in --type-map. "
                f"Provided keys: {sorted(type_map.keys())}"
            )
        symbols.append(type_map[it])
    atoms.set_chemical_symbols(symbols)


def _to_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def _is_valid_symbol(symbol: str) -> bool:
    return symbol in chemical_symbols


def _infer_symbol_from_mass(mass: float) -> str | None:
    # Match against ASE atomic masses; skip index 0 (dummy "X")
    best_idx = None
    best_diff = None
    for z in range(1, len(chemical_symbols)):
        ref_mass = float(atomic_masses[z])
        diff = abs(ref_mass - mass)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            best_idx = z
    if best_idx is None or best_diff is None:
        return None
    # 0.6 amu is permissive enough for rounded LAMMPS mass tables.
    if best_diff > 0.6:
        return None
    return chemical_symbols[best_idx]


def _parse_data_file_masses(path: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    in_masses = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not in_masses:
                if stripped == "Masses":
                    in_masses = True
                continue
            if not stripped:
                continue
            if stripped.startswith(("Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers")):
                break
            parts = stripped.split()
            if len(parts) < 2:
                continue
            if not parts[0].isdigit():
                continue
            type_id = int(parts[0])
            try:
                mass = float(parts[1])
            except ValueError:
                continue

            symbol = None
            if "#" in line:
                comment = line.split("#", 1)[1].strip()
                comment_tok = comment.split()[0] if comment else ""
                if _is_valid_symbol(comment_tok):
                    symbol = comment_tok
            if symbol is None:
                symbol = _infer_symbol_from_mass(mass)
            if symbol:
                mapping[type_id] = symbol
    return mapping


def _parse_input_script_elements(path: str, seen: Set[str] | None = None) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if seen is None:
        seen = set()
    apath = os.path.abspath(path)
    if apath in seen or not os.path.isfile(apath):
        return mapping
    seen.add(apath)

    basedir = os.path.dirname(apath)
    with open(apath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # LAMMPS comments: everything after '#'
            if "#" in stripped:
                stripped = stripped.split("#", 1)[0].strip()
            if not stripped:
                continue
            try:
                tokens = shlex.split(stripped, comments=False, posix=True)
            except ValueError:
                continue
            if not tokens:
                continue

            cmd = tokens[0].lower()
            if cmd == "include" and len(tokens) >= 2:
                inc_path = tokens[1]
                if not os.path.isabs(inc_path):
                    inc_path = os.path.join(basedir, inc_path)
                submap = _parse_input_script_elements(inc_path, seen=seen)
                for k, v in submap.items():
                    mapping.setdefault(k, v)

            # Prefer explicit element listing if present:
            # dump_modify <dumpid> element C H O ...
            elif cmd == "dump_modify" and "element" in [t.lower() for t in tokens[1:]]:
                idx = [t.lower() for t in tokens].index("element")
                elems = tokens[idx + 1 :]
                for i, symbol in enumerate(elems, start=1):
                    if _is_valid_symbol(symbol):
                        mapping.setdefault(i, symbol)

            # Common ML pair_style mapping:
            # pair_coeff * * model.pt C H O ...
            elif cmd == "pair_coeff" and len(tokens) >= 5 and tokens[1] == "*" and tokens[2] == "*":
                elems = tokens[4:]
                for i, symbol in enumerate(elems, start=1):
                    if _is_valid_symbol(symbol):
                        mapping.setdefault(i, symbol)

    return mapping


def discover_type_map(input_path: str) -> Dict[int, str]:
    """
    Discover type->element mappings from nearby LAMMPS data files.
    """
    root = os.path.dirname(os.path.abspath(input_path)) or "."
    discovered: Dict[int, str] = {}

    # First, prefer explicit mapping in nearby LAMMPS input scripts.
    input_candidates = []
    for name in os.listdir(root):
        low = name.lower()
        if low.startswith("in.") or low.startswith("in_") or low.startswith("input"):
            input_candidates.append(os.path.join(root, name))
    for path in input_candidates:
        by_input = _parse_input_script_elements(path)
        for k, v in by_input.items():
            discovered.setdefault(k, v)

    # Then fill remaining gaps from data-file masses.
    candidates = [
        os.path.join(root, "data.lammps"),
        os.path.join(root, "data.after_min.lammps"),
    ]
    # Include any additional .lammps/.data files in the trajectory directory.
    for name in os.listdir(root):
        low = name.lower()
        if low.endswith(".lammps") or low.endswith(".data"):
            candidates.append(os.path.join(root, name))

    seen: Set[str] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not os.path.isfile(path):
            continue
        by_mass = _parse_data_file_masses(path)
        for k, v in by_mass.items():
            discovered.setdefault(k, v)
    return discovered


def complete_type_map_from_elements(atoms, type_map: Dict[int, str]) -> List[int]:
    """
    Fill missing type_map entries using per-atom 'element' data when available.
    Returns a sorted list of still-missing atom types after attempting completion.
    """
    atom_types = [int(t) for t in atoms.arrays.get("type", [])]
    missing: Set[int] = {t for t in atom_types if t not in type_map}
    if not missing:
        return []

    if "element" not in atoms.arrays:
        return sorted(missing)

    elements = atoms.arrays["element"]
    if len(elements) != len(atom_types):
        return sorted(missing)

    inferred: Dict[int, str] = {}
    for t, raw_el in zip(atom_types, elements):
        if t in type_map:
            continue
        symbol = _to_text(raw_el)
        if not symbol:
            continue
        prev = inferred.get(t)
        if prev is not None and prev != symbol:
            raise ValueError(
                f"Inconsistent element values for atom type {t}: '{prev}' vs '{symbol}'."
            )
        inferred[t] = symbol

    for t, symbol in inferred.items():
        type_map[t] = symbol

    return sorted({t for t in atom_types if t not in type_map})


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LAMMPS dump to extxyz.")
    parser.add_argument(
        "input",
        help=(
            "Input LAMMPS dump file to convert, or a directory containing a "
            ".lammpstrj file."
        ),
    )
    parser.add_argument(
        "--type-map",
        default="",
        help=(
            "Comma-separated type-to-element mapping (e.g. '1:K,2:Mn'). "
            "Optional; script auto-discovers mapping from nearby LAMMPS files."
        ),
    )
    parser.add_argument(
        "--no-auto-complete-map",
        action="store_false",
        dest="auto_complete_map",
        help=(
            "Disable automatic filling of missing atom types from trajectory "
            "'element' column."
        ),
    )
    parser.set_defaults(auto_complete_map=True)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Write every Nth frame (default: 1).",
    )
    parser.add_argument(
        "--strict-type-map",
        action="store_true",
        help=(
            "Fail if any atom type is missing from --type-map and cannot be inferred "
            "from trajectory 'element' data."
        ),
    )
    parser.add_argument(
        "--fallback-symbol",
        default="X",
        help=(
            "Chemical symbol used for unresolved atom types when not in strict mode "
            "(default: X)."
        ),
    )
    parser.add_argument(
        "--no-wrap",
        action="store_false",
        dest="wrap_atoms",
        help="Disable wrapping atoms back into the simulation cell before writing.",
    )
    parser.set_defaults(wrap_atoms=True)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    input_path = resolve_input_path(args.input)

    output = "trajectory.extxyz"

    type_map = parse_type_map(args.type_map)
    discovered_map = discover_type_map(input_path)
    for k, v in discovered_map.items():
        type_map.setdefault(k, v)

    if os.path.exists(output):
        os.remove(output)

    count_in = 0
    count_out = 0
    warned_fallback = False
    for atoms in iread(input_path, format="lammps-dump-text", index=":"):
        if count_in % args.stride != 0:
            count_in += 1
            continue
        if args.auto_complete_map:
            still_missing = complete_type_map_from_elements(atoms, type_map)
            if still_missing:
                if args.strict_type_map:
                    raise KeyError(
                        "Atom types missing from --type-map and could not infer from trajectory "
                        f"'element' data: {still_missing}. Provided keys: {sorted(type_map.keys())}"
                    )
                for t in still_missing:
                    type_map[t] = args.fallback_symbol
                if not warned_fallback:
                    print(
                        "Warning: some atom types could not be inferred from trajectory "
                        f"'element' data and were mapped to '{args.fallback_symbol}'."
                    )
                    warned_fallback = True
        map_symbols(atoms, type_map)
        if args.wrap_atoms:
            atoms.wrap()
        write(output, atoms, format="extxyz", append=True)
        count_in += 1
        count_out += 1

    if args.auto_complete_map:
        print(f"Final type map used: {', '.join(f'{k}:{v}' for k, v in sorted(type_map.items()))}")

    print(
        f"Converted {count_out} frame(s) from '{input_path}' to '{output}' "
        f"(read {count_in} total frame(s), stride={args.stride})."
    )


if __name__ == "__main__":
    main()
