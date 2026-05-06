#!/usr/bin/env python3
"""
Convert a LAMMPS trajectory dump (lammpstrj) to extxyz.

Example:
  python convert_trajectory_to_extxyz.py trajectory.lammpstrj
  python convert_trajectory_to_extxyz.py .                    # recursive under .
  python convert_trajectory_to_extxyz.py 300K/ 700K/
"""

from __future__ import annotations

import argparse
import os
import shlex
from typing import Any, Dict, List, Optional, Set

import numpy as np
from ase.io import iread, write
from ase.data import atomic_masses, chemical_symbols


def collect_lammpstrj_files(
    paths: List[str], *, recursive: bool = True
) -> List[str]:
    """
    Expand CLI paths to a sorted, unique list of *.lammpstrj file paths.

    Each path may be a trajectory file or a directory. Directories are
    scanned for .lammpstrj (recursively by default).
    """
    found: Set[str] = set()
    for path in paths:
        apath = os.path.abspath(path)
        if os.path.isfile(apath):
            if apath.lower().endswith(".lammpstrj"):
                found.add(apath)
            else:
                raise ValueError(f"Not a .lammpstrj file: {path}")
        elif os.path.isdir(apath):
            if recursive:
                for root, _dirs, files in os.walk(apath):
                    for name in files:
                        if name.lower().endswith(".lammpstrj"):
                            found.add(os.path.join(root, name))
            else:
                for name in os.listdir(apath):
                    fp = os.path.join(apath, name)
                    if (
                        name.lower().endswith(".lammpstrj")
                        and os.path.isfile(fp)
                    ):
                        found.add(fp)
        else:
            raise FileNotFoundError(f"Input path not found: {path}")
    return sorted(found)


def default_output_path_for_trajectory(traj_path: str) -> str:
    """Sidecar output: same directory, stem replaced with .extxyz."""
    d = os.path.dirname(os.path.abspath(traj_path))
    stem = os.path.splitext(os.path.basename(traj_path))[0]
    return os.path.join(d, f"{stem}.extxyz")


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


def map_symbols(
    atoms,
    type_map: Dict[int, str],
    sym_state: Optional[List[Any]] = None,
) -> None:
    """Assign chemical symbols using LAMMPS atom types.

    Optionally pass ``sym_state = [frozenset|None, list|None]`` to reuse a lookup
    table while ``type_map`` is unchanged between frames."""
    if "type" not in atoms.arrays:
        raise KeyError(
            "No 'type' array found in trajectory frame. "
            "Ensure input is a LAMMPS dump with a 'type' column."
        )
    fk = frozenset(type_map.items())
    atom_types = atoms.arrays["type"]

    if sym_state is not None and sym_state[0] == fk:
        tab = sym_state[1]
    else:
        mx_key = max(type_map.keys(), default=0)
        tab = [None] * (mx_key + 1)
        for k, v in type_map.items():
            if k >= 0:
                tab[k] = v
        if sym_state is not None:
            sym_state[0] = fk
            sym_state[1] = tab
    ta_max = int(np.max(atom_types))
    ta_min = int(np.min(atom_types))
    if ta_min < 0 or ta_max >= len(tab):
        raise KeyError(
            f"Atom type out of compiled table ({ta_min}..{ta_max}); "
            f"type_map keys: {sorted(type_map.keys())}"
        )

    symbols_list: List[str] = []
    get = tab.__getitem__
    for t in atom_types:
        it = int(t)
        s = get(it)
        if s is None:
            raise KeyError(
                f"Atom type {it} not found in --type-map. "
                f"Provided keys: {sorted(type_map.keys())}"
            )
        symbols_list.append(s)
    atoms.set_chemical_symbols(symbols_list)


def frame_needs_element_inference(
    atom_types_arr: np.ndarray, type_map: Dict[int, str]
) -> bool:
    """True if some integer types in ``atom_types_arr`` are missing from ``type_map``."""
    if atom_types_arr.size == 0:
        return False
    uniq = np.unique(atom_types_arr.astype(np.int64, copy=False))
    keys = type_map.keys()
    for u in uniq:
        if int(u) not in keys:
            return True
    return False


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


def convert_trajectory(
    input_path: str, output_path: str, args: argparse.Namespace
) -> None:
    """Read one lammpstrj and write frames to output_path as extxyz."""
    type_map = parse_type_map(args.type_map)
    discovered_map = discover_type_map(input_path)
    for k, v in discovered_map.items():
        type_map.setdefault(k, v)

    if os.path.exists(output_path):
        os.remove(output_path)

    sym_cache: List[Any] = [None, None]
    count_in = 0
    count_out = 0
    warned_fallback = False
    pending_batch: List[Any] = []
    wb = max(1, int(args.write_batch))
    write_kwargs = dict(
        format="extxyz",
        parallel=False,
        write_results=False,
    )

    with open(output_path, "w", encoding="utf-8", newline="\n") as fd:
        for atoms in iread(input_path, format="lammps-dump-text", index=":"):
            if count_in % args.stride != 0:
                count_in += 1
                continue

            ta = atoms.arrays.get("type")
            if ta is None:
                raise KeyError(
                    "No 'type' array found in trajectory frame. "
                    "Ensure input is a LAMMPS dump with a 'type' column."
                )

            if args.auto_complete_map and frame_needs_element_inference(ta, type_map):
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

            map_symbols(atoms, type_map, sym_state=sym_cache)
            if args.wrap_atoms:
                atoms.wrap()

            pending_batch.append(atoms)
            if len(pending_batch) >= wb:
                write(fd, pending_batch, **write_kwargs)
                pending_batch.clear()

            count_in += 1
            count_out += 1

        if pending_batch:
            write(fd, pending_batch, **write_kwargs)

    if args.auto_complete_map:
        print(
            f"Final type map used: {', '.join(f'{k}:{v}' for k, v in sorted(type_map.items()))}"
        )

    print(
        f"Converted {count_out} frame(s) from '{input_path}' to '{output_path}' "
        f"(read {count_in} total frame(s), stride={args.stride})."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LAMMPS dump to extxyz.")
    parser.add_argument(
        "input",
        nargs="+",
        help=(
            "One or more LAMMPS dump files and/or directories. "
            "Directories are searched for *.lammpstrj (recursively by default)."
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="When input is a directory, only look for .lammpstrj in that directory.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output extxyz path (only when exactly one trajectory is converted). "
            "Default: './trajectory.extxyz' for a single file; for multiple trajectories, "
            "each input is written next to it as <stem>.extxyz."
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
        "--write-batch",
        type=int,
        default=64,
        metavar="N",
        help=(
            "Buffer this many converted frames before each extxyz write to the output file "
            "(default: 64). Larger batches reduce overhead; reduce if memory is tight."
        ),
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

    recursive = not args.no_recursive
    traj_files = collect_lammpstrj_files(args.input, recursive=recursive)
    if not traj_files:
        roots = ", ".join(os.path.abspath(p) for p in args.input)
        raise FileNotFoundError(
            f"No .lammpstrj file found under input path(s): {roots}"
        )

    if args.output and len(traj_files) > 1:
        raise ValueError("--output is only allowed when converting one trajectory.")

    for i, input_path in enumerate(traj_files):
        if len(traj_files) == 1:
            output = (
                os.path.abspath(args.output)
                if args.output
                else os.path.abspath("trajectory.extxyz")
            )
        else:
            output = default_output_path_for_trajectory(input_path)
        if len(traj_files) > 1:
            print(f"[{i + 1}/{len(traj_files)}] ", end="")
        convert_trajectory(input_path, output, args)


if __name__ == "__main__":
    main()
