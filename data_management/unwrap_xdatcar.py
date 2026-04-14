#!/usr/bin/env python3
"""
Recursively unwrap VASP XDATCAR trajectories in a directory tree.

For each XDATCAR found, writes a new file with unwrapped coordinates:
    XDATCAR -> unwrapped_XDATCAR

Unwrapping is done in fractional coordinates using the minimum-image
convention between consecutive frames:
    delta_unwrapped = delta - round(delta)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def _parse_header(lines: List[str]) -> Tuple[int, List[str], int]:
    """
    Parse XDATCAR header and return:
      - natoms_total
      - header_lines (everything before first coordinate line)
      - first_config_idx (line index where first "Direct configuration" appears)
    """
    if len(lines) < 8:
        raise ValueError("File too short to be a valid XDATCAR.")

    # Standard XDATCAR:
    # 1 comment
    # 2 scale
    # 3-5 lattice vectors
    # 6 element names
    # 7 element counts
    counts_line_idx = 6
    try:
        counts = [int(x) for x in lines[counts_line_idx].split()]
    except Exception as exc:
        raise ValueError("Could not parse atom counts line in header.") from exc

    if not counts:
        raise ValueError("Atom counts line is empty.")
    natoms = sum(counts)

    first_cfg = None
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("direct configuration"):
            first_cfg = i
            break

    if first_cfg is None:
        raise ValueError("No 'Direct configuration' lines found.")

    header = lines[:first_cfg]
    return natoms, header, first_cfg


def _read_frame_coords(lines: List[str], start: int, natoms: int) -> Tuple[List[List[float]], int]:
    """
    Read natoms coordinate lines from lines[start:start+natoms].
    Returns (coords, next_index).
    coords are fractional [x, y, z].
    """
    if start + natoms > len(lines):
        raise ValueError("Unexpected end of file while reading coordinates.")

    coords: List[List[float]] = []
    for i in range(start, start + natoms):
        parts = lines[i].split()
        if len(parts) < 3:
            raise ValueError(f"Malformed coordinate line at line {i + 1}.")
        try:
            xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
        except Exception as exc:
            raise ValueError(f"Non-numeric coordinate at line {i + 1}.") from exc
        coords.append(xyz)
    return coords, start + natoms


def _unwrap_frames(frames: List[List[List[float]]]) -> List[List[List[float]]]:
    """
    Unwrap fractional coordinates over time.
    """
    if not frames:
        return []

    unwrapped = [frames[0]]
    prev_wrapped = frames[0]
    prev_unwrapped = [c[:] for c in frames[0]]

    for frame in frames[1:]:
        this_unwrapped: List[List[float]] = []
        for i, cur in enumerate(frame):
            prev_w = prev_wrapped[i]
            prev_u = prev_unwrapped[i]
            # Minimum image in fractional space:
            # wrapped delta in (-0.5, 0.5] style via round
            d0 = cur[0] - prev_w[0]
            d1 = cur[1] - prev_w[1]
            d2 = cur[2] - prev_w[2]
            du0 = d0 - round(d0)
            du1 = d1 - round(d1)
            du2 = d2 - round(d2)
            this_unwrapped.append([prev_u[0] + du0, prev_u[1] + du1, prev_u[2] + du2])
        unwrapped.append(this_unwrapped)
        prev_wrapped = frame
        prev_unwrapped = this_unwrapped

    return unwrapped


def unwrap_xdatcar(path: Path, output_name: str = "unwrapped_XDATCAR", overwrite: bool = False) -> Path:
    text = path.read_text()
    lines = text.splitlines()
    natoms, header, first_cfg_idx = _parse_header(lines)

    # Parse frames by scanning for "Direct configuration" markers.
    # Some XDATCAR files (e.g., concatenated restart segments) contain
    # repeated header blocks between frames; these are skipped.
    frames: List[List[List[float]]] = []
    frame_headers: List[str] = []
    idx = first_cfg_idx
    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        if not line.lower().startswith("direct configuration"):
            idx += 1
            continue

        frame_headers.append(lines[idx])
        idx += 1
        coords, idx = _read_frame_coords(lines, idx, natoms)
        frames.append(coords)

    unwrapped = _unwrap_frames(frames)

    out_lines: List[str] = []
    out_lines.extend(header)
    for fh, frame in zip(frame_headers, unwrapped):
        out_lines.append(fh)
        for x, y, z in frame:
            out_lines.append(f"{x:20.16f} {y:20.16f} {z:20.16f}")

    out_path = path.with_name(output_name)
    if overwrite:
        out_path = path

    out_path.write_text("\n".join(out_lines) + "\n")
    return out_path


def find_xdatcar_files(root: Path, filename: str) -> List[Path]:
    return sorted(p for p in root.rglob(filename) if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively unwrap XDATCAR trajectories in a directory tree."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Root directory to search recursively for XDATCAR files.",
    )
    parser.add_argument(
        "--filename",
        default="XDATCAR",
        help="Filename to look for recursively (default: XDATCAR).",
    )
    parser.add_argument(
        "--output-name",
        default="unwrapped_XDATCAR",
        help="Output filename to write next to each input (default: unwrapped_XDATCAR).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input files instead of writing suffixed outputs.",
    )
    args = parser.parse_args()

    root = args.directory.resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: directory does not exist or is not a directory: {root}", file=sys.stderr)
        return 2

    files = find_xdatcar_files(root, args.filename)
    if not files:
        print(f"No files named '{args.filename}' found under: {root}")
        return 0

    print(f"Found {len(files)} file(s) named '{args.filename}' under {root}")
    ok = 0
    failed = 0
    for fp in files:
        try:
            out = unwrap_xdatcar(fp, output_name=args.output_name, overwrite=args.overwrite)
            if args.overwrite:
                print(f"[OK]  {fp} (overwritten)")
            else:
                print(f"[OK]  {fp} -> {out}")
            ok += 1
        except Exception as exc:
            print(f"[ERR] {fp}: {exc}", file=sys.stderr)
            failed += 1

    print(f"Done. success={ok}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
