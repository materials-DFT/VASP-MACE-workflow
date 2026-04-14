#!/usr/bin/env python3
"""
Analyze NPT MD trajectories (XDATCAR + INCAR) to find equilibrium volume at a
target temperature (TEBEG/TEEND) and write POSCAR(s).

Equilibration: by default the script estimates where the volume series has
converged (no manual --skip). The production mean volume is computed after that
cutoff; the POSCAR uses the frame closest to that mean with the lattice scaled
isotropically so the written cell volume equals the mean exactly.

Examples:
  python md_optimal_volume.py /path/to/md_runs --temperature 300
  python md_optimal_volume.py . --list-runs
  python md_optimal_volume.py /path/to/md_runs --run 300K --volume-stat mode --out-dir POSCARs
  # Optional second snapshot (lowest E_pot): add --write-min-epot
  # POSCARs use ASE to wrap fractional coords into the primary cell (see --no-wrap).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# INCAR temperature
# -----------------------------------------------------------------------------

def parse_tebeg_teend(incar_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """Return (TEBEG, TEEND) from INCAR, or (None, None) if missing."""
    text = incar_path.read_text(errors="replace")
    tebeg = teend = None
    m = re.search(r"^\s*TEBEG\s*=\s*([\d.]+)", text, re.MULTILINE | re.IGNORECASE)
    if m:
        tebeg = float(m.group(1))
    m = re.search(r"^\s*TEEND\s*=\s*([\d.]+)", text, re.MULTILINE | re.IGNORECASE)
    if m:
        teend = float(m.group(1))
    return tebeg, teend


def effective_temperature(tebeg: Optional[float], teend: Optional[float]) -> Optional[float]:
    """Single target T for filtering: use TEEND if both set, else whichever exists."""
    if tebeg is not None and teend is not None:
        return teend
    return tebeg if tebeg is not None else teend


def folder_temperature_guess(run_dir: Path) -> Optional[float]:
    """Parse trailing number from names like '300K' or '1000_K'."""
    name = run_dir.name
    m = re.match(r"^(\d+(?:\.\d+)?)\s*K?$", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*K", name, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


# -----------------------------------------------------------------------------
# XDATCAR
# -----------------------------------------------------------------------------

@dataclass
class XdatFrame:
    index: int  # 1-based configuration index from file
    scale: float
    lattice: np.ndarray  # 3x3, rows are lattice vectors (Å)
    species: List[str]
    counts: List[int]
    direct: np.ndarray  # Nx3 fractional coords


def cell_volume(lattice: np.ndarray, scale: float) -> float:
    """Volume (Å^3) for POSCAR-style lattice rows and global scale."""
    m = lattice * scale
    return float(abs(np.linalg.det(m)))


def _sanitize_line(line: str) -> str:
    """Strip embedded NULs and stray CR (some XDATCARs are corrupted / binary-padded)."""
    if not line:
        return line
    return line.replace("\x00", "").replace("\r", "").strip()


def _three_floats(line: str) -> Tuple[float, float, float]:
    """Parse three numbers from a coordinate or lattice line; tolerate garbage after."""
    s = _sanitize_line(line)
    parts = s.split()
    if len(parts) >= 3:
        try:
            return float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            pass
    nums = re.findall(
        r"[-+]?(?:\d*\.\d+|\d+(?:\.\d+)?)(?:[eE][-+]?\d+)?",
        s,
    )
    if len(nums) >= 3:
        return float(nums[0]), float(nums[1]), float(nums[2])
    raise ValueError(f"expected 3 floats, got line: {line[:120]!r}")


class _CoordParseError(Exception):
    """Bad coordinate line; caller has skipped the rest of that frame's coordinates."""

    pass


def isotropic_scale_frame_to_volume(frame: XdatFrame, target_volume: float) -> XdatFrame:
    """
    Scale lattice vectors uniformly so cell volume equals target_volume (Å³).
    Fractional coordinates unchanged (isotropic strain).
    """
    v_now = cell_volume(frame.lattice, frame.scale)
    if v_now <= 0:
        raise ValueError("Non-positive cell volume")
    factor = (target_volume / v_now) ** (1.0 / 3.0)
    new_lat = frame.lattice * factor
    new_scale = frame.scale  # absorb into lattice rows; keep scale as-is for VASP
    return XdatFrame(
        index=frame.index,
        scale=new_scale,
        lattice=new_lat,
        species=frame.species,
        counts=frame.counts,
        direct=frame.direct.copy(),
    )


def wrap_xdat_frame(frame: XdatFrame) -> XdatFrame:
    """
    Map fractional coordinates into the primary unit cell using ASE wrap()
    (handles triclinic cells; fixes unwrapped MD trajectories).
    """
    try:
        from ase import Atoms
    except ImportError as e:
        raise ImportError(
            "ASE is required to wrap coordinates (pip install ase), or pass --no-wrap"
        ) from e

    symbols: List[str] = []
    for sym, n in zip(frame.species, frame.counts):
        symbols.extend([sym] * n)
    cell = frame.scale * frame.lattice
    atoms = Atoms(
        symbols=symbols,
        scaled_positions=np.asarray(frame.direct, dtype=float),
        cell=cell,
        pbc=True,
    )
    atoms.wrap()
    wrapped = atoms.get_scaled_positions()
    return XdatFrame(
        index=frame.index,
        scale=frame.scale,
        lattice=frame.lattice.copy(),
        species=frame.species,
        counts=frame.counts,
        direct=np.asarray(wrapped, dtype=np.float64),
    )


def auto_equilibration_skip(
    vols: np.ndarray,
    *,
    ref_frac: float,
    rtol: float,
    atol: float,
    max_skip_frac: float,
) -> Tuple[int, dict]:
    """
    Estimate equilibration length: compare tail means to the mean of the last
    ref_frac of the trajectory (taken as equilibrium reference).

    Returns the smallest skip k such that mean(vols[k:]) is within rtol/atol of
    that reference. If none found, uses argmin over k in [0, floor(max_skip_frac*n)].
    """
    n = len(vols)
    meta: dict = {"method": "tail_reference"}
    if n < 20:
        meta["note"] = "short_traj"
        return 0, meta

    ref_frac = min(0.49, max(0.05, ref_frac))
    i0 = int(round(n * (1.0 - ref_frac)))
    i0 = max(1, min(i0, n - 2))
    ref = float(np.mean(vols[i0:]))
    thr = max(atol, rtol * abs(ref))
    meta["ref_volume_tail"] = ref
    meta["ref_tail_first_index"] = i0
    meta["ref_frac"] = ref_frac

    upper = min(n - 2, int(n * max_skip_frac))
    chosen = None
    for k in range(0, upper + 1):
        tail = float(np.mean(vols[k:]))
        if abs(tail - ref) <= thr:
            chosen = k
            break
    if chosen is None:
        ks = np.arange(0, upper + 1, dtype=int)
        errs = np.array([abs(float(np.mean(vols[k:])) - ref) for k in ks])
        chosen = int(ks[int(np.argmin(errs))])
        meta["method"] = "tail_reference_min_err"

    meta["auto_skip"] = chosen
    return chosen, meta


def _parse_frame_after_title(f, title: str, path: Path) -> XdatFrame:
    """Read one frame from a VASP 5+ XDATCAR after the title line has been read."""
    scale_line = f.readline()
    if not scale_line:
        raise ValueError(f"EOF after title in {path}")
    try:
        scale = float(_sanitize_line(scale_line).split()[0])
    except (IndexError, ValueError) as e:
        raise ValueError(f"bad scale line in {path}") from e
    lat = np.zeros((3, 3))
    for i in range(3):
        line = f.readline()
        if not line:
            raise ValueError(f"Truncated lattice in {path}")
        lat[i] = _three_floats(line)
    elem_line = f.readline()
    if not elem_line:
        raise ValueError(f"Missing element line in {path}")
    species = _sanitize_line(elem_line).split()
    if not species:
        raise ValueError(f"Empty element line in {path}")
    count_line = f.readline()
    if not count_line:
        raise ValueError(f"Missing counts in {path}")
    try:
        counts = [int(x) for x in _sanitize_line(count_line).split()]
    except ValueError as e:
        raise ValueError(f"bad counts line in {path}") from e
    ntot = sum(counts)
    head = f.readline()
    head = _sanitize_line(head) if head else ""
    if not head or ("Direct" not in head and "direct" not in head):
        raise ValueError(f"Expected 'Direct configuration' line, got: {head!r}")
    m = re.search(r"=\s*(\d+)", head)
    idx = int(m.group(1)) if m else 0
    direct = np.zeros((ntot, 3))
    for i in range(ntot):
        line = f.readline()
        if not line:
            raise ValueError(f"Truncated coords at atom {i} in {path}")
        try:
            direct[i] = _three_floats(line)
        except ValueError:
            for _ in range(ntot - i - 1):
                f.readline()
            raise _CoordParseError(f"bad coordinate line {i} in {path}")
    return XdatFrame(
        index=idx,
        scale=scale,
        lattice=lat,
        species=species,
        counts=counts,
        direct=direct,
    )


def iter_xdatcar(path: Path, stats: Optional[dict] = None) -> Generator[XdatFrame, None, None]:
    """
    Stream frames from VASP 5+ XDATCAR (repeated title blocks per frame).
    Tolerates embedded NUL bytes in lines (corrupted or non-text-safe files).
    Frames that cannot be parsed (corrupt lines) are skipped; if the file is
    seekable, resynchronizes by scanning forward one line at a time until the
    next valid frame. If stats is set, stats["skipped_frames"] is incremented
    for each dropped frame.
    """
    with path.open(encoding="utf-8", errors="replace") as f:
        seekable = getattr(f, "seekable", lambda: False)()
        while True:
            pos = f.tell()
            title = f.readline()
            if not title:
                break
            try:
                yield _parse_frame_after_title(f, title, path)
            except _CoordParseError:
                # Definite bad frame: header parsed, coordinate line failed.
                if stats is not None:
                    stats["skipped_frames"] = stats.get("skipped_frames", 0) + 1
                continue
            except ValueError:
                # Likely wrong line used as frame start; resync without counting as a frame.
                if stats is not None:
                    stats["resync_steps"] = stats.get("resync_steps", 0) + 1
                if not seekable:
                    raise ValueError(
                        f"cannot resync XDATCAR (not seekable): {path}; "
                        "use a regular file on disk."
                    ) from None
                try:
                    f.seek(pos)
                except OSError as e:
                    raise ValueError(f"cannot resync XDATCAR: {path}") from e
                discarded = f.readline()
                if not discarded:
                    break


def collect_volumes(xdat_path: Path, stats: Optional[dict] = None) -> np.ndarray:
    vols: List[float] = []
    for fr in iter_xdatcar(xdat_path, stats=stats):
        vols.append(cell_volume(fr.lattice, fr.scale))
    return np.array(vols, dtype=np.float64)


def read_xdatcar_frame_at_offset(path: Path, zero_based: int) -> XdatFrame:
    """Return the n-th frame in file order (0-based)."""
    for i, fr in enumerate(iter_xdatcar(path)):
        if i == zero_based:
            return fr
    raise IndexError(f"Frame index {zero_based} out of range for {path}")


# -----------------------------------------------------------------------------
# REPORT (optional E_pot)
# -----------------------------------------------------------------------------

def parse_report_epot(report_path: Path) -> np.ndarray:
    """
    Extract potential energy per MD step from REPORT (same order as XDATCAR frames).
    Lines look like:   e_b>   E_tot   E_pot   E_kin ...
    """
    epots: List[float] = []
    epot_re = re.compile(
        r"^\s*e_b>\s+[\d.E+-]+\s+([\d.E+-]+)",
        re.IGNORECASE,
    )
    with report_path.open(errors="replace") as f:
        for line in f:
            m = epot_re.match(line)
            if m:
                epots.append(float(m.group(1)))
    return np.array(epots, dtype=np.float64)


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def volume_statistic(
    vols: np.ndarray,
    skip: int,
    stat: str,
    hist_bins: int,
) -> Tuple[float, dict]:
    """
    Compute target volume from production segment vols[skip:].
    Returns (value, extra_info dict).
    """
    if skip < 0 or skip >= len(vols):
        raise ValueError(f"skip={skip} invalid for {len(vols)} frames")
    v = vols[skip:]
    if len(v) == 0:
        raise ValueError("No frames left after --skip")
    out: dict = {"n_frames": len(v), "skip": skip}
    if stat == "mean":
        val = float(np.mean(v))
        out["std"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        return val, out
    if stat == "median":
        return float(np.median(v)), out
    if stat == "mode":
        hist, edges = np.histogram(v, bins=hist_bins, density=False)
        j = int(np.argmax(hist))
        mode = 0.5 * (edges[j] + edges[j + 1])
        out["hist_peak_count"] = int(hist[j])
        out["hist_bin_edges"] = (float(edges[j]), float(edges[j + 1]))
        return float(mode), out
    raise ValueError(f"Unknown volume-stat {stat!r}")


def index_closest_volume(vols: np.ndarray, target: float, skip: int) -> int:
    """Global index (into full vols) of production frame minimizing |V - target|."""
    segment = vols[skip:]
    rel = int(np.argmin(np.abs(segment - target)))
    return skip + rel


def index_min_epot(epots: np.ndarray, skip: int) -> int:
    segment = epots[skip:]
    rel = int(np.argmin(segment))
    return skip + rel


# -----------------------------------------------------------------------------
# POSCAR writer
# -----------------------------------------------------------------------------

def write_poscar(
    path: Path,
    frame: XdatFrame,
    title: str = "written by md_optimal_volume.py",
    *,
    wrap: bool = True,
) -> None:
    """Write VASP 5 format POSCAR with Direct coordinates. Optionally wrap into the cell (ASE)."""
    if wrap:
        frame = wrap_xdat_frame(frame)
    lines: List[str] = [
        title.strip()[:80] or "structure",
        f"{frame.scale:.16f}".rstrip("0").rstrip(".") if "." in f"{frame.scale}" else str(frame.scale),
    ]
    for row in frame.lattice:
        lines.append(" ".join(f"{x:18.12f}" for x in row))
    lines.append(" ".join(frame.species))
    lines.append(" ".join(str(c) for c in frame.counts))
    lines.append("Direct")
    for r in frame.direct:
        lines.append(" ".join(f"{x:18.12f}" for x in r))
    path.write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------------
# Discovery
# -----------------------------------------------------------------------------

@dataclass
class MdRun:
    path: Path
    tebeg: Optional[float]
    teend: Optional[float]
    t_eff: Optional[float]
    folder_guess: Optional[float]

    @property
    def xdatcar(self) -> Path:
        return self.path / "XDATCAR"

    @property
    def incar(self) -> Path:
        return self.path / "INCAR"

    @property
    def report(self) -> Path:
        return self.path / "REPORT"


def discover_runs(root: Path) -> List[MdRun]:
    """Find every directory under root that contains INCAR + XDATCAR (recursive)."""
    runs: List[MdRun] = []
    for incar in sorted(root.rglob("INCAR")):
        run_dir = incar.parent
        if not (run_dir / "XDATCAR").is_file():
            continue
        tebeg, teend = parse_tebeg_teend(incar)
        t_eff = effective_temperature(tebeg, teend)
        runs.append(
            MdRun(
                path=run_dir.resolve(),
                tebeg=tebeg,
                teend=teend,
                t_eff=t_eff,
                folder_guess=folder_temperature_guess(run_dir),
            )
        )
    return runs


def match_temperature(run: MdRun, target: float, atol: float) -> bool:
    if run.t_eff is not None and abs(run.t_eff - target) <= atol:
        return True
    if run.folder_guess is not None and abs(run.folder_guess - target) <= atol:
        return True
    return False


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "workdir",
        nargs="?",
        default=None,
        type=Path,
        metavar="DIR",
        help="Directory to run in before resolving paths (optional; default is the current working directory)",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory searched recursively for INCAR+XDATCAR pairs (default: .)",
    )
    p.add_argument("--run", type=Path, default=None, help="Single run directory (skips discovery)")
    p.add_argument("--temperature", "-T", type=float, default=None, help="Target temperature (K), match TEEND/TEBEG or folder name")
    p.add_argument("--temp-tol", type=float, default=0.5, help="Tolerance when matching temperature (K)")
    p.add_argument("--list-runs", action="store_true", help="List discovered runs and exit")
    p.add_argument(
        "--skip",
        type=str,
        default="auto",
        help="Initial frames to discard: 'auto' (default: detect equilibration) or a non‑negative integer",
    )
    p.add_argument(
        "--auto-ref-frac",
        type=float,
        default=0.35,
        help="Fraction of trajectory at the end used as equilibrium reference for auto skip (default: 0.35)",
    )
    p.add_argument(
        "--auto-rtol",
        type=float,
        default=5e-4,
        help="Relative tolerance for matching tail mean (auto skip)",
    )
    p.add_argument(
        "--auto-atol",
        type=float,
        default=0.02,
        help="Absolute tolerance (Å³) combined with auto-rtol for auto skip",
    )
    p.add_argument(
        "--auto-max-skip-frac",
        type=float,
        default=0.55,
        help="Max fraction of trajectory to search as equilibration (auto skip)",
    )
    p.add_argument(
        "--volume-stat",
        choices=("mean", "median", "mode"),
        default="mean",
        help="How to define optimal volume from production segment (default: mean)",
    )
    p.add_argument("--hist-bins", type=int, default=60, help="Histogram bins for mode")
    p.add_argument(
        "--output-name",
        type=str,
        default="POSCAR_MD_equil",
        metavar="NAME",
        help="Output filename for the equilibrated structure (default: POSCAR_MD_equil)",
    )
    p.add_argument(
        "--write-min-epot",
        action="store_true",
        help="Also write a second POSCAR from the lowest E_pot frame (REPORT); "
        "default is a single file from mean volume only",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Directory for POSCAR(s) (default: run dir)")
    p.add_argument(
        "--no-wrap",
        action="store_true",
        help="Do not wrap fractional coords into the primary cell (default: use ASE wrap)",
    )

    args = p.parse_args(argv)

    if args.workdir is not None:
        work = args.workdir.expanduser().resolve()
        if not work.is_dir():
            print(f"ERROR: not a directory: {work}", file=sys.stderr)
            return 1
        try:
            os.chdir(work)
        except OSError as e:
            print(f"ERROR: cannot change to {work}: {e}", file=sys.stderr)
            return 1

    skip_auto = args.skip.strip().lower() == "auto"
    fixed_skip: Optional[int] = None
    if not skip_auto:
        try:
            fixed_skip = int(args.skip, 10)
        except ValueError:
            print("ERROR: --skip must be 'auto' or a non-negative integer", file=sys.stderr)
            return 1
        if fixed_skip < 0:
            print("ERROR: --skip must be non-negative", file=sys.stderr)
            return 1

    if args.run is not None:
        run_path = args.run.resolve()
        incar = run_path / "INCAR"
        if not incar.is_file():
            print(f"ERROR: no INCAR in {run_path}", file=sys.stderr)
            return 1
        tebeg, teend = parse_tebeg_teend(incar)
        runs = [
            MdRun(
                path=run_path,
                tebeg=tebeg,
                teend=teend,
                t_eff=effective_temperature(tebeg, teend),
                folder_guess=folder_temperature_guess(run_path),
            )
        ]
    else:
        runs = discover_runs(args.root.resolve())

    if not runs:
        print("No runs with INCAR and XDATCAR found.", file=sys.stderr)
        return 1

    if args.list_runs:
        print(f"{'Path':<50} {'TEBEG':>8} {'TEEND':>8} {'T_eff':>8} {'folder':>8}")
        for r in runs:
            print(
                f"{str(r.path):<50} "
                f"{r.tebeg if r.tebeg is not None else '—':>8} "
                f"{r.teend if r.teend is not None else '—':>8} "
                f"{r.t_eff if r.t_eff is not None else '—':>8} "
                f"{r.folder_guess if r.folder_guess is not None else '—':>8}"
            )
        return 0

    if args.temperature is not None:
        selected = [r for r in runs if match_temperature(r, args.temperature, args.temp_tol)]
        if not selected:
            print(
                f"No run matched T={args.temperature} K (tol={args.temp_tol}). Use --list-runs.",
                file=sys.stderr,
            )
            return 1
        runs = selected

    out_dir = args.out_dir
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    for run in runs:
        xc = run.xdatcar
        if not xc.is_file():
            print(f"SKIP (no XDATCAR): {run.path}", file=sys.stderr)
            continue

        print(f"\n=== {run.path} ===")
        tebeg, teend = run.tebeg, run.teend
        print(f"INCAR TEBEG={tebeg} TEEND={teend}  (effective T for ramp: {run.t_eff})")

        xc_stats: dict = {}
        try:
            vols = collect_volumes(xc, stats=xc_stats)
        except (ValueError, OSError) as e:
            print(f"SKIP (cannot read XDATCAR): {run.path}", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
            exit_code = 1
            continue
        except Exception as e:
            print(f"SKIP (unexpected error reading XDATCAR): {run.path}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            exit_code = 1
            continue

        n_skipped = int(xc_stats.get("skipped_frames", 0))
        if n_skipped:
            print(
                f"WARNING: skipped {n_skipped} unreadable frame(s) in XDATCAR; "
                "volumes use only good frames (REPORT/E_pot alignment may be off if used).",
                file=sys.stderr,
            )

        n = len(vols)
        if n == 0:
            print(f"ERROR: no readable frames in XDATCAR: {run.path}", file=sys.stderr)
            exit_code = 1
            continue
        print(f"Frames: {n}, volume range [{vols.min():.4f}, {vols.max():.4f}] Å³")

        try:
            if skip_auto:
                skip, skip_meta = auto_equilibration_skip(
                    vols,
                    ref_frac=args.auto_ref_frac,
                    rtol=args.auto_rtol,
                    atol=args.auto_atol,
                    max_skip_frac=args.auto_max_skip_frac,
                )
                rtail = skip_meta.get("ref_volume_tail")
                rtail_s = f"{rtail:.6f}" if rtail is not None else "nan"
                print(
                    f"Auto equilibration: skip={skip} frames "
                    f"(ref tail mean={rtail_s} Å³, method={skip_meta.get('method', '')})"
                )
            else:
                assert fixed_skip is not None
                skip = fixed_skip
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            exit_code = 1
            continue

        try:
            opt_v, info = volume_statistic(vols, skip, args.volume_stat, args.hist_bins)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            exit_code = 1
            continue

        print(f"Optimal volume ({args.volume_stat}, after skip={skip}): {opt_v:.6f} Å³")
        if "std" in info:
            print(f"  ± {info['std']:.6f} Å³ (1σ over production)")

        # --- Primary output: one POSCAR per run (mean/median/mode volume + isotropic scale) ---
        g_idx = index_closest_volume(vols, opt_v, skip)
        frame = read_xdatcar_frame_at_offset(xc, g_idx)
        v_raw = cell_volume(frame.lattice, frame.scale)
        frame_out = isotropic_scale_frame_to_volume(frame, opt_v)
        v_out = cell_volume(frame_out.lattice, frame_out.scale)

        base = (out_dir if out_dir is not None else run.path)
        out_main = base / args.output_name
        title_main = (
            f"NPT equil. MD T={tebeg}/{teend} K cfg={frame.index} "
            f"V={v_out:.4f}A3 isotropic_to_{args.volume_stat}"
        )
        write_poscar(out_main, frame_out, title=title_main, wrap=not args.no_wrap)
        print(
            f"Wrote {out_main}  (frame {g_idx + 1}, config={frame.index}, "
            f"V_raw={v_raw:.4f} Å³ → scaled to {args.volume_stat} V={v_out:.6f} Å³)"
        )

        # --- Optional second file: lowest E_pot snapshot (not at mean volume) ---
        if args.write_min_epot:
            rp = run.report
            if not rp.is_file():
                print("ERROR: --write-min-epot requires REPORT", file=sys.stderr)
                exit_code = 1
                continue
            epots = parse_report_epot(rp)
            vols_e = vols
            n_e = n
            if len(epots) != n_e:
                print(
                    f"WARNING: REPORT E_pot steps ({len(epots)}) != XDATCAR frames ({n_e}); "
                    f"truncating to min length.",
                    file=sys.stderr,
                )
                mlen = min(len(epots), n_e)
                epots = epots[:mlen]
                vols_e = vols_e[:mlen]
            g2 = index_min_epot(epots, skip)
            fr2 = read_xdatcar_frame_at_offset(xc, g2)
            v2 = cell_volume(fr2.lattice, fr2.scale)
            out_ep = base / "POSCAR_MD_min_epot"
            title_ep = (
                f"min E_pot MD T={tebeg}/{teend} K cfg={fr2.index} V={v2:.4f}A3"
            )
            write_poscar(out_ep, fr2, title=title_ep, wrap=not args.no_wrap)
            print(
                f"Wrote {out_ep}  (frame {g2 + 1}, E_pot={epots[g2]:.8f} eV, V={v2:.4f} Å³)"
            )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
