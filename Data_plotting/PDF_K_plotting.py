#!/usr/bin/env python3
"""Plot the **pair distribution function** g(r) for K–X pairs from MD trajectories.

For isotropic fluids and the usual MD normalization, **g(r) is the same object**
whether you call it the pair distribution function (PDF) or the radial
distribution function (RDF). This script shows partial g(r) for every channel that
involves K, including **K–K** by default (use ``--no-kk`` to omit K–K). The analysis
code uses the name ``compute_rdf`` / result key ``rdf`` for
historical reasons only—the quantity plotted is g(r).

Simulation discovery matches ``RDF_MSD_evaluation.py``: directories are expanded with
``VACF_RDF_MSD_evaluation.resolve_dirs``, and trajectories are read by the same
``analyze_one`` driver as in that workflow:

- **VASP:** ``XDATCAR`` (preferred) or ``OUTCAR``; timestep from ``INCAR`` / ``OUTCAR``.
- **LAMMPS:** ``*.extxyz`` / ``*.xyz``; timestep from ``in.*``.

**MLFF vs DFT:** pass two parent trees (e.g. ``vasp_runs/ mlff_runs/``) that each contain
matching temperature leaves (``300K/``, …). Curves at the same temperature are drawn in
one panel. When leaf folder names collide (two ``300K``), legend prefixes **DFT** /
**MLFF** come from ``finalize_plot_labels`` (VASP → DFT, LAMMPS → MLFF), same as
``RDF_MSD_evaluation.py``. Override with ``--labels`` or ``--series-prefixes``.

In a panel that contains **both** VASP and LAMMPS, **DFT is drawn with solid lines**
and **MLFF with dotted lines** (``:``). Each **pair type** (K–O, K–Mn, …) has its **own
subplot** within the temperature row; every subplot lives in the **same** matplotlib
window (one X11 figure).

Examples
--------
  %(prog)s 300K/ 700K/
  %(prog)s . --elements O Mn
  %(prog)s ~/vasp/oms6/md/ ~/lammps/oms6/       # DFT vs MLFF, auto labels if leaves match
  %(prog)s dft/ mlff/ --series-prefixes DFT MLFF
  %(prog)s dft/300K mlff/300K --labels DFT MLFF
  %(prog)s 300K/ --all-frames
  %(prog)s 300K/ --skip-equil-15k
  %(prog)s 300K/ --no-kk          # heteronuclear K–X only (drop K–K)

Requires a graphical session: ``DISPLAY`` must be set (X11). Figures are shown
interactively only; no image files are written.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_WORKFLOW_PLOT = Path.home() / "VASP-MACE-workflow" / "Data_plotting"
for _p in (_SCRIPT_DIR, _WORKFLOW_PLOT):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

import VACF_RDF_MSD_evaluation as core


def _pair_species(pair_key: str) -> tuple[str, str] | None:
    parts = pair_key.split("-")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def select_k_pairs(
    rdf_keys: set[str] | frozenset,
    *,
    elements: list[str] | None,
    include_kk: bool,
) -> list[str]:
    """Return sorted partial g(r) channel labels that involve potassium (``K``).

    Labels match ``compute_rdf`` output keys: ``\"A-B\"`` with ``A`` <= ``B`` lexicographically.
    """
    out: list[str] = []
    for k in rdf_keys:
        if k == "total":
            continue
        sp = _pair_species(k)
        if sp is None:
            continue
        a, b = sp
        if a != "K" and b != "K":
            continue
        if not include_kk and a == "K" and b == "K":
            continue
        if elements is not None:
            elems_set = set(elements)
            other = b if a == "K" else a
            if other == "K":
                if not include_kk or "K" not in elems_set:
                    continue
            elif other not in elems_set:
                continue
        out.append(k)
    return sorted(out)


def _sort_series_pairs_for_comparison(
    series_pairs: list[tuple[dict, str]],
) -> list[tuple[dict, str]]:
    """VASP (DFT) before LAMMPS (MLFF) within each temperature panel."""
    _order = {"VASP": 0, "LAMMPS": 1}

    def _key(item: tuple[dict, str]) -> tuple[int, str]:
        R, lab = item
        eng = R.get("meta", {}).get("md_engine", "unknown")
        return (_order.get(eng, 99), lab)

    return sorted(series_pairs, key=_key)


def _panel_compare_dft_mlff(grp_results: list[dict]) -> bool:
    """True if this temperature panel has both VASP and LAMMPS (DFT vs MLFF overlay)."""
    engines = {R.get("meta", {}).get("md_engine") for R in grp_results}
    return "VASP" in engines and "LAMMPS" in engines


def _linestyle_for_overlay(R: dict, compare_dft_mlff: bool) -> str:
    """Solid for VASP (DFT), dotted for LAMMPS (MLFF); all solid if not comparing."""
    if not compare_dft_mlff:
        return "-"
    eng = R.get("meta", {}).get("md_engine", "")
    if eng == "LAMMPS":
        return ":"
    return "-"


def _plot_k_pair_pdf_figure(
    results_list: list[dict],
    labels: list[str],
    sim_dirs: list,
    pair_keys: list[str],
    *,
    maximize_window: bool,
) -> None:
    if not os.environ.get("DISPLAY"):
        raise SystemExit(
            "DISPLAY is not set — cannot open an interactive plot window. "
            "Use an X11 session or SSH with X11 forwarding (e.g. `ssh -X`). "
            "This script does not save PNG/PDF files."
        )
    core._configure_matplotlib_backend(save=None)
    import matplotlib.pyplot as plt

    def _line_palette(n: int):
        """Distinct colors (one per K-involved pair channel in this plot)."""
        if n <= 0:
            return []
        if n <= 10:
            return list(plt.cm.tab10.colors[:n])
        if n <= 20:
            return [plt.cm.tab20(i / 19.0) for i in np.linspace(0, 19, n)]
        return [plt.cm.turbo(i / max(n - 1, 1)) for i in range(n)]

    from matplotlib.ticker import AutoMinorLocator

    temp_groups = core.group_results_by_temperature(sim_dirs, results_list, labels)
    n_blocks = max(len(temp_groups), 1)
    n_pk = len(pair_keys)
    # One row of pair-panels per temperature; widen the window when there are many pairs.
    fig_w = float(np.clip(3.4 * max(n_pk, 1), 11.0, 30.0))
    fig_h = max(4.2, 2.85 * n_blocks + 0.9)
    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(
        "K-involved pair distribution functions g(r) — MD",
        fontsize=13,
        weight="bold",
        y=0.995,
    )

    subfigs = fig.subfigures(n_blocks, 1, hspace=core._SUBFIG_HSPACE_TEMP_BLOCKS)
    subfigs = np.atleast_1d(subfigs).ravel()

    for (group_title, series_pairs), sf in zip(temp_groups, subfigs):
        series_pairs = _sort_series_pairs_for_comparison(list(series_pairs))
        sf.suptitle(group_title, fontsize=10, weight="bold", y=1.02)
        grp_results = [R for R, _lab in series_pairs]
        grp_labels = [_lab for _R, _lab in series_pairs]
        multi_dir = len(grp_results) > 1
        compare_dm = _panel_compare_dft_mlff(grp_results)
        series_palette = _line_palette(len(grp_results))

        ax_row = sf.subplots(1, n_pk, sharey=True)
        axs = np.atleast_1d(ax_row).ravel()

        for ip, pk in enumerate(pair_keys):
            ax = axs[ip]
            for si, (R, lab) in enumerate(zip(grp_results, grp_labels)):
                if pk not in R["rdf"]:
                    continue
                r, g = R["rdf"][pk]
                ls = _linestyle_for_overlay(R, compare_dm)
                lbl = lab
                ax.plot(
                    r,
                    g,
                    color=series_palette[si % len(series_palette)],
                    ls=ls,
                    label=lbl,
                    lw=1.3,
                )
            ax.axhline(1, c="grey", lw=0.4, ls="--")
            ax.set_title(pk, fontsize=10, weight="bold")
            ax.set_xlabel("r (Å)")
            if ip == 0:
                ax.set_ylabel("g(r)")
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            if multi_dir:
                ax.legend(
                    fontsize=6.5,
                    loc="upper right",
                    framealpha=0.92,
                    fancybox=False,
                )

    core._finalize_md_figure(
        fig,
        None,
        maximize_window,
        tight_layout_rect=core._TIGHT_LAYOUT_RDF_MSD,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot the pair distribution function g(r) for K–element pairs "
            "(same directory resolution as RDF_MSD_evaluation.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "dirs",
        nargs="+",
        help=(
            "simulation or parent directories (VASP and/or LAMMPS). "
            "Pass two roots (e.g. vasp_tree mlff_tree) to overlay DFT vs MLFF per temperature."
        ),
    )
    ap.add_argument(
        "--labels",
        nargs="+",
        help="legend labels (default: auto from dir names)",
    )
    ap.add_argument(
        "--series-prefixes",
        nargs="+",
        default=None,
        metavar="PREFIX",
        help="one prefix per DIR argument; forces PREFIX/<T> (same as workflow script)",
    )
    ap.add_argument(
        "--elements",
        nargs="+",
        default=None,
        metavar="X",
        help="only K–X pairs for these X (e.g. O Mn). Default: all neighbors of K.",
    )
    ap.add_argument(
        "--no-kk",
        action="store_true",
        help="omit the K–K partial g(r) (default: K–K is plotted with other K–X channels)",
    )
    _eq = ap.add_mutually_exclusive_group()
    _eq.add_argument(
        "--skip",
        type=int,
        default=core.DEFAULT_EQUIL_SKIP_FRAMES,
        metavar="N",
        help="skip first N trajectory frames "
        f"(default: {core.DEFAULT_EQUIL_SKIP_FRAMES}; incompatible with "
        "--all-frames/--skip-equil-15k)",
    )
    _eq.add_argument(
        "--all-frames",
        action="store_true",
        help="use the full trajectory (no equilibration skip)",
    )
    _eq.add_argument(
        "--skip-equil-15k",
        action="store_true",
        help=f"skip first {core.EQUIL_SKIP_15K_FRAMES:,} trajectory frames "
        "(incompatible with --skip/--all-frames)",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="max frames to read per directory",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=None,
        help="override timestep in fs (default: auto from INCAR/LAMMPS input)",
    )
    ap.add_argument(
        "--rdf-stride",
        type=int,
        default=10,
        metavar="N",
        help="frame stride when averaging g(r) (default: 10)",
    )
    ap.add_argument(
        "--rdf-rmax",
        type=float,
        default=6.0,
        metavar="R",
        help="g(r) cutoff radius in Å (default: 6.0)",
    )
    ap.add_argument(
        "--rdf-bins",
        type=int,
        default=300,
        metavar="N",
        help="number of histogram bins for g(r) (default: 300)",
    )
    ap.add_argument(
        "--no-maximize",
        action="store_true",
        help="do not maximize the plot window",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="skip matplotlib (exit after analysis prints)",
    )
    args = ap.parse_args()
    skip_frames = core.resolve_equil_skip_frames(
        args.all_frames,
        args.skip_equil_15k,
        args.skip,
    )

    sim_dirs, origin_indices = core.resolve_dirs(args.dirs)
    print(f"Resolved {len(sim_dirs)} simulation(s):")
    for sd in sim_dirs:
        print(f"  {sd}")

    all_res: list[dict] = []
    for d in sim_dirs:
        res = core.analyze_one(
            str(d),
            skip_frames,
            args.max_frames,
            args.rdf_stride,
            5000,
            args.rdf_rmax,
            args.rdf_bins,
            dt_override=args.dt,
            use_precomputed_msd=False,
            compute_psd=False,
        )
        all_res.append(res)

    labels = core.finalize_plot_labels(
        sim_dirs,
        args.labels,
        all_res,
        origin_indices=origin_indices,
        n_raw_dirs=len(args.dirs),
        series_prefixes=args.series_prefixes,
    )
    print("Plot legend labels:")
    for lb, sd in zip(labels, sim_dirs):
        print(f"  {lb:24s} ← {sd}")

    union_keys: set[str] = set()
    for R in all_res:
        union_keys.update(R["rdf"].keys())
    pair_keys = select_k_pairs(
        union_keys,
        elements=args.elements,
        include_kk=not args.no_kk,
    )
    if not pair_keys:
        need = "K in the trajectory"
        if args.elements:
            need += f" and K–({', '.join(args.elements)}) pairs present in g(r)"
        raise SystemExit(
            f"No K-involved partial g(r) channels to plot ({need}). "
            f"Available pair keys: {sorted(k for k in union_keys if k != 'total')}"
        )
    print("K-involved partial g(r) channels:", ", ".join(pair_keys))

    if args.no_plot:
        return

    _plot_k_pair_pdf_figure(
        all_res,
        labels,
        sim_dirs,
        pair_keys,
        maximize_window=not args.no_maximize,
    )


if __name__ == "__main__":
    main()
