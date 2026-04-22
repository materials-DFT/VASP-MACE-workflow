#!/usr/bin/env python3
"""Plot RDF/MSD for DFT (VASP) runs with explicit structure labels.

This script is tailored for VASP data and labels each curve using the exact
structure path relative to the provided input root(s), e.g.:
  1_layer/300K, 2_layers/700K, 3_layers/300K
"""

import argparse
import sys
from pathlib import Path
from collections import OrderedDict
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import VACF_RDF_MSD_evaluation as core


def _build_structure_labels(sim_dirs, raw_dirs):
    """Return stable, explicit labels for each simulation directory."""
    roots = [Path(d).resolve() for d in raw_dirs]
    labels = []
    for sd in sim_dirs:
        p = Path(sd).resolve()
        best_rel = None
        for r in roots:
            try:
                rel = p.relative_to(r)
            except ValueError:
                continue
            rel_s = str(rel)
            if best_rel is None or len(rel_s) < len(best_rel):
                best_rel = rel_s
        labels.append(best_rel if best_rel is not None else str(p))
    return labels


def _group_indices_by_structure(sim_dirs):
    """Group resolved simulation indices by structure folder (e.g. 1_layer)."""
    groups = OrderedDict()
    for i, sd in enumerate(sim_dirs):
        p = Path(sd).resolve()
        # Expected leaf layout: <structure>/<temperature> (e.g. 2_layers/700K)
        key = p.parent.name if p.parent != p else p.name
        groups.setdefault(key, []).append(i)
    return groups


def _group_save_path(save, group_key, n_groups):
    """Derive per-group save filename when multiple structures are plotted."""
    if save is None:
        return None
    if n_groups <= 1:
        return save
    out = Path(save)
    stem = out.stem
    suffix = out.suffix if out.suffix else ".png"
    safe_group = group_key.replace("/", "_")
    return str(out.with_name(f"{stem}_{safe_group}{suffix}"))


def _make_grouped_rdf_msd_figure(
    grouped_payloads,
    rdf_pairs=None,
    msd_tmax=None,
    save=None,
    maximize_window=True,
):
    """Single window with one RDF/MSD row per structure group."""
    core._configure_matplotlib_backend(save=save)
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors
    ls_cycle = ["-", "--", "-.", ":"]

    n_blocks = len(grouped_payloads)
    fig_h = max(5.8, 2.9 * n_blocks + 0.8)
    fig = plt.figure(figsize=(13.5, fig_h))
    fig.suptitle("DFT structure comparison: RDF & MSD", fontsize=13, weight="bold", y=0.995)

    subfigs = fig.subfigures(n_blocks, 1, hspace=core._SUBFIG_HSPACE_TEMP_BLOCKS)
    subfigs = np.atleast_1d(subfigs).ravel()

    for sf, (group_key, g_dirs, g_res, g_labels) in zip(subfigs, grouped_payloads):
        sf.suptitle(group_key, fontsize=10, weight="bold", y=1.01)
        ax_rdf, ax_msd = sf.subplots(1, 2)
        core._plot_rdf_msd_axes(
            ax_rdf,
            ax_msd,
            g_res,
            g_labels,
            rdf_pairs,
            msd_tmax,
            colors,
            ls_cycle,
        )

    core._finalize_md_figure(
        fig,
        save,
        maximize_window,
        tight_layout_rect=core._TIGHT_LAYOUT_RDF_MSD,
    )


def main():
    ap = argparse.ArgumentParser(
        description="RDF/MSD for DFT (VASP) trajectories with structure-explicit labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/palmetto_calculations/npt_dataset/oms6/md/neutral/1k/
  %(prog)s 1_layer/ 2_layers/ --all-frames --save dft_rdf_msd.png
  %(prog)s 1k/ --rdf-pairs total Mn-O --no-plot --data-log dft_rdf_msd.dat
""",
    )
    ap.add_argument("dirs", nargs="+", help="DFT simulation or parent directories")
    ap.add_argument(
        "--labels",
        nargs="+",
        help="optional manual labels (must match number of resolved runs)",
    )
    ap.add_argument(
        "--skip",
        type=int,
        default=core.DEFAULT_EQUIL_SKIP_FRAMES,
        metavar="N",
        help=f"skip first N frames (default: {core.DEFAULT_EQUIL_SKIP_FRAMES})",
    )
    ap.add_argument(
        "--all-frames",
        action="store_true",
        help="use all frames (same as --skip 0)",
    )
    ap.add_argument("--max-frames", type=int, default=None, help="max frames per run")
    ap.add_argument("--dt", type=float, default=None, help="override timestep in fs")
    ap.add_argument("--rdf-stride", type=int, default=10, help="RDF frame stride")
    ap.add_argument("--rdf-rmax", type=float, default=6.0, help="RDF cutoff in Angstrom")
    ap.add_argument("--rdf-bins", type=int, default=300, help="RDF bins")
    ap.add_argument("--rdf-pairs", nargs="+", default=None, help="RDF pairs (default: total)")
    ap.add_argument("--msd-tmax", type=float, default=None, help="max MSD lag time (ps)")
    ap.add_argument(
        "--precomputed-msd",
        action="store_true",
        help="use msd_back_all.dat when available",
    )
    ap.add_argument("--save", type=str, default=None, help="save figure path")
    ap.add_argument("--data-log", type=str, default=None, help="write numeric data log")
    ap.add_argument("--no-plot", action="store_true", help="run analysis without plotting")
    ap.add_argument("--no-maximize", action="store_true", help="do not maximize GUI window")
    args = ap.parse_args()

    skip_frames = 0 if args.all_frames else args.skip
    sim_dirs, _ = core.resolve_dirs(args.dirs)

    print(f"Resolved {len(sim_dirs)} simulation(s):")
    for sd in sim_dirs:
        print(f"  {sd}")

    all_res = []
    kept_dirs = []
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
            use_precomputed_msd=args.precomputed_msd,
            compute_psd=False,
        )
        if res["meta"].get("md_engine") != "VASP":
            print(f"  Skipping non-DFT run: {d}", flush=True)
            continue
        kept_dirs.append(d)
        all_res.append(res)

    if not all_res:
        raise RuntimeError("No DFT/VASP runs found after filtering.")

    if args.labels:
        if len(args.labels) != len(kept_dirs):
            raise ValueError(
                f"--labels expects {len(kept_dirs)} values, got {len(args.labels)}"
            )
        labels = args.labels
    else:
        labels = _build_structure_labels(kept_dirs, args.dirs)

    print("Plot legend labels:")
    for lb, sd in zip(labels, kept_dirs):
        print(f"  {lb:30s} ← {sd}")

    if args.data_log:
        core.write_data_log(args.data_log, kept_dirs, labels, all_res)

    if args.no_plot:
        if args.save:
            print("Note: --no-plot ignores --save (no figure written).", flush=True)
        return

    groups = _group_indices_by_structure(kept_dirs)
    print("Structure groups:")
    for gk, idxs in groups.items():
        print(f"  {gk}: {len(idxs)} run(s)")

    grouped_payloads = []
    for gk, idxs in groups.items():
        g_dirs = [kept_dirs[i] for i in idxs]
        g_res = [all_res[i] for i in idxs]
        g_labels = [labels[i] for i in idxs]
        grouped_payloads.append((gk, g_dirs, g_res, g_labels))

    if args.save:
        print(f"  Plotting all structure groups in one figure -> {args.save}")
    else:
        print("  Plotting all structure groups in one X11 window")
    _make_grouped_rdf_msd_figure(
        grouped_payloads,
        rdf_pairs=args.rdf_pairs,
        msd_tmax=args.msd_tmax,
        save=args.save,
        maximize_window=not args.no_maximize,
    )


if __name__ == "__main__":
    main()

