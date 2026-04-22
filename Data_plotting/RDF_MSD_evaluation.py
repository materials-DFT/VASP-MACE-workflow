#!/usr/bin/env python3
"""Plot radial distribution function g(r) and mean square displacement only.

Skips VACF / power spectrum computation for faster runs. Trajectory analysis
otherwise matches VACF_RDF_MSD_evaluation.py (including the default
equilibration frame skip). RDF and MSD data are still written with --data-log
if requested.
"""

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import VACF_RDF_MSD_evaluation as core


def main():
    ap = argparse.ArgumentParser(
        description="RDF and MSD from MD trajectories (no velocity PSD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 300K/
  %(prog)s dft/300K mlff/300K --labels DFT MLFF
  %(prog)s run/ --rdf-pairs total Mn-O --save rdf_msd.png
  %(prog)s 300K/ --all-frames   # include all frames (default skips first {n})
""".format(n=core.DEFAULT_EQUIL_SKIP_FRAMES),
    )
    ap.add_argument(
        "dirs",
        nargs="+",
        help="simulation or parent directories to analyze",
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
        help="one legend prefix per DIR argument; forces PREFIX/<T> and "
        "overrides VASP→DFT / LAMMPS→MLFF detection",
    )
    ap.add_argument(
        "--skip",
        type=int,
        default=core.DEFAULT_EQUIL_SKIP_FRAMES,
        metavar="N",
        help="skip first N frames for equilibration "
        f"(default: {core.DEFAULT_EQUIL_SKIP_FRAMES}; use --all-frames for N=0)",
    )
    ap.add_argument(
        "--all-frames",
        action="store_true",
        help="use the full trajectory (no equilibration skip; same as --skip 0)",
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
        help="frame stride for RDF (default: 10)",
    )
    ap.add_argument(
        "--rdf-rmax",
        type=float,
        default=6.0,
        help="RDF cutoff in Å (default: 6.0)",
    )
    ap.add_argument(
        "--rdf-bins",
        type=int,
        default=300,
        help="number of RDF bins (default: 300)",
    )
    ap.add_argument(
        "--rdf-pairs",
        nargs="+",
        default=None,
        help="partial RDFs to plot, e.g. Mn-O K-O (default: total)",
    )
    ap.add_argument(
        "--msd-tmax",
        type=float,
        default=None,
        help="max lag time τ on MSD plot (ps; default: full range)",
    )
    ap.add_argument(
        "--precomputed-msd",
        action="store_true",
        help="use msd_back_all.dat when available (default: compute from trajectory)",
    )
    ap.add_argument(
        "--no-maximize",
        action="store_true",
        help="do not maximize the plot window to the screen (interactive only)",
    )
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="save figure to this path (png/pdf); use Agg when DISPLAY is unset",
    )
    ap.add_argument(
        "--data-log",
        type=str,
        default=None,
        metavar="FILE",
        help="write RDF, MSD, and metadata to FILE (tab-separated; no PSD block)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="skip matplotlib; use with --data-log for analysis only",
    )
    args = ap.parse_args()
    skip_frames = 0 if args.all_frames else args.skip

    sim_dirs, origin_indices = core.resolve_dirs(args.dirs)

    print(f"Resolved {len(sim_dirs)} simulation(s):")
    for sd in sim_dirs:
        print(f"  {sd}")

    all_res = []
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

    if args.data_log:
        core.write_data_log(args.data_log, sim_dirs, labels, all_res)

    if args.no_plot:
        if args.save:
            print("Note: --no-plot ignores --save (no figure written).", flush=True)
    else:
        core.make_rdf_msd_figure(
            all_res,
            labels,
            sim_dirs=sim_dirs,
            rdf_pairs=args.rdf_pairs,
            msd_tmax=args.msd_tmax,
            save=args.save,
            maximize_window=not args.no_maximize,
        )


if __name__ == "__main__":
    main()
