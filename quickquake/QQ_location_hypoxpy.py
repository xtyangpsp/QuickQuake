#!/usr/bin/env python3
"""
QuickQuake - HypoXPy relocation (HypoInverse + HypoDD) ONCE using merged GaMMA outputs.

Expected filesystem after QQ_run_all_hypoxpy.py:
- <repo>/data/merged/gammacatalog_id.csv
- <repo>/data/merged/gammapicks_id.csv
- <repo>/data/merged/input/GAMMA_station_list.json  (join from all)

This script will:
- Create/ensure:
    <repo>/data/merged/input/
    <repo>/data/merged/output/
- Create symlinks inside input/ with the canonical names expected by HypoXPy:
    input/GAMMA_catalog.csv  -> ../gammacatalog_id.csv
    input/GAMMA_picks.csv    -> ../gammapicks_id.csv
- Link/copy velocity models + templates from <repo>/hypox_templates into input/
- Run hypoxpy.workflow.relocate()

Outputs (in data/merged/output/):
- <namebase>_hypoinv_good.csv
- <namebase>_hypodd_catalog.csv
"""

import os
import argparse
import shutil
from pathlib import Path

import numpy as np
from hypoxpy.workflow import relocate


# 
# helpers
# 
def link_or_copy(src: Path, dst: Path):
    """
    Prefer symlink to avoid duplication; fallback to copy (better for systems without symlink perms).
    Overwrites dst if it exists.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def ensure_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def stage_in_merged(merged_dir: Path, templates_dir: Path, namebase: str):
    """
    Work INSIDE merged_dir:
      merged_dir/input
      merged_dir/output

    Create canonical files in input/ expected by HypoXPy.
    """
    merged_dir = merged_dir.resolve()
    templates_dir = templates_dir.resolve()

    indir = merged_dir / "input"
    outdir = merged_dir / "output"
    indir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # merged CSVs produced by your merge step
    cat_src = merged_dir / "gammacatalog_id.csv"
    picks_src = merged_dir / "gammapicks_id.csv"
    ensure_file(cat_src, "merged catalog (gammacatalog_id.csv)")
    ensure_file(picks_src, "merged picks (gammapicks_id.csv)")

    # station list should have been copied by run_all into merged/input/
    station_src = indir / f"{namebase}_station_list.json"  # typically GAMMA_station_list.json
    ensure_file(station_src, "station list JSON in merged/input (GAMMA_station_list.json)")

    # Create canonical names expected by HypoXPy example workflow
    link_or_copy(cat_src, indir / f"{namebase}_catalog.csv")   # input/GAMMA_catalog.csv
    link_or_copy(picks_src, indir / f"{namebase}_picks.csv")   # input/GAMMA_picks.csv

    # Velocity models + templates live in repo/hypox_templates (your current layout)
    pmodel = templates_dir / "velo_p_eg.cre"
    smodel = templates_dir / "velo_s_eg.cre"
    t_hypoinv = templates_dir / "template_hypoinv_vp-vs.txt"
    t_ph2dt = templates_dir / "template_ph2dt_par.inp"
    t_hypodd = templates_dir / "template_hypodd_par.inp"

    ensure_file(pmodel, "P velocity model (velo_p_eg.cre)")
    ensure_file(smodel, "S velocity model (velo_s_eg.cre)")
    ensure_file(t_hypoinv, "HypoInverse template (template_hypoinv_vp-vs.txt)")
    ensure_file(t_ph2dt, "ph2dt template (template_ph2dt_par.inp)")
    ensure_file(t_hypodd, "HypoDD template (template_hypodd_par.inp)")

    # Link/copy into input/ so paths are short (Fortran-friendly)
    link_or_copy(pmodel, indir / pmodel.name)
    link_or_copy(smodel, indir / smodel.name)
    link_or_copy(t_hypoinv, indir / t_hypoinv.name)
    link_or_copy(t_ph2dt, indir / t_ph2dt.name)
    link_or_copy(t_hypodd, indir / t_hypodd.name)

    return indir, outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binpath", required=True, help="Folder containing hyp1.40/hypoinverse + hypoDD + ph2dt binaries")
    ap.add_argument("--namebase", default="GAMMA")

    # sensible defaults for your repo layout
    ap.add_argument("--merged_dir", default=None, help="Default: <repo>/data/merged")
    ap.add_argument("--templates_dir", default=None, help="Default: <repo>/hypox_templates")

    ap.add_argument("--min_nsta", type=int, default=4)
    ap.add_argument("--depth_min", type=float, default=0.0)
    ap.add_argument("--depth_max", type=float, default=20.0)
    ap.add_argument("--depth_step", type=float, default=1.0)

    ap.add_argument("--dep_corr", type=float, default=5.0)
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--qc_phase", action="store_true")  # default False
    ap.add_argument("--skip_hypoinverse", action="store_true")
    ap.add_argument("--skip_hypodd", action="store_true")
    args = ap.parse_args()

    # repo root inferred from this file: <repo>/quickquake/QQ_location_hypoxpy.py
    repo_root = Path(__file__).resolve().parent.parent

    merged_dir = Path(args.merged_dir) if args.merged_dir else (repo_root / "data" / "merged")
    templates_dir = Path(args.templates_dir) if args.templates_dir else (repo_root / "hypox_templates")

    indir, outdir = stage_in_merged(
        merged_dir=merged_dir,
        templates_dir=templates_dir,
        namebase=args.namebase,
    )

    # Work from merged_dir so we can use short relative paths like the original script
    os.chdir(merged_dir)

    depth_try_list = np.arange(args.depth_min, args.depth_max + 1e-9, args.depth_step)

    # Short relative paths (exactly the style of the original)
    indir_rel = "input"
    outdir_rel = "output"
    namebase = args.namebase

    station_file = os.path.join(indir_rel, f"{namebase}_station_list.json")         # input/GAMMA_station_list.json
    station_file_hypoinv = os.path.join(indir_rel, f"{namebase}_station_hypoinv.dat")
    station_file_hypodd = os.path.join(indir_rel, f"{namebase}_station_hypodd.dat")

    event_file = os.path.join(indir_rel, f"{namebase}_catalog.csv")                # input/GAMMA_catalog.csv
    phase_file = os.path.join(indir_rel, f"{namebase}_picks.csv")                  # input/GAMMA_picks.csv

    phase_hypoinv = os.path.join(indir_rel, f"{namebase}_phase_hypoinv.pha")
    phase_hypodd = os.path.join(indir_rel, f"{namebase}_phase_hypodd.pha")

    out_hypoinv_bad = os.path.join(outdir_rel, f"{namebase}_hypoinv_bad.csv")
    out_hypoinv_good = os.path.join(outdir_rel, f"{namebase}_hypoinv_good.csv")
    out_hypodd_final = os.path.join(outdir_rel, f"{namebase}_hypodd_catalog.csv")

    cleaned_eventfile = os.path.join(indir_rel, f"{namebase}_catalog_cleaned.csv")
    cleaned_pickfile = os.path.join(indir_rel, f"{namebase}_picks_cleaned.csv")

    hypox_pars = {
        "paths": {
            "binpath": args.binpath,
            "indir": indir_rel,
            "outdir": outdir_rel,
            "namebase": namebase,
        },

        "preprocess": {
            "save_cleaned_data": True,
            "cleaned_eventfile": cleaned_eventfile,
            "cleaned_pickfile": cleaned_pickfile,
            "combine_net_sta": True,
            "cleanup": args.cleanup,
            "qc_phase": args.qc_phase,
        },

        # Your event_id is not integer -> mapping must be True
        "event_id": {
            "evid_label": "event_id",
            "mapping_evid": True,
            "evid_label_mapped": "event_id_mapped",
        },

        "files": {
            "stations": {
                "json": station_file,
                "hypoinv": station_file_hypoinv,
                "hypodd": station_file_hypodd,
            },
            "events": event_file,
            "phases": {
                "raw": phase_file,
                "hypoinv": phase_hypoinv,
                "hypodd": phase_hypodd,
            },
            "final_catalogs": {
                "hypoinv_bad": out_hypoinv_bad,
                "hypoinv_good": out_hypoinv_good,
                "hypodd_final": out_hypodd_final,
            },
        },

        "hypoinverse": {
            "p_model": os.path.join(indir_rel, "velo_p_eg.cre"),
            "s_model": os.path.join(indir_rel, "velo_s_eg.cre"),
            "depth_list": depth_try_list,
            "min_nsta": args.min_nsta,
            "hypoinv_template": os.path.join(indir_rel, "template_hypoinv_vp-vs.txt"),
        },

        "hypodd": {
            "dep_corr": float(args.dep_corr),
            "ph2dt_template": os.path.join(indir_rel, "template_ph2dt_par.inp"),
            "hypodd_template": os.path.join(indir_rel, "template_hypodd_par.inp"),
        },
    }

    relocate(
        hypox_pars,
        input_type="gamma",
        skip_hypoinverse=args.skip_hypoinverse,
        skip_hypodd=args.skip_hypodd,
        allow_skip_hypoinverse=False,
        verbose=True,
    )

    print("\n Relocation terminado.")
    print(f"   Working dir : {merged_dir}")
    print(f"   Input dir   : {indir}")
    print(f"   Output dir  : {outdir}")
    print(f"   HypoInv good: {merged_dir / out_hypoinv_good}")
    print(f"   HypoDD final: {merged_dir / out_hypodd_final}")


if __name__ == "__main__":
    main()
