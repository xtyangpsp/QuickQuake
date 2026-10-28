#!/usr/bin/env python3
"""
QuickQuake - HypoXPy relocation (HypoInverse + HypoDD) ONCE using merged GaMMA outputs.

Expected files, written by the merge step (QQ_merge_gamma_outputs.py):
- <data_root>/merged/gammacatalog_id.csv
- <data_root>/merged/gammapicks_id.csv
- <data_root>/merged/input/GAMMA_station_list.json  (union of all chunks)

This script will:
- Create/ensure:
    <data_root>/merged/input/
    <data_root>/merged/output/
- Create symlinks inside input/ with the canonical names expected by HypoXPy:
    input/GAMMA_catalog.csv  -> ../gammacatalog_id.csv
    input/GAMMA_picks.csv    -> ../gammapicks_id.csv
- Link/copy velocity models + templates into input/ (models may be a name inside
  <repo>/hypox_templates or a full path to your own .cre file)
- Write the same velocity models into the HypoDD template copy, so HypoInverse
  and HypoDD locate with one model
- Run hypoxpy.workflow.relocate(), with HypoDD measuring depth from the top of
  the velocity model (ref_ele) so that events in the volcanic edifice, above sea
  level, are kept

Outputs (in <data_root>/merged/output/):
- <namebase>_hypoinv_good.csv
- <namebase>_hypodd_catalog.csv
"""

import os
import json
import argparse
import shutil
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import hypoxpy.utils as hypox_utils
from hypoxpy.workflow import relocate
from typing import Optional


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


def drop_location_codes(indir: Path, namebase: str):
    """
    HypoXPy reads pick ids as NET.STA.COMPONENT.CHANNEL, so a location code
    (DK.GDH.00.HH) ends up in the HypoInverse channel field ("HH00") and
    HypoInverse rejects the phase line ("SKIP BAD PHASE CARD").
    Give HypoXPy ids without the location code (DK.GDH..HH), as long as every
    station id stays unique. Only the copies in input/ change; the merged
    picks (gammapicks_id.csv) keep their location codes.
    """
    station_json = indir / f"{namebase}_station_list.json"
    picks_csv = indir / f"{namebase}_picks.csv"

    def strip(sid):
        parts = str(sid).split(".")
        return ".".join([parts[0], parts[1], "", parts[3]]) if len(parts) == 4 else sid

    with open(station_json) as fp:
        stations = json.load(fp)
    new_ids = [strip(k) for k in stations]
    if len(set(new_ids)) != len(new_ids):
        print("[location] warning: station ids differ only by location code; kept as they are")
        return
    if new_ids != list(stations):
        with open(station_json, "w") as fp:
            json.dump({strip(k): v for k, v in stations.items()}, fp, indent=2)

    picks = pd.read_csv(picks_csv)
    new_pick_ids = picks["id"].map(strip)
    if not new_pick_ids.equals(picks["id"]):
        picks_csv.unlink()   # input/ holds a link to the merged picks: replace it with a real copy
        picks["id"] = new_pick_ids
        picks.to_csv(picks_csv, index=False)
        print("[location] Station ids without location codes given to HypoXPy (e.g. DK.GDH..HH).")


def replace_missing_magnitudes(indir: Path, namebase: str):
    """
    Without amplitudes (gamma_use_amplitude = False) GaMMA writes magnitude 999.
    That value overflows HypoDD's magnitude field ("*****") and HypoDD stops with
    "Format error in earthquake file". Give HypoXPy 0.0 instead, the value HypoXPy
    already uses for HypoInverse. The merged GaMMA catalog keeps its 999.
    """
    catalog_csv = indir / f"{namebase}_catalog.csv"
    catalog = pd.read_csv(catalog_csv)
    if "magnitude" not in catalog.columns or not (catalog["magnitude"] >= 999).any():
        return
    catalog_csv.unlink()   # input/ holds a link to the merged catalog: replace it with a real copy
    catalog.loc[catalog["magnitude"] >= 999, "magnitude"] = 0.0
    catalog.to_csv(catalog_csv, index=False)
    print("[location] GaMMA magnitudes of 999 (no amplitudes) given to HypoXPy as 0.0.")


def remove_previous_results(outdir: Path, namebase: str):
    """
    Delete the HypoInverse/HypoDD results of an earlier run, so that a step that
    fails now stops the run instead of silently reusing the old files.
    """
    old = [outdir / f"{namebase}_hypoinv_good.csv", outdir / f"{namebase}_hypoinv_bad.csv",
           outdir / f"{namebase}_hypodd_catalog.csv"]
    old += list(outdir.glob(f"hypoDD_{namebase}.*"))
    for path in old:
        if path.is_file():
            path.unlink()


def read_cre_model(path: Path):
    """
    Read a HypoInverse CRE velocity model: a title line, then one line per layer
    with the velocity (km/s) and the depth of the layer top (km below the top of
    the model, which HypoInverse places at the reference elevation ref_ele).
    """
    with open(path) as fp:
        lines = fp.read().splitlines()[1:]   # first line is the title
    vel, top = [], []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            vel.append(float(parts[0]))
            top.append(float(parts[1]))
    if not vel:
        raise ValueError(f"No layers found in velocity model {path}")
    return vel, top


def hypodd_model_block(pmodel: Path, smodel: Path, ref_ele: float):
    """
    HypoDD forward model equal to the HypoInverse CRE models, so that location and
    relocation use the same velocities. IMOD = 1: layered Vp with one Vp/Vs ratio
    per layer, taken from the P and S models.

    Depth datum: the CRE layer tops are measured from the top of the model, which
    HypoInverse places at ref_ele km above sea level. HypoDD measures depth from
    the same place (see hypodd_depths_from_model_top), so the layer tops are the
    CRE tops unchanged.
    """
    vp, top_p = read_cre_model(pmodel)
    vs, top_s = read_cre_model(smodel)
    if top_p != top_s:
        raise ValueError(f"The P and S models must have the same layers for HypoDD "
                         f"(IMOD = 1): {pmodel.name} {top_p}, {smodel.name} {top_s}")
    if top_p[0] != 0.0:
        raise ValueError(f"The first layer of {pmodel.name} must start at depth 0, not {top_p[0]}")
    top = top_p
    ratio = [p / s for p, s in zip(vp, vs)]

    def row(values, ndigits):
        return " ".join(f"{round(v, ndigits):g}" for v in values)

    return [
        "*--- FORWARD MODEL SPECIFICATIONS:\n",
        f"* Written by QuickQuake (QQ_location.py) from the HypoInverse models {pmodel.name}\n",
        f"* and {smodel.name}. Depths (layer tops and events) are measured from the\n",
        f"* top of the model, {ref_ele} km above sea level (location_ref_ele).\n",
        "* IMOD\n",
        "1\n",
        "* TOP:\n",
        row(top, 3) + "\n",
        "* VELP:\n",
        row(vp, 3) + "\n",
        "* RAT:\n",
        row(ratio, 4) + "\n",
        "*\n",
    ]


def write_hypodd_template(template: Path, dst: Path, pmodel: Path, smodel: Path, ref_ele: float):
    """
    Copy the HypoDD template to dst with its forward-model section (from
    '*--- FORWARD MODEL' to '*--- CLUSTER/EVENT SELECTION') replaced by the
    HypoInverse velocity models.
    """
    with open(template) as fp:
        lines = fp.readlines()
    starts = [i for i, l in enumerate(lines) if l.startswith("*--- FORWARD MODEL")]
    ends = [i for i, l in enumerate(lines) if l.startswith("*--- CLUSTER/EVENT SELECTION")]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError(f"{template} needs one '*--- FORWARD MODEL SPECIFICATIONS:' line followed by "
                         "one '*--- CLUSTER/EVENT SELECTION:' line, where QuickQuake writes the velocity model")
    lines = lines[:starts[0]] + hypodd_model_block(pmodel, smodel, ref_ele) + lines[ends[0]:]

    # dst may be a link to the template of an earlier run: remove it, never write through it
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    with open(dst, "w") as fp:
        fp.writelines(lines)


@contextmanager
def hypodd_depths_from_model_top(ref_ele: float, indir: Path):
    """
    HypoDD 2.1 does not accept events above its depth zero: getdata.f moves any
    depth shallower than 0.01 km down to 0.01 km, and an event that moves above
    zero is an air quake. HypoInverse reports CRE depths below sea level, so events
    in the volcanic edifice (up to the model top, -ref_ele) are negative and HypoDD
    would push them down to sea level or drop them.

    HypoDD therefore measures depth from the model top: while relocate() writes the
    HypoDD phase file, it reads a copy of the event catalog with every depth
    + ref_ele (the HypoInverse catalog itself is not changed). HypoXPy's dep_corr,
    set to ref_ele, subtracts it again when it writes <namebase>_hypodd_catalog.csv,
    so that catalog stays below sea level like HypoInverse. hypoDD_<namebase>.reloc
    keeps HypoDD's own depths, from the model top.
    """
    original = hypox_utils.convert_gamma2hypodd

    def convert_from_model_top(eventfile, pickfile, *args, **kwargs):
        events = pd.read_csv(eventfile, sep=kwargs.get("separator", ","))
        label, scale = hypox_utils.resolve_event_depth_column(events)
        events[label] = events[label] + ref_ele / scale
        shifted = indir / f"{Path(eventfile).stem}_depth_from_model_top.csv"
        events.to_csv(shifted, index=False, sep=kwargs.get("separator", ","))
        return original(str(shifted), pickfile, *args, **kwargs)

    hypox_utils.convert_gamma2hypodd = convert_from_model_top
    try:
        yield
    finally:
        hypox_utils.convert_gamma2hypodd = original


def resolve_model(model: str, templates_dir: Path) -> Path:
    """
    A velocity model can be given as a file name inside templates_dir
    (e.g. velo_p_eg.cre) or as a path to the user's own .cre file.
    """
    candidate = Path(model).expanduser()
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return candidate.resolve()
    return templates_dir / model


# def stage_in_merged(merged_dir: Path, templates_dir: Path, namebase: str,
#                     p_model_name: str, s_model_name: str):

def stage_in_merged(merged_dir: Path, templates_dir: Path, namebase: str,
                    p_model_name: str, s_model_name: str, ref_ele: float,
                    picks_override: Optional[Path] = None):
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
    # cat_src = merged_dir / "gammacatalog_id.csv"
    # picks_src = merged_dir / "gammapicks_id.csv"
    cat_src = merged_dir / "gammacatalog_id.csv"
    picks_src = picks_override if picks_override is not None else (merged_dir / "gammapicks_id.csv")
    ensure_file(cat_src, "merged catalog (gammacatalog_id.csv)")
    ensure_file(picks_src, "merged picks (gammapicks_id.csv)")

    # station list is written into merged/input/ by the merge step
    station_src = indir / f"{namebase}_station_list.json"
    ensure_file(station_src, "station list JSON in merged/input")

    # Create canonical names expected by HypoXPy example workflow
    link_or_copy(cat_src, indir / f"{namebase}_catalog.csv")
    link_or_copy(picks_src, indir / f"{namebase}_picks.csv")

    # Velocity models + templates live in repo/hypox_templates
    pmodel = resolve_model(p_model_name, templates_dir)
    smodel = resolve_model(s_model_name, templates_dir)
    t_hypoinv = templates_dir / "template_hypoinv_vp-vs.txt"
    t_ph2dt = templates_dir / "template_ph2dt_par.inp"
    t_hypodd = templates_dir / "template_hypodd_par.inp"

    ensure_file(pmodel, f"P velocity model ({p_model_name})")
    ensure_file(smodel, f"S velocity model ({s_model_name})")
    ensure_file(t_hypoinv, "HypoInverse template (template_hypoinv_vp-vs.txt)")
    ensure_file(t_ph2dt, "ph2dt template (template_ph2dt_par.inp)")
    ensure_file(t_hypodd, "HypoDD template (template_hypodd_par.inp)")

    # Link/copy into input/ so paths are short
    link_or_copy(pmodel, indir / pmodel.name)
    link_or_copy(smodel, indir / smodel.name)
    link_or_copy(t_hypoinv, indir / t_hypoinv.name)
    link_or_copy(t_ph2dt, indir / t_ph2dt.name)
    # HypoDD gets the same velocity model as HypoInverse, written into its template copy
    write_hypodd_template(t_hypodd, indir / t_hypodd.name, pmodel, smodel, ref_ele)
    print(f"[location] HypoDD velocity model written from {pmodel.name} and {smodel.name} "
          f"(depths from the model top, ref_ele = {ref_ele} km above sea level)")

    return indir, outdir, pmodel.name, smodel.name


def resolve_combine_net_sta(mode: str, station_json: Path) -> bool:
    """
    ph2dt/HypoDD read only 7 characters of a station name.
      yes  -> NET.STA (original behavior)
      no   -> STA only
      auto -> NET.STA if every name fits in 7 characters,
              otherwise STA only, provided station codes are unique across networks
    """
    if mode == "yes":
        return True
    if mode == "no":
        return False

    with open(station_json) as fp:
        pairs = {tuple(key.split(".")[:2]) for key in json.load(fp)}

    if all(len(f"{net}.{sta}") <= 7 for net, sta in pairs):
        return True

    by_sta = defaultdict(list)
    for net, sta in pairs:
        by_sta[sta].append(f"{net}.{sta}")
    clashes = {sta: names for sta, names in by_sta.items() if len(names) > 1}
    if clashes:
        raise ValueError(
            "Some NET.STA names exceed 7 characters (HypoDD limit), and station codes are not "
            f"unique across networks, so they cannot be shortened safely: {clashes}"
        )

    print("[location] Some NET.STA names exceed 7 characters (HypoDD limit); "
          "using station codes only for HypoDD.")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binpath", required=True, help="Folder containing hyp1.40/hypoinverse + hypoDD + ph2dt binaries")
    ap.add_argument("--namebase", default="GAMMA")
    ap.add_argument("--p_model", default="velo_p_eg.cre",
                help="P-wave velocity model: a file name inside templates_dir, or a full path to your own .cre file")
    ap.add_argument("--s_model", default="velo_s_eg.cre",
                    help="S-wave velocity model: a file name inside templates_dir, or a full path to your own .cre file")
    ap.add_argument("--ref_ele", type=float, default=3.0,
                    help="Reference elevation (km above sea level): top of the velocity models, "
                         "usually the highest topography. HypoInverse and HypoDD measure depth from it")

    # sensible defaults for your repo layout
    ap.add_argument("--merged_dir", default=None, help="Default: <repo>/data/merged")
    ap.add_argument("--templates_dir", default=None, help="Default: <repo>/hypox_templates")

    ap.add_argument("--min_nsta", type=int, default=4,
                    help="Minimum stations for a HypoInverse location (default: 4)")
    ap.add_argument("--depth_min", type=float, default=0.0)
    ap.add_argument("--depth_max", type=float, default=20.0)
    ap.add_argument("--depth_step", type=float, default=1.0)
    ap.add_argument("--picks_file", default=None, help="Optional picks CSV to use instead of merged/gammapicks_id.csv")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--qc_phase", action="store_true")  # default False
    ap.add_argument("--skip_hypoinverse", action="store_true")
    ap.add_argument("--skip_hypodd", action="store_true")
    ap.add_argument("--combine_net_sta", choices=["yes", "no", "auto"], default="yes",
                    help="HypoDD station names: yes = NET.STA (default), no = STA only, "
                         "auto = STA only when NET.STA exceeds 7 characters")
    args = ap.parse_args()

    # repo root inferred from this file: <repo>/quickquake/QQ_location.py
    repo_root = Path(__file__).resolve().parent.parent

    merged_dir = Path(args.merged_dir) if args.merged_dir else (repo_root / "data" / "merged")
    templates_dir = Path(args.templates_dir) if args.templates_dir else (repo_root / "hypox_templates")

    picks_override = Path(args.picks_file).resolve() if args.picks_file else None

    indir, outdir, p_model_used, s_model_used = stage_in_merged(
        merged_dir=merged_dir,
        templates_dir=templates_dir,
        namebase=args.namebase,
        p_model_name=args.p_model,
        s_model_name=args.s_model,
        ref_ele=args.ref_ele,
        picks_override=picks_override
    )
    drop_location_codes(indir, args.namebase)
    replace_missing_magnitudes(indir, args.namebase)
    remove_previous_results(outdir, args.namebase)

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

    combine_net_sta = resolve_combine_net_sta(args.combine_net_sta, Path(station_file))

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
            "combine_net_sta": combine_net_sta,
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
            "p_model": os.path.join(indir_rel, p_model_used),
            "s_model": os.path.join(indir_rel, s_model_used),
            "ref_ele": float(args.ref_ele),
            "depth_list": depth_try_list,
            "min_nsta": args.min_nsta,
            "hypoinv_template": os.path.join(indir_rel, "template_hypoinv_vp-vs.txt"),
        },

        "hypodd": {
            "dep_corr": float(args.ref_ele),   # HypoDD depths are from the model top: see hypodd_depths_from_model_top
            "ph2dt_template": os.path.join(indir_rel, "template_ph2dt_par.inp"),
            "hypodd_template": os.path.join(indir_rel, "template_hypodd_par.inp"),
        },
    }

    with hypodd_depths_from_model_top(args.ref_ele, Path(indir_rel)):
        relocate(
            hypox_pars,
            input_type="gamma",
            skip_hypoinverse=args.skip_hypoinverse,
            skip_hypodd=args.skip_hypodd,
            allow_skip_hypoinverse=False,
            verbose=True,
        )

    print("\nRelocation completed.")
    print(f"   Working dir : {merged_dir}")
    print(f"   Input dir   : {indir}")
    print(f"   Output dir  : {outdir}")
    print(f"   HypoInv good: {merged_dir / out_hypoinv_good}")
    print(f"   HypoDD final: {merged_dir / out_hypodd_final}")


if __name__ == "__main__":
    main()
