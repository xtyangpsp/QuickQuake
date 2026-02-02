#!/usr/bin/env python3
"""
QuickQuake: run pipeline in time chunks (Config -> Stations -> Download -> PhaseNet -> GaMMA),
then run a separate merge script, then (optionally) run HypoXPy relocation.

This file is an ORCHESTRATOR only: it should not contain merge/location logic.
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import sys

# =================================================================
# CONFIGURATION
# =================================================================

START = "2021-09-19T15:00:00"
END   = "2021-09-19T17:00:00"
HOUR_STEP = 1

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"

CENTER   = (-161.8903, 55.4133)
DEG      = 1.0
NETWORKS = ["AV"]
CHANNELS = "BHZ,BHN,BHE,SHZ,SHN,SHE"
CLIENT   = "IRIS"
REGION   = "pavlof"

SCRIPTS = {
    "config":   BASE_DIR / "quickquake/QQ_config.py",
    "stations": BASE_DIR / "quickquake/QQ_dl_stations.py",
    "download": BASE_DIR / "quickquake/QQ_dl_data.py",
    "phasenet": BASE_DIR / "quickquake/QQ_predict.py",
    "gamma":    BASE_DIR / "quickquake/QQ_gamma.py",

    # NUEVOS: llamados como subprocess
    "merge":    BASE_DIR / "quickquake/QQ_merge_gamma_outputs.py",
    "location": BASE_DIR / "quickquake/QQ_location_hypoxpy.py",
}

RUN_CONFIG   = False
RUN_DL       = False
RUN_PHASENET = False
RUN_GAMMA    = False

RUN_MERGE_GAMMA = False

RUN_LOCATION     = True
LOCATION_BINPATH = "/home/elizabeth/bin"  # ajusta si cambia
LOCATION_NAMEBASE = "GAMMA"
LOCATION_EXTRA_ARGS = []  # ej: ["--cleanup"]


# =================================================================
# HELPERS
# =================================================================

def run_step(cmd, step, cwd=None):
    print("\n" + "=" * 70)
    print(f"Running: {step}")
    print("CMD:", " ".join(str(x) for x in cmd))
    print("=" * 70 + "\n")
    subprocess.run(cmd, check=True, cwd=cwd)


def process_window(start, end, out_dir: Path):
    """
    Runs ONLY until GaMMA inside each chunk folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config   = out_dir / "config.json"
    stations = out_dir / "stations.json"
    picks    = out_dir / "picks.csv"

    steps = [
        (RUN_CONFIG, [
            sys.executable, str(SCRIPTS["config"]),
            "--start", start.isoformat(),
            "--end", end.isoformat(),
            "--output", str(config),
            "--center=" + f"{CENTER[0]},{CENTER[1]}",
            "--deg", str(DEG),
            "--networks", ",".join(NETWORKS),
            "--channels", CHANNELS,
            "--client", CLIENT,
            "--region", REGION
        ], "Config"),

        (RUN_DL, [
            sys.executable, str(SCRIPTS["stations"]),
            "--config", str(config),
            "--output_dir", str(out_dir),
            "--plot"
        ], "Stations"),

        (RUN_DL, [
            sys.executable, str(SCRIPTS["download"]),
            "--config", str(config),
            "--output_dir", str(out_dir)
        ], "Data Download"),

        (RUN_PHASENET, [
            sys.executable, str(SCRIPTS["phasenet"]),
            "--model", str(MODEL_DIR),
            "--data_dir", str(out_dir / "waveforms"),
            "--data_list", str(out_dir / "input_data.csv"),
            "--stations", str(stations),
            "--result_dir", str(out_dir),
            "--format", "mseed_array",
            "--amplitude"
        ], "PhaseNet"),

        (RUN_GAMMA, [
            sys.executable, str(SCRIPTS["gamma"]),
            "--config", str(config),
            "--picks", str(picks),
            "--stations", str(stations),
            "--output_dir", str(out_dir)
        ], "GaMMA"),
    ]

    for enabled, cmd, name in steps:
        if enabled:
            run_step(cmd, name, cwd=BASE_DIR)


# =================================================================
# MAIN
# =================================================================

def main():
    current  = datetime.fromisoformat(START)
    end_time = datetime.fromisoformat(END)

    # 1) Run chunks
    while current < end_time:
        window_end = min(current + timedelta(hours=HOUR_STEP), end_time)
        date_str   = current.strftime("%Y%m%dT%H%M%S")
        out_dir    = DATA_ROOT / date_str

        print(f"\n{'='*70}\nProcessing chunk: {current} -> {window_end}\nOutput: {out_dir}\n{'='*70}")
        process_window(current, window_end, out_dir)
        current = window_end

    # 2) Merge (as subprocess)
    if RUN_MERGE_GAMMA:
        cmd_merge = [
            sys.executable, str(SCRIPTS["merge"]),
            "--data_root", str(DATA_ROOT),
            "--namebase", str(LOCATION_NAMEBASE),
        ]
        run_step(cmd_merge, "Merge GaMMA outputs", cwd=BASE_DIR)

    # 3) Location (as subprocess)
    if RUN_LOCATION:
        cmd_loc = [
            sys.executable, str(SCRIPTS["location"]),
            "--binpath", str(LOCATION_BINPATH),
            "--namebase", str(LOCATION_NAMEBASE),
        ] + list(LOCATION_EXTRA_ARGS)
        run_step(cmd_loc, "HypoXPy relocation (HypoInverse + HypoDD)", cwd=BASE_DIR)


if __name__ == "__main__":
    main()
