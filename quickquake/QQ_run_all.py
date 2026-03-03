#!/usr/bin/env python3
"""
QuickQuake: run pipeline in time chunks Config > Stations > Download > PhaseNet > GaMMA,
then run a separate merge script, then optionally run HypoXPy relocation, quality control and classification of volcanic signals 

This file is an ORCHESTRATOR only
"""

import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# 
# CONFIGURATION
# 

START = "2021-07-29T08:00:00"
END   = "2021-07-30T00:00:00"
HOUR_STEP = 8

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"


RUN_CONFIG   =  False

# config options (user edits )
center   = (-161.8903, 55.4133) # central cooordinates for your region of interest 
deg      = 1.0
networks = ["AV"]
channels = "BHZ,BHN,BHE,SHZ,SHN,SHE"
client   = "IRIS"
region   = "pavlof"

RUN_DL       = False

RUN_PHASENET = False
# phasenet options (user edits )
phasenet_min_p_prob = 0.30
phasenet_min_s_prob = 0.30
phasenet_mpd        = 50  # minimum peak distance

RUN_GAMMA    = False
# gamma options (user edits )
gamma_method           = "BGMM"   # default: BGMM
gamma_oversample_factor = 30       # default: 30 for BGMM
gamma_min_picks_per_eq = 6        # default: 6
gamma_max_sigma11      = 2      # S
gamma_max_sigma22      = 1     #m/s
gamma_max_sigma12      = 1     #covariance

RUN_MERGE_GAMMA = True
RUN_LOCATION      = True
LOCATION_BINPATH  = "/home/elizabeth/bin"
LOCATION_NAMEBASE = "GAMMA"
LOCATION_EXTRA_ARGS = []  # ej: ["--cleanup"]

RUN_QC= True
# qc options (user edits)
qc_vmin_curve   = 2.0
qc_vmax_curve   = 8.0
qc_vsteps_curve = 150
qc_winlen = 1
qc_noise_percentile  = 50.0
qc_signal_percentile = 90.0
qc_min_ratio = 2.0 
qc_make_plot = False
qc_max_plots = 50 #None change 

# SCRIPTS 

SCRIPTS = {
    "config":      BASE_DIR / "quickquake/QQ_config.py",
    "stations":    BASE_DIR / "quickquake/QQ_dl_stations.py",
    "download":    BASE_DIR / "quickquake/QQ_dl_data.py",
    "phasenet":    BASE_DIR / "quickquake/QQ_predict.py",
    "gamma":       BASE_DIR / "quickquake/QQ_gamma.py",
    "merge":       BASE_DIR / "quickquake/QQ_merge_gamma_outputs.py",
    "location":    BASE_DIR / "quickquake/QQ_location.py",
    "qc_velocity": BASE_DIR / "quickquake/QQ_qc.py",
}

# HELPERS
#

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
            "--center=" + f"{center[0]},{center[1]}",
            "--deg", str(deg),
            "--networks", ",".join(networks),
            "--channels", channels,
            "--client", client,
            "--region", region
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
            "--model_dir", str(MODEL_DIR),  # predict.py expects --model_dir
            "--data_dir", str(out_dir / "waveforms"),
            "--data_list", str(out_dir / "input_data.csv"),
            "--stations", str(stations),
            "--result_dir", str(out_dir),
            "--format", "mseed_array",
            "--amplitude",
            "--min_p_prob", str(phasenet_min_p_prob),
            "--min_s_prob", str(phasenet_min_s_prob),
            "--mpd", str(phasenet_mpd),
        ], "PhaseNet"),

        (RUN_GAMMA, [
            sys.executable, str(SCRIPTS["gamma"]),
            "--config", str(config),
            "--picks", str(picks),
            "--stations", str(stations),
            "--output_dir", str(out_dir),

            # user-facing gamma args
            "--method", str(gamma_method),
            "--oversample_factor", str(gamma_oversample_factor),
            "--min_picks_per_eq", str(gamma_min_picks_per_eq),
            "--max_sigma11", str(gamma_max_sigma11),
            "--max_sigma22", str(gamma_max_sigma22),
            "--max_sigma12", str(gamma_max_sigma12),
        ], "GaMMA"),
    ]

    for enabled, cmd, name in steps:
        if enabled:
            if name == "GaMMA" and not picks.exists():
                raise FileNotFoundError(
                    f"GaMMA requires picks.csv but it was not found:\n  {picks}\n"
                    f"Did you run PhaseNet for this chunk (RUN_PHASENET=True)?"
                )
            try:
                run_step(cmd, name, cwd=BASE_DIR)
            except Exception as e:
                print("Error running"+name+": "+str(e))
                continue


#
# MAIN
#

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
            "--dep_corr", "0",
        ] + list(LOCATION_EXTRA_ARGS)
        run_step(cmd_loc, "HypoXPy relocation (HypoInverse + HypoDD)", cwd=BASE_DIR)




    # 4) QC (as subprocess)
    if RUN_QC:
        cmd_qc = [
            sys.executable, str(SCRIPTS["qc_velocity"]),
            "--data_root", str(DATA_ROOT),
            "--namebase", str(LOCATION_NAMEBASE),

            "--vmin_curve", str(qc_vmin_curve),
            "--vmax_curve", str(qc_vmax_curve),
            "--vsteps_curve", str(qc_vsteps_curve),

            "--winlen", str(qc_winlen),

            "--noise_percentile", str(qc_noise_percentile),
            "--signal_percentile", str(qc_signal_percentile),
            "--min_ratio", str(qc_min_ratio),
            "--max_plots", str(qc_max_plots),   
        ]

        if qc_make_plot:
            cmd_qc.append("--make_plot")

        run_step(cmd_qc, "QC: velocity percentile ratio filter", cwd=BASE_DIR)







if __name__ == "__main__":
    main()
