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
import traceback

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

RUN_ARPICK = True                 # Enable ar_pick repicking step

arpick_namebase = "GAMMA"         # Base name used in pipeline files
arpick_pre_p = 1.0                # Seconds before original P pick in local window
arpick_post_p = 2.0               # Seconds after original P pick in local window
arpick_pre_s = 1.5                # Seconds before original S pick in local window
arpick_post_s = 3.0               # Seconds after original S pick in local window
arpick_max_dt_p = 0.13            # Max allowed P repick shift from original pick
arpick_max_dt_s = 0.30            # Max allowed S repick shift from original pick
arpick_freqmin = 1.0              # Bandpass low cutoff before ar_pick
arpick_freqmax = 20.0             # Bandpass high cutoff before ar_pick

arpick_lta_p = 1.0                # Long-term window for P trigger
arpick_sta_p = 0.1                # Short-term window for P trigger
arpick_lta_s = 2.0                # Long-term window for S trigger
arpick_sta_s = 0.2                # Short-term window for S trigger
arpick_m_p = 2                    # AR model order for P
arpick_m_s = 8                    # AR model order for S
arpick_l_p = 0.1                  # P picker smoothing/control parameter
arpick_l_s = 0.2                  # S picker smoothing/control parameter

arpick_keep_debug_cols = True     # Save debug columns in output CSV

RUN_LOCATION      =  True
location_binpath = "/home/elizabeth/bin"
location_namebase = "GAMMA"
location_extra_args = []  # ej: ["--cleanup"]
location_p_model = "velo_p_rv_avo.cre"
location_s_model = "velo_s_rv_avo.cre"
location_ref_ele = 3.2 #reference location highest part in the topography 
location_depth_min = 0.0
location_depth_max = 20.0
location_depth_step = 0.5



RUN_QC= True
# qc options (user edits)
qc_min_total_stations = 3 #must be a positive number 
qc_min_valid_stations_per_v = None  # None => defaults to qc_min_total_stations
qc_vmin_curve   = 2.0
qc_vmax_curve   = 8.0
qc_vsteps_curve = 150
qc_winlen = 3
qc_noise_percentile  = 50.0
qc_signal_percentile = 90.0
qc_min_ratio = 2.0 
qc_energy_type = "squared_median" # other option is "envelope_median" or "squared_median"
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
    "arpick":      BASE_DIR / "quickquake/QQ_pick_arpick.py",
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
                print(f"[SKIP] GaMMA: does not exit {picks} (PhaseNet failed or there was no data).")
                continue
            try:
                run_step(cmd, name, cwd=BASE_DIR)
            except Exception as e:
                print("Error running"+name+": "+str(e))
                traceback.print_exc()
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
        # process_window(current, window_end, out_dir)
        # current = window_end
        try:
            process_window(current, window_end, out_dir)
        except Exception as e:
            print(f"[CHUNK FAILED] {current} -> {window_end}\n  {e}")
            traceback.print_exc()
        finally:
            current = window_end

    # 2) Merge (as subprocess)
    if RUN_MERGE_GAMMA:
        cmd_merge = [
            sys.executable, str(SCRIPTS["merge"]),
            "--data_root", str(DATA_ROOT),
            "--namebase", str(location_namebase),
        ]
        run_step(cmd_merge, "Merge GaMMA outputs", cwd=BASE_DIR)

   # 3) ar_pick refinement (as subprocess)
    if RUN_ARPICK:
        cmd_arpick = [
            sys.executable, str(SCRIPTS["arpick"]),
            "--data_root", str(DATA_ROOT),
            "--namebase", str(arpick_namebase),

            "--pre_p", str(arpick_pre_p),
            "--post_p", str(arpick_post_p),
            "--pre_s", str(arpick_pre_s),
            "--post_s", str(arpick_post_s),

            "--max_dt_p", str(arpick_max_dt_p),
            "--max_dt_s", str(arpick_max_dt_s),

            "--freqmin", str(arpick_freqmin),
            "--freqmax", str(arpick_freqmax),

            "--lta_p", str(arpick_lta_p),
            "--sta_p", str(arpick_sta_p),
            "--lta_s", str(arpick_lta_s),
            "--sta_s", str(arpick_sta_s),

            "--m_p", str(arpick_m_p),
            "--m_s", str(arpick_m_s),
            "--l_p", str(arpick_l_p),
            "--l_s", str(arpick_l_s),
        ]

        if arpick_keep_debug_cols:
            cmd_arpick.append("--keep_debug_cols")

        run_step(cmd_arpick, "ar_pick refinement", cwd=BASE_DIR)   

    # 3) Location (as subprocess)
    if RUN_ARPICK:
        picks_file = DATA_ROOT / "merged" / "gammapicks_id_arpick.csv"
    else:
        picks_file = DATA_ROOT / "merged" / "gammapicks_id.csv"
    if RUN_LOCATION:
        cmd_loc = [
        sys.executable, str(SCRIPTS["location"]),
        "--merged_dir", str(DATA_ROOT / "merged"),
        "--templates_dir", str(BASE_DIR / "hypox_templates"),
        "--binpath", str(location_binpath),
        "--namebase", str(location_namebase),
         "--picks_file", str(picks_file),
        "--p_model", str(location_p_model),
        "--s_model", str(location_s_model),
        "--ref_ele", str(location_ref_ele),

        "--depth_min", str(location_depth_min),
        "--depth_max", str(location_depth_max),
        "--depth_step", str(location_depth_step),
    ] + list(location_extra_args)
        run_step(cmd_loc, "HypoXPy relocation (HypoInverse + HypoDD)", cwd=BASE_DIR)

    


    # 4) QC (as subprocess)
    if RUN_QC:
        cmd_qc = [
            sys.executable, str(SCRIPTS["qc_velocity"]),
            "--data_root", str(DATA_ROOT),
            "--namebase", str(location_namebase),

            "--vmin_curve", str(qc_vmin_curve),
            "--vmax_curve", str(qc_vmax_curve),
            "--vsteps_curve", str(qc_vsteps_curve),

            "--winlen", str(qc_winlen),
            "--energy_type", str(qc_energy_type),
            "--min_total_stations", str(qc_min_total_stations),

            "--noise_percentile", str(qc_noise_percentile),
            "--signal_percentile", str(qc_signal_percentile),
            "--min_ratio", str(qc_min_ratio),
            "--max_plots", str(qc_max_plots),   
        ]

        if qc_make_plot:
            cmd_qc.append("--make_plot")
        if qc_min_valid_stations_per_v is not None:
            cmd_qc += ["--min_valid_stations_per_v", str(qc_min_valid_stations_per_v)]    

        run_step(cmd_qc, "QC: velocity percentile ratio filter", cwd=BASE_DIR)







if __name__ == "__main__":
    main()
