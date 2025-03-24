#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# =================================================================
# CONFIGURATION
# =================================================================

# Time settings
START = "2021-09-25T00:00:00"
END = "2021-09-25T06:00:00"
HOUR_STEP = 2

# Path setup
BASE_DIR = Path(__file__).parent.parent
DATA_ROOT = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"
VMODELS = BASE_DIR / "quickquake/vmodels"
HYPO_BIN = BASE_DIR / "dependencies/hyp1.40/src/hyp1.40"

SCRIPTS = {
    "config": BASE_DIR / "quickquake/QQ_config.py",
    "stations": BASE_DIR / "quickquake/QQ_dl_stations.py",
    "download": BASE_DIR / "quickquake/QQ_dl_data.py",
    "phasenet": BASE_DIR / "quickquake/QQ_predict.py",
    "gamma": BASE_DIR / "quickquake/QQ_gamma.py",
    "location": BASE_DIR / "quickquake/QQ_location.py"
}

RUN_CONFIG = True
RUN_DL = True
RUN_PHASENET = True
RUN_GAMMA = True
RUN_LOC = True
MERGE = True

# =================================================================
# PROCESSING FUNCTIONS
# =================================================================

def run_step(cmd, step):
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{step} error: {e}")
        return False

def process_window(start, end, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    config = out_dir / "config.json"
    stations = out_dir / "stations.json"
    picks = out_dir / "picks.csv"
    
    # Workflow steps
    steps = [
        (RUN_CONFIG, [sys.executable, str(SCRIPTS["config"]),
                      "--start", start.isoformat(),
                      "--end", end.isoformat(),
                      "--output", str(config)], "Config"),
        
        (RUN_DL, [sys.executable, str(SCRIPTS["stations"]),
                  "--config", str(config), "--output_dir", str(out_dir), "--plot"], "Stations"),
        
        (RUN_DL, [sys.executable, str(SCRIPTS["download"]),
                  "--config", str(config), "--output_dir", str(out_dir)], "Data Download"),
        
        (RUN_PHASENET, [sys.executable, str(SCRIPTS["phasenet"]),
                        "--model", str(MODEL_DIR), "--data_dir", str(out_dir/"waveforms"),
                        "--data_list", str(out_dir/"input_data.csv"), "--stations", str(stations),
                        "--result_dir", str(out_dir), "--format", "mseed_array", "--amplitude"], "PhaseNet"),
        
        (RUN_GAMMA, [sys.executable, str(SCRIPTS["gamma"]),
                     "--config", str(config), "--picks", str(picks),
                     "--stations", str(stations), "--output_dir", str(out_dir)], "Gamma"),
        
        (RUN_LOC, [sys.executable, str(SCRIPTS["location"]),
                   "--date_dir", str(out_dir), "--vmodel_dir", str(VMODELS),
                   "--hypo_bin", str(HYPO_BIN)], "Location")
    ]
    
    for condition, cmd, name in steps:
        if condition:
            print(f"Running: {name}")
            if not run_step(cmd, name): return False
    return True

def merge_results():
    files = list(DATA_ROOT.glob("**/output/hyp_good.csv"))
    if not files: return
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, header=None).iloc[:, :5]
            df.columns = ["time", "lat", "lon", "depth", "mag"]
            dfs.append(df)
            print(f"Merging: {f}")
        except Exception as e:
            print(f"Skipped {f}: {e}")
    
    if dfs:
        pd.concat(dfs).to_csv(DATA_ROOT/"consolidated_hyp_good.csv", index=False)
        print("\nMerged catalog saved")

# =================================================================
# MAIN EXECUTION
# =================================================================

def main():
    current = datetime.fromisoformat(START)
    end_time = datetime.fromisoformat(END)
    
    while current < end_time:
        window_end = min(current + timedelta(hours=HOUR_STEP), end_time)
        date_str = current.strftime("%Y%m%dT%H%M%S")
        output_dir = DATA_ROOT / date_str
        
        print(f"\n{'='*50}\nProcessing: {current} - {window_end}\n{'='*50}")
        if process_window(current, window_end, output_dir):
            print(f"Completed: {output_dir}")
        current = window_end
    
    if MERGE: merge_results()

if __name__ == "__main__":
    main()
