#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# =================================================================
# ADJUST THESE VALUES AS NEEDED
# =================================================================

# Time configuration
START_TIME_STR = "2021-09-25T00:00:00"
END_TIME_STR = "2021-09-25T06:00:00"
INCREMENT_HOURS = 2

# --- Relative paths configuration ---
# Base project directory (root of QuickQuake)
BASE_DIR = Path(__file__).parent.parent  # Adjust according to the actual location of run_all.py

# Key directories
DATA_ROOT = BASE_DIR / "data"  # Seismic data and results
PHASENET_MODEL = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"  # PhaseNet model
VMODEL_DIR = BASE_DIR / "quickquake/vmodels"  # Velocity models for HypoInverse
HYPO_BIN = BASE_DIR / "dependencies/hyp1.40/src/hyp1.40"  # HypoInverse binary (must be compiled)

# Scripts (located in quickquake/)
SCRIPTS = {
    "generate_config": BASE_DIR / "quickquake/QQ_config.py",
    "download_stations": BASE_DIR / "quickquake/QQ_dl_stations.py",
    "data_download": BASE_DIR / "quickquake/QQ_dl_data.py",
    "phasenet_predict": BASE_DIR / "quickquake/QQ_predict.py",
    "gamma_association": BASE_DIR / "quickquake/QQ_gamma.py",
    "localizacion": BASE_DIR / "quickquake/QQ_location.py"
}

# Control  (only modify these True/False values)
RUN_GENERATE_CONFIG = True
RUN_DOWNLOAD = True
RUN_PHASENET = True
RUN_GAMMA = True
RUN_LOCATION = True
MERGE_RESULTS = True

# =================================================================
# DO NOT MODIFY BELOW THIS LINE
# =================================================================

def create_directory(base_path, date_str):
    """Creates the directory structure for a time interval."""
    dir_path = base_path / date_str
    (dir_path / "waveforms").mkdir(parents=True, exist_ok=True)
    return dir_path

def process_interval(start, end, output_dir):
    """Processes a time interval."""
    # Step 1: Generate configuration
    config_file = output_dir / "config.json"
    if RUN_GENERATE_CONFIG:
        command = [
            sys.executable,  # Use the Python from the current environment (QuickQuake)
            str(SCRIPTS["generate_config"]),
            "--start", start.isoformat(),
            "--end", end.isoformat(),
            "--output", str(config_file)
        ]
        print("Executing: Configuration generation")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in configuration generation: {str(e)}")
            return False

    # Step 2: Download stations
    stations_file = output_dir / "stations.json"
    if RUN_DOWNLOAD:
        # Download stations
        command = [
            sys.executable,
            str(SCRIPTS["download_stations"]),
            "--config", str(config_file),
            "--output_dir", str(output_dir),
            "--plot"
        ]
        print("Executing: Stations download")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in stations download: {str(e)}")
            return False

        # Step 3: Download seismic data
        command = [
            sys.executable,
            str(SCRIPTS["data_download"]),
            "--config", str(config_file),
            "--output_dir", str(output_dir)
        ]
        print("Executing: Seismic data download")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in seismic data download: {str(e)}")
            return False

    # Step 4: Execute PhaseNet
    picks_file = output_dir / "picks.csv"
    if RUN_PHASENET:
        command = [
            sys.executable,
            str(SCRIPTS["phasenet_predict"]),
            "--model", str(PHASENET_MODEL),
            "--data_dir", str(output_dir / "waveforms"),
            "--data_list", str(output_dir / "input_data.csv"),
            "--stations", str(stations_file),
            "--result_dir", str(output_dir),
            "--format", "mseed_array",
            "--amplitude"
        ]
        print("Executing: PhaseNet detection")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in PhaseNet: {str(e)}")
            return False

    # Step 5: Gamma association
    gamma_file = output_dir / "gamma_catalog.csv"
    if RUN_GAMMA:
        command = [
            sys.executable,
            str(SCRIPTS["gamma_association"]),
            "--config", str(config_file),
            "--picks", str(picks_file),
            "--stations", str(stations_file),
            "--output_dir", str(output_dir)
        ]
        print("Executing: Gamma association")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in Gamma: {str(e)}")
            return False

    # Step 6: Localization with HypoInverse
    if RUN_LOCATION:
        command = [
            sys.executable,
            str(SCRIPTS["localizacion"]),
            "--date_dir", str(output_dir),
            "--vmodel_dir", str(VMODEL_DIR),
            "--hypo_bin", str(HYPO_BIN)
        ]
        print("Executing: HypoInverse localization")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error in HypoInverse: {str(e)}")
            return False

    return True

def merge_hyp_good(data_root, output_filename="consolidated_hyp_good.csv"):
    """Combines all hyp_good.csv files into a single catalog."""
    hyp_good_files = list(data_root.glob("**/output/hyp_good.csv"))
    
    if not hyp_good_files:
        print("No hyp_good.csv files found.")
        return

    dfs = []
    for file in hyp_good_files:
        try:
            df = pd.read_csv(file, header=None)
            print(f"Reading: {file}")
            if df.shape[1] >= 5:
                df = df.iloc[:, :5]
                df.columns = ["time", "latitude", "longitude", "depth", "magnitude"]
                dfs.append(df)
            else:
                print(f"{file} has less than 5 columns. Skipping.")
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")

    if not dfs:
        print("No data to merge.")
        return

    consolidated_df = pd.concat(dfs, ignore_index=True)
    output_path = data_root / output_filename
    consolidated_df.to_csv(output_path, index=False)
    print(f"\nCatalog consolidated at: {output_path}")

def main():
    start = datetime.fromisoformat(START_TIME_STR)
    end = datetime.fromisoformat(END_TIME_STR)
    
    current = start
    while current < end:
        interval_end = min(current + timedelta(hours=INCREMENT_HOURS), end)
        date_str = current.strftime("%Y%m%dT%H%M%S")
        output_dir = create_directory(DATA_ROOT, date_str)
        
        print(f"\n{'='*50}\nProcessing: {current} - {interval_end}\n{'='*50}")
        if process_interval(current, interval_end, output_dir):
            print(f"\nProcess successful: {output_dir}")
        else:
            print(f"\nProcess failed: {output_dir}")
        
        current = interval_end

    if MERGE_RESULTS:
        print("\n\n" + "="*50)
        print("Merging hyp_good.csv...")
        merge_hyp_good(DATA_ROOT)

if __name__ == "__main__":
    main()

