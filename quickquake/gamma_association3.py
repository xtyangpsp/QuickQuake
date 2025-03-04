import os
import pandas as pd
import json
from pyproj import Proj
from gamma.utils import association
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Seismic event association with GaMMA")
    parser.add_argument("--config", required=True, help="JSON configuration file")
    parser.add_argument("--picks", required=True, help="CSV file with picks")
    parser.add_argument("--stations", required=True, help="JSON file with stations")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as fp:
        config = json.load(fp)

    # Load picks
    picks = pd.read_csv(args.picks, parse_dates=["phase_time"])
    
    # Remove existing amplitude columns (to avoid duplicates)
    for col in ['phase_amplitude', 'phase_amplitude.1']:
        if col in picks.columns:
            picks.drop(columns=[col], inplace=True)
    
    # Create necessary columns
    picks["id"] = picks["station_id"]
    picks["timestamp"] = picks["phase_time"]
    picks["amp"] = picks["phase_amp"]
    picks["type"] = picks["phase_type"]
    picks["prob"] = picks["phase_score"]

    # Load stations
    with open(args.stations, "r") as fp:
        stations = json.load(fp)
        stations = pd.DataFrame.from_dict(stations, orient="index")
        stations["id"] = stations.index
        proj = Proj(f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km")
        stations[["x(km)", "y(km)"]] = stations.apply(
            lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
        stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)

    # GaMMA configuration (keeping original parameters)
    config["use_dbscan"] = False
    config["use_amplitude"] = True
    config["method"] = "BGMM"
    
    # Critical missing parameter in last association script version
    if config["method"] == "BGMM":
        config["oversample_factor"] = 4
    elif config["method"] == "GMM":
        config["oversample_factor"] = 1

    config["dims"] = ["x(km)", "y(km)", "z(km)"]
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.73}
    config["x(km)"] = (np.array(config["xlim_degree"]) - np.array(config["center"][0])) * config["degree2km"]
    config["y(km)"] = (np.array(config["ylim_degree"]) - np.array(config["center"][1])) * config["degree2km"]
    config["z(km)"] = (0, 60)
    config["bfgs_bounds"] = (
        (config["x(km)"][0] - 1, config["x(km)"][1] + 1),
        (config["y(km)"][0] - 1, config["y(km)"][1] + 1),
        (0, config["z(km)"][1] + 1),
        (None, None),
    )
    config["dbscan_eps"] = 10
    config["dbscan_min_samples"] = 3
    config["min_picks_per_eq"] = 3
    config["max_sigma11"] = 2.0
    config["max_sigma22"] = 2.0
    config["max_sigma12"] = 1.0

    if config["use_amplitude"]:
        picks = picks[picks["amp"] != -1]

    # GaMMA association
    event_idx0 = 1
    catalogs, assignments = association(picks, stations, config, event_idx0, method=config["method"])

    # Process results
    catalogs = pd.DataFrame(
        catalogs,
        columns=["time"] + config["dims"] + [
            "magnitude", "sigma_time", "sigma_amp", "cov_time_amp", 
            "event_index", "gamma_score"
        ]
    )
    
    # Coordinate conversion
    catalogs[["longitude", "latitude"]] = catalogs.apply(
        lambda x: pd.Series(proj(longitude=x["x(km)"], latitude=x["y(km)"], inverse=True)),
        axis=1,
    )
    catalogs["depth(m)"] = catalogs["z(km)"].apply(lambda x: x * 1e3)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Catalog
    catalog_path = os.path.join(args.output_dir, "gamma_catalog.csv")
    catalogs.sort_values(by=["time"]).to_csv(
        catalog_path,
        index=False,
        float_format="%.3f",
        date_format="%Y-%m-%dT%H:%M:%S.%f",
        columns=[
            "time", "magnitude", "longitude", "latitude", "depth(m)",
            "sigma_time", "sigma_amp", "cov_time_amp", "gamma_score", "event_index"
        ]
    )
    
    # Picks
    picks_path = os.path.join(args.output_dir, "gamma_picks.csv")
    assignments_df = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
    picks_output = picks.join(assignments_df.set_index("pick_idx"), how="left").fillna(-1).astype({'event_idx': int})
    
    # Rename column "amp" to "phase_amplitude" in the output file
    picks_output = picks_output.rename(columns={"amp": "phase_amplitude"})
    
    picks_output[["id", "timestamp", "type", "prob", "phase_amplitude", "event_idx", "prob_gamma"]].to_csv(
        picks_path, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f"
    )

    print(f"Process completed. Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()

