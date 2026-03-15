#!/usr/bin/env python3
"""
This code is based on the "QuakeFlow" repository by Weiqiang Zhu (2021)
See https://github.com/AI4EPS/QuakeFlow for more details.
This code has been modified
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from pyproj import Proj
from gamma.utils import association


def main():
    parser = argparse.ArgumentParser(description="Seismic event association with GaMMA")
    parser.add_argument("--config", required=True, help="JSON configuration file")
    parser.add_argument("--picks", required=True, help="CSV file with picks")
    parser.add_argument("--stations", required=True, help="JSON file with stations")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")

    # user-facing overrides (from orchestrator)
    parser.add_argument("--method", default="BGMM", type=str, help="Association method: BGMM or GMM")
    parser.add_argument("--min_picks_per_eq", default=6, type=int, help="Minimum total picks per event")
    parser.add_argument("--max_sigma11", default=1.1, type=float, help="Max sigma11")
    parser.add_argument("--max_sigma22", default=2.2, type=float, help="Max sigma22")
    parser.add_argument("--max_sigma12", default=1.2, type=float, help="Max sigma12")
    parser.add_argument(
        "--oversample_factor",
        default=None,
        type=int,
        help="Oversample factor (e.g., 30 for BGMM, 1 for GMM). If not set, uses method default.",
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as fp:
        config = json.load(fp)

    # Load picks
    picks = pd.read_csv(args.picks, parse_dates=["phase_time"])

    # Remove existing amplitude columns (to avoid duplicates)
    for col in ["phase_amplitude", "phase_amplitude.1"]:
        if col in picks.columns:
            picks.drop(columns=[col], inplace=True)

    # Create necessary columns expected by GaMMA
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

        proj = Proj(
            f"+proj=sterea +lon_0={config['center'][0]} +lat_0={config['center'][1]} +units=km"
        )
        stations[["x(km)", "y(km)"]] = stations.apply(
            lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
        )
        stations["z(km)"] = stations["elevation(m)"].apply(lambda x: -x / 1e3)

    # GaMMA configuration (base)
    config["use_dbscan"] = False
    config["use_amplitude"] = True

    # --- user-facing overrides ---
    config["method"] = str(args.method).upper()
    config["min_picks_per_eq"] = int(args.min_picks_per_eq)
    config["max_sigma11"] = float(args.max_sigma11)
    config["max_sigma22"] = float(args.max_sigma22)
    config["max_sigma12"] = float(args.max_sigma12)

    # oversample_factor: method default unless user overrides
    if args.oversample_factor is None:
        if config["method"] == "BGMM":
            config["oversample_factor"] = 30
        elif config["method"] == "GMM":
            config["oversample_factor"] = 1
        else:
            raise ValueError(f"Unsupported method: {config['method']}. Use BGMM or GMM.")
    else:
        config["oversample_factor"] = int(args.oversample_factor)

    # Remaining fixed config (keep your existing values)
    config["dims"] = ["x(km)", "y(km)", "z(km)"]
    config["vel"] = {"p": 6.0, "s": 6.0 / 1.73}

    lon_min, lon_max = config["xlim_degree"]
    lat_min, lat_max = config["ylim_degree"]
    x0, y0 = config["center"]  # center (lon, lat)

    # project x(km) holding latitude constant at center
    x_min, _ = proj(lon_min, y0)
    x_max, _ = proj(lon_max, y0)
    config["x(km)"] = (x_min, x_max)

    # project y(km) holding longitude constant at center
    _, y_min = proj(x0, lat_min)
    _, y_max = proj(x0, lat_max)
    config["y(km)"] = (y_min, y_max)

    config["z(km)"] = (0, 60)

    config["bfgs_bounds"] = (
        (config["x(km)"][0] - 1, config["x(km)"][1] + 1),
        (config["y(km)"][0] - 1, config["y(km)"][1] + 1),
        (0, config["z(km)"][1] + 1),
        (None, None),
    )

    config["dbscan_eps"] = 15
    config["dbscan_min_samples"] = 4

    # keep your existing defaults for these
    config["min_p_picks_per_eq"] = 0
    config["min_s_picks_per_eq"] = 0

    # keep your existing values unless you later make them user-facing
    config["use_dbscan"] = False
    config["use_amplitude"] = True

    # sigma bounds (your script uses these keys later; keep them consistent)
    # NOTE: GaMMA expects these names; you were already using them.
    # They are already set above via args -> config.

    if config["use_amplitude"]:
        picks = picks[picks["amp"] != -1]

    # GaMMA association
    event_idx0 = 1
    catalogs, assignments = association(picks, stations, config, event_idx0, method=config["method"])

    # Process results
    catalogs = pd.DataFrame(
        catalogs,
        columns=["time"] + config["dims"] + [
            "magnitude",
            "sigma_time",
            "sigma_amp",
            "cov_time_amp",
            "event_index",
            "gamma_score",
        ],
    )

    # Coordinate conversion
   
    if len(catalogs) == 0:
        # No events: create empty columns so saving doesn't crash
        catalogs["longitude"] = pd.Series(dtype=float)
        catalogs["latitude"]  = pd.Series(dtype=float)
        catalogs["depth(m)"]  = pd.Series(dtype=float)
    else:
        def xy_to_lonlat(row):
            # inverse=True expects x, y (NOT longitude/latitude)
            lon, lat = proj(row["x(km)"], row["y(km)"], inverse=True)
            return pd.Series([lon, lat], index=["longitude", "latitude"])

        lonlat = catalogs.apply(xy_to_lonlat, axis=1)
        catalogs[["longitude", "latitude"]] = lonlat
        catalogs["depth(m)"] = catalogs["z(km)"].astype(float) * 1e3


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
            "time",
            "magnitude",
            "longitude",
            "latitude",
            "depth(m)",
            "sigma_time",
            "sigma_amp",
            "cov_time_amp",
            "gamma_score",
            "event_index",
        ],
    )

    # Picks
    picks_path = os.path.join(args.output_dir, "gamma_picks.csv")
    assignments_df = pd.DataFrame(assignments, columns=["pick_idx", "event_idx", "prob_gamma"])
    picks_output = picks.join(assignments_df.set_index("pick_idx"), how="left").fillna(-1).astype({"event_idx": int})

    # Rename column "amp" to "phase_amplitude" in the output file
    picks_output = picks_output.rename(columns={"amp": "phase_amplitude"})

    picks_output[["id", "timestamp", "type", "prob", "phase_amplitude", "event_idx", "prob_gamma"]].to_csv(
        picks_path,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )

    print(f"Process completed. Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
