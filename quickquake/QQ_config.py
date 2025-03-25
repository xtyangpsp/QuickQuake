#!/usr/bin/env python3
"""
This code is based on the "QuakeFlow" repository by Weiqiang Zhu (2021) 
See https://github.com/AI4EPS/QuakeFlow for more details.
This code has been modified 
"""

# generate_config.py
import json
import obspy
import numpy as np
import argparse

def set_config(start_time, end_time, config_json):
    center = (-161.8903, 55.4133)
    deg = 1.0
    xlim = [center[0] - deg/2, center[0] + deg/2]
    ylim = [center[1] - deg/2, center[1] + deg/2]

    config = {
        "region": "pavlof",
        "center": center,
        "xlim_degree": xlim,
        "ylim_degree": ylim,
        "min_longitude": xlim[0],
        "max_longitude": xlim[1],
        "min_latitude": ylim[0],
        "max_latitude": ylim[1],
        "degree2km": np.pi * 6371 / 180,
        "starttime": obspy.UTCDateTime(start_time).datetime.isoformat(timespec="milliseconds"),
        "endtime": obspy.UTCDateTime(end_time).datetime.isoformat(timespec="milliseconds"),
        "networks": ['AV'],
        "channels": "BHZ,BHN,BHE,SHZ,SHN,SHE",
        "client": "IRIS",
        "phasenet": {},
        "gamma": {},
        "hypodd": {"MAXEVENT": 1e4}
    }

    with open(config_json, 'w') as fp:
        json.dump(config, fp, indent=2)
    
    print(f"Configuration saved to {config_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Initial date (e.g., 2021-09-25T00:00:00)")
    parser.add_argument("--end", required=True, help="End date (e.g., 2021-09-26T00:00:00)")
    parser.add_argument("--output", required=True, help="Output config.json path (e.g., data_root/20210925/config.json)")
    args = parser.parse_args()
    
    set_config(args.start, args.end, args.output)
