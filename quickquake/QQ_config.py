#!/usr/bin/env python3
"""
This code has been modified, and  is based on the "QuakeFlow" repository by Weiqiang Zhu (2021) 
See https://github.com/AI4EPS/QuakeFlow for more details.

"""

# generate_config.py
import json
import obspy
import numpy as np
import argparse

def set_config(start_time, end_time, config_json,
               center, deg, networks, channels, client, region):
    xlim = [center[0] - deg/2, center[0] + deg/2]
    ylim = [center[1] - deg/2, center[1] + deg/2]

    config = {
        "region": region,
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
        "networks": networks,
        "channels": channels,
        "client": client,
        "phasenet": {},
        "gamma": {},
        "hypodd": {"MAXEVENT": 1e4}
    }

    with open(config_json, 'w') as fp:
        json.dump(config, fp, indent=2) ## This JSON serves as the central input for the entire QuickQuake pipeline.
    
    print(f"Configuration saved to {config_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="Initial date (e.g., 2021-09-25T00:00:00)")
    parser.add_argument("--end", required=True, help="End date (e.g., 2021-09-26T00:00:00)")
    parser.add_argument("--output", required=True, help="Output config.json path (e.g., data_root/20210925/config.json)")
    parser.add_argument("--center", required=True, help="Center coordinate as lon,lat (e.g., -161.8903,55.4133)")
    parser.add_argument("--deg", required=True, type=float, help="Degree span (e.g., 1.0)")
    parser.add_argument("--networks", required=True, help="Comma separated networks (e.g., AV)")
    parser.add_argument("--channels", required=True, help="Comma separated channels (e.g., BHZ,BHN,BHE,SHZ,SHN,SHE)")
    parser.add_argument("--client", required=True, help="Client (e.g., IRIS)")
    parser.add_argument("--region", required=True, help="Region (e.g., pavlof)")
    args = parser.parse_args()
    
    # Convert the center string to a tuple of floats
    try:
        center = tuple(map(float, args.center.split(',')))
    except Exception as e:
        raise ValueError("The --center argument must use the format lon,lat") from e

    networks = args.networks.split(',')
# This section runs the configuration step and writes config.json.

    set_config(args.start, args.end, args.output, center, args.deg, networks, args.channels, args.client, args.region)
