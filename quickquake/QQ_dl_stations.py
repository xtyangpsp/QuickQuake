#!/usr/bin/env python3
"""
This code is based on the "QuakeFlow" repository by Weiqiang Zhu (2021) 
See https://github.com/AI4EPS/QuakeFlow for more details.
This code has been modified 
"""
import os
import json
import pickle
import argparse
from collections import defaultdict
from obspy.clients.fdsn import Client

def download_stations(config_json, output_dir, plot=True):
    # INPUT: config.json 
    with open(config_json) as fp:
        config = json.load(fp)

    # Usa el client definido en config.json (que viene del orquestador).
    # Si por alguna razón no existe la clave "client", cae a "IRIS".
    client_name = config.get("client", "IRIS")
    client = Client(client_name)
    print(f"[stations] FDSN client = {client_name}")

    os.makedirs(output_dir, exist_ok=True)

    stations = client.get_stations(
        network=",".join(config["networks"]),
        station="*",
        starttime=config["starttime"],
        endtime=config["endtime"],
        minlongitude=config["xlim_degree"][0],
        maxlongitude=config["xlim_degree"][1],
        minlatitude=config["ylim_degree"][0],
        maxlatitude=config["ylim_degree"][1],
        channel=config["channels"],
        level="response",
    )

    station_locs = defaultdict(dict)
    for network in stations:
        for station in network:
            for chn in station:
                sid = f"{network.code}.{station.code}.{chn.location_code}.{chn.code[:-1]}"
                if sid in station_locs:
                    if chn.code[-1] not in station_locs[sid]["component"]:
                        station_locs[sid]["component"].append(chn.code[-1])
                        station_locs[sid]["response"].append(
                            round(chn.response.instrument_sensitivity.value, 2)
                        )
                else:
                    station_locs[sid] = {
                        "longitude": chn.longitude,
                        "latitude": chn.latitude,
                        "elevation(m)": chn.elevation,
                        "component": [chn.code[-1]],
                        "response": [round(chn.response.instrument_sensitivity.value, 2)],
                        "unit": chn.response.instrument_sensitivity.input_units.lower(),
                    }

    stations.write(os.path.join(output_dir, 'stations.xml'), format='STATIONXML')
    
    with open(os.path.join(output_dir, 'stations.json'), 'w') as fp:
        json.dump(station_locs, fp, indent=2)

    with open(os.path.join(output_dir, 'stations.pkl'), 'wb') as fp:
        pickle.dump(stations, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download seismic station information')
    parser.add_argument('--config', required=True, help='Path to config.json')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--plot', action='store_true', help='(Deprecated) Plot flag kept for compatibility')
    
    args = parser.parse_args()
    
    download_stations(
        config_json=args.config,
        output_dir=args.output_dir,
        plot=args.plot  # Still passed but no longer used
    )
