#!/usr/bin/env python3



import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import argparse 
from collections import defaultdict
from obspy.clients.fdsn import Client

def download_stations(config_json, output_dir, plot=True):#Function to download seismic station information and save it to files
    client = Client("IRIS")
    
    with open(config_json, "r") as fp:#Load parameters from the config.json file
        config = json.load(fp)
        
    # Create the output directory if it does not exist.
    os.makedirs(output_dir, exist_ok=True)
         
    ####### Download stations ########
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
    )#Query IRIS and download data for stations that meet the criteria in config.json

    ####### Save stations ########
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
                    tmp_dict = {
                        "longitude": chn.longitude,
                        "latitude": chn.latitude,
                        "elevation(m)": chn.elevation,
                        "component": [chn.code[-1]],
                        "response": [round(chn.response.instrument_sensitivity.value, 2)],
                        "unit": chn.response.instrument_sensitivity.input_units.lower(),
                    }
                    station_locs[sid] = tmp_dict #Iterate through all downloaded stations, extract key information (location, elevation, sensor type, instrument response), store the data in a dictionary station_locs.
                    
    # Save files in the output directory
    stations.write(os.path.join(output_dir, 'stations.xml'), format='STATIONXML')
    
    station_json = os.path.join(output_dir, 'stations.json')
    with open(station_json, "w") as fp:
        json.dump(station_locs, fp, indent=2)

    station_pkl = os.path.join(output_dir, 'stations.pkl')
    with open(station_pkl, "wb") as fp:
        pickle.dump(stations, fp)
    
    if plot:
        ######## Plot stations ########
        station_locs_df = pd.DataFrame.from_dict(station_locs, orient="index")
        plt.figure(figsize=(10, 8))
        plt.plot(station_locs_df["longitude"], station_locs_df["latitude"], "^r", markersize=10, label="Stations")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.axis("scaled")
        plt.legend()
        plt.title(f"Number of stations: {len(station_locs_df)}")
        plot_path = os.path.join(output_dir, "station_map.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # Configure command-line arguments
    parser = argparse.ArgumentParser(description='Download seismic station information')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to the config.json file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for the files')
    parser.add_argument('--plot', action='store_true',
                       help='Generate a plot of station locations')
    
    args = parser.parse_args()
    
    # Run main function
    download_stations(
        config_json=args.config,
        output_dir=args.output_dir,
        plot=args.plot
    )#Receive parameters from the terminal with argparse e.g. python download_stations.py --config config.json --output_dir ./data --plot

