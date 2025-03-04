
import os
import json
import obspy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
import time 
from collections import defaultdict
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


import json

# json file path
file_path = "/home/elizabeth/work/research/Pavlof_Project/stations.json"

# Open json 
with open(file_path, "r") as file:
    stations_data = json.load(file)


for station, details in stations_data.items():
    print(f"Station: {station}")
    for key, value in details.items():
        print(f"  {key}: {value}")
    print("\n")



import json

# input and output for json files
input_path = "/home/elizabeth/work/research/Pavlof_Project/stations.json"
output_path = "/home/elizabeth/work/research/Pavlof_Project/converted_station_list.json"

# Read the othe firts json file
with open(input_path, "r") as file:
    original_data = json.load(file)

# new dictionary 
converted_data = {}

# Transform each entry
for full_station_code, details in original_data.items():
    # Split station code  "."
    parts = full_station_code.split(".")
    
    # Extract network, station_code, and sensor_type 
    network = parts[0]  #  "AV"
    station_code = parts[1]  # "PS4A"
    sensor_type = parts[-1]  #  "BH"
    
    # Create the list of channels, sensor type prefixed
    channels = [f"{sensor_type}Z", f"{sensor_type}N", f"{sensor_type}E"]  # ["BHZ", "BHN", "BHE"]
    
    # Reorganize the information 
    converted_data[station_code] = {
        "network": network,
        "channels": channels,
        "coords": [details["latitude"], details["longitude"], details["elevation(m)"]]
    }

# Save tjson 
with open(output_path, "w") as file:
    json.dump(converted_data, file, indent=4)

print(f"data saved at  {output_path}")





