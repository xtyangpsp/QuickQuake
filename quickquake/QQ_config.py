#!/usr/bin/env python3



# generate_config.py
import os
import json
import obspy
import numpy as np
import argparse  # terminal parameters 

def set_config(start_time, end_time, config_json):#fuction generate json archive with region information
    region_name = "pavlof"
    center = (-161.8903, 55.4133)
    horizontal_degree = 1.0
    vertical_degree = 1.0

    # use the arguments dates
    starttime = obspy.UTCDateTime(start_time)# convert date to obspy.utc datatime
    endtime = obspy.UTCDateTime(end_time)

    
    client = "IRIS"#network
    network_list = ['AV']
    channel_list = "BHZ,BHN,BHE,SHZ,SHN,SHE"
    
    ####### Save config ########
    degree2km = np.pi * 6371 / 180# from ° to km 
    config = {
        "region": region_name,
        "center": center,
        "xlim_degree": [center[0] - horizontal_degree / 2, center[0] + horizontal_degree / 2],
        "ylim_degree": [center[1] - vertical_degree / 2, center[1] + vertical_degree / 2],
        "min_longitude": center[0] - horizontal_degree / 2,
        "max_longitude": center[0] + horizontal_degree / 2,
        "min_latitude": center[1] - vertical_degree / 2,
        "max_latitude": center[1] + vertical_degree / 2,
        "degree2km": degree2km,
        "starttime": starttime.datetime.isoformat(timespec="milliseconds"),
        "endtime": endtime.datetime.isoformat(timespec="milliseconds"),
        "networks": network_list,
        "channels": channel_list,
        "client": client,
        "phasenet": {},
        "gamma": {},
        "hypodd": {"MAXEVENT": 1e4}
    }# the infromation is storage in a dicctionary named config 

    with open(config_json, "w") as fp:
        json.dump(config, fp, indent=2)# Save the previously created dictionary in a JSON file. The file is saved in the path specified in the terminal

    print(f"Configuration saved to {config_json}")

if __name__ == "__main__":# execution with terminal arguments 
    
    parser = argparse.ArgumentParser()# it creates an object and manage the arguments that user define in the terminal 
    parser.add_argument("--start", required=True, help="initial date (ej. 2021-09-25T00:00:00)")
    parser.add_argument("--end", required=True, help="Finald date (ej. 2021-09-26T00:00:00)")
    parser.add_argument("--output", required=True, help="config.json rut (ej. data_root/20210925/config.json)")
    args = parser.parse_args()#read the user values and storage in args 

    set_config(args.start, args.end, args.output)  # Call the set_config() function with the values captured from the terminal.
