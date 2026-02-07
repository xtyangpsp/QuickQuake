#!/usr/bin/env python3
"""
This code is based on the "QuakeFlow" repository by Weiqiang Zhu (2021) 
See https://github.com/AI4EPS/QuakeFlow for more details.
This code has been modified 
"""
import os
import json
import pickle
import time
import argparse
import obspy
from obspy.clients.fdsn import Client

def download_waveforms(config_json, output_dir):
    # INPUT 1: config.json 
    with open(config_json) as fp:
        config = json.load(fp)

    # Usa el client definido en config.json (que viene del orquestador).
    # Si falta la clave "client", cae a IRIS.
    client_name = config.get("client", "IRIS")
    client = Client(client_name)
    print(f"[waveforms] FDSN client = {client_name}")

    # OUTPUT DIR: waveforms/ (creado aquí; PhaseNet lo usa después como --data_dir)
    waveform_dir = os.path.join(output_dir, "waveforms")
    os.makedirs(waveform_dir, exist_ok=True)

    with open(os.path.join(output_dir, "stations.pkl"), "rb") as fp:
        stations = pickle.load(fp)

    stream = obspy.Stream()
    starttime = obspy.UTCDateTime(config['starttime'])
    endtime = obspy.UTCDateTime(config['endtime'])
    fname = f"{starttime.datetime.strftime('%Y-%m-%dT%H-%M-%S')}.mseed"

    for network in stations:
        for station in network:
            print(f"********{network.code}.{station.code}********")
            retry = 0
            while retry < 10:
                try:
                    tmp = client.get_waveforms(
                        network.code, station.code, "*", config["channels"],
                        starttime, endtime
                    )
                    stream += tmp
                    break
                except Exception as err:
                    print(f"Error {network.code}.{station.code}: {err}")
                    retry += 1
                    time.sleep(5)

    if stream:
        stream.write(os.path.join(waveform_dir, fname), format="MSEED")
        print(f'Waveforms saved at: {os.path.join(waveform_dir, fname)}')
        with open(os.path.join(output_dir, "input_data.csv"), "w") as fp:
            fp.write(f"fname\n{fname}\n")
    else:
        print('Download failed - Empty Stream')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    download_waveforms(args.config, args.output_dir)

