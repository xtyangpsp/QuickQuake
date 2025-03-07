import os
import json
import pickle
import time
import argparse  
import obspy
from obspy.clients.fdsn import Client

def download_waveforms(config_json, output_dir): # Function to download seismic waveforms.
    client = Client("IRIS")
    
    with open(config_json, "r") as fp: # Load the configuration from config.json with the time range and sensors.
        config = json.load(fp)
        
    # Create a subdirectory for waveforms.
    waveform_dir = os.path.join(output_dir, "waveforms")
    os.makedirs(waveform_dir, exist_ok=True)

    # Load stations from the current directory (not fixed)
    station_pkl = os.path.join(output_dir, "stations.pkl")
    with open(station_pkl, "rb") as fp:
        stations = pickle.load(fp)

    # Download waveforms. Iterate through all stations in stations.pkl
    max_retry = 10
    stream = obspy.Stream()
    starttime = obspy.UTCDateTime(config['starttime'])
    endtime = obspy.UTCDateTime(config['endtime'])

    for network in stations:
        for station in network:
            print(f"********{network.code}.{station.code}********")
            retry = 0
            while retry < max_retry:
                try:
                    tmp = client.get_waveforms(
                        network.code,
                        station.code,
                        "*",
                        config["channels"],
                        starttime,
                        endtime,
                    )
                    stream += tmp
                    break
                except Exception as err:
                    print(f"Error {network.code}.{station.code}: {err}")
                    retry += 1
                    time.sleep(5)

    # Save waveforms
    fname = f"{starttime.datetime.strftime('%Y-%m-%dT%H-%M-%S')}.mseed"
    if len(stream) > 0:
        stream.write(os.path.join(waveform_dir, fname), format="MSEED")
        print(f'Waveforms saved at: {os.path.join(waveform_dir, fname)}')
    else:
        print('Download failed - Empty Stream')
        
    
    # Save file list to CSV within the day's directory
    fname_csv = os.path.join(output_dir, "input_data.csv")
    with open(fname_csv, "w") as fp:
        fp.write("fname\n")  # Header
        fp.write(fname + "\n")  # MSEED file name    

if __name__ == "__main__":
    # Configure arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    args = parser.parse_args()
    
    download_waveforms(args.config, args.output_dir)  # 

