from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os

def convert_to_hypoinverse(picks_csv, events_csv, output_dir):
    # Cargar datos
    picks = pd.read_csv(picks_csv, sep=",")
    events = pd.read_csv(events_csv, sep=",")
    
    # Preprocesamiento
    picks = picks.loc[:, ~picks.columns.duplicated()]
    events.rename(columns={"event_index": "event_idx"}, inplace=True)
    events["match_id"] = events["event_idx"].astype(str)
    picks["match_id"] = picks["event_idx"].astype(str)
    
    # Crear directorio de salida si no existe
    hypo_dir = os.path.join(output_dir, "hypo_inverse")
    os.makedirs(hypo_dir, exist_ok=True)
    
    # Generar archivo HypoInverse
    output_path = os.path.join(hypo_dir, "hypoInput.arc")
    with open(output_path, "w") as out_file:
        picks_by_event = picks.groupby("match_id").groups
    
        for i in tqdm(range(len(events)), desc="Procesando eventos"):
            event = events.iloc[i]
            
            # Formatear línea del evento
            event_time = datetime.strptime(event["time"], "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y%m%d%H%M%S%f")[:-4]
            lat = abs(event["latitude"])
            lng = abs(event["longitude"])
            
            event_line = (
                f"{event_time}"
                f"{int(lat):02d}{'S' if event['latitude'] <0 else ' '}{int((lat - int(lat))*60*100):04.0f}"
                f"{int(lng):03d}{'E' if event['longitude'] >=0 else ' '}{int((lng - int(lng))*60*100):04.0f}"
                f"{int(event['depth(m)']/1e3*100):05.0f}"
            )
            out_file.write(event_line + "\n")
    
            # Formatear picks asociados
            picks_idx = picks_by_event.get(event["match_id"], [])
            for j in picks_idx:
                pick = picks.iloc[j]
                network_code, station_code, comp_code, channel_code = pick['id'].split('.')
                phase_type = pick['type'].upper()
                
                # Calcular peso y tiempo
                phase_weight = min(max(int((1 - pick['prob'])/(1 - 0.3)*4) - 1, 0), 3)
                pick_time = datetime.strptime(pick["timestamp"], "%Y-%m-%dT%H:%M:%S.%f")
                
                # Escribir línea según tipo de fase
                if phase_type == 'P':
                    line = (
                        f"{station_code:<5}{network_code:<2} {comp_code}{channel_code:<3}"
                        f" P {phase_weight}{pick_time.strftime('%Y%m%d%H%M')} {pick_time.strftime('%S%f')[:-4]}"
                    )
                elif phase_type == 'S':
                    line = (
                        f"{station_code:<5}{network_code:<2} {comp_code}{channel_code:<3}"
                        f"   4{pick_time.strftime('%Y%m%d%H%M')} {'':<12}{pick_time.strftime('%S%f')[:-4]} S {phase_weight}"
                    )
                out_file.write(line + "\n")
            
            out_file.write("\n")
    
    print(f"Archivo HypoInverse generado en: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convertir resultados Gamma a formato HypoInverse")
    parser.add_argument("--picks", required=True, help="Ruta a gamma_picks.csv")
    parser.add_argument("--events", required=True, help="Ruta a gamma_catalog.csv")
    parser.add_argument("--output_dir", required=True, help="Directorio de salida")
    args = parser.parse_args()
    
    convert_to_hypoinverse(args.picks, args.events, args.output_dir)
