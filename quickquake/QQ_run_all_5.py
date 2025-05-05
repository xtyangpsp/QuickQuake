import os
import subprocess
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# =================================================================
# CONFIGURATION
# =================================================================

# Time settings
START = "2021-09-25T00:00:00"
END = "2021-09-25T06:00:00"
HOUR_STEP = 2

# Path setup
BASE_DIR = Path(__file__).parent.parent
DATA_ROOT = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"
VMODELS = BASE_DIR / "vmodels"
HYPO_BIN = BASE_DIR / "dependencies/hyp1.40/src/hyp1.40"

# Configuration setup
CENTER = (-161.8903, 55.4133)  # Longitud, Latitud
DEG = 1.0
NETWORKS = ['AV']
CHANNELS = "BHZ,BHN,BHE,SHZ,SHN,SHE"
CLIENT = "IRIS"
REGION = "pavlof"

SCRIPTS = {
    "config": BASE_DIR / "quickquake/QQ_config.py",
    "stations": BASE_DIR / "quickquake/QQ_dl_stations.py",
    "download": BASE_DIR / "quickquake/QQ_dl_data.py",
    "phasenet": BASE_DIR / "quickquake/QQ_predict.py",
    "gamma": BASE_DIR / "quickquake/QQ_gamma.py",
    "location": BASE_DIR / "quickquake/QQ_location.py"
}

RUN_CONFIG = True
RUN_DL = True
RUN_PHASENET = True
RUN_GAMMA = True
RUN_LOC = True
MERGE = True

# =================================================================
# LOGGING FUNCTIONS
# =================================================================

def log_config_details(out_dir):
    config_file = out_dir / "config.json"
    print("\n[CONFIG DETAILS]")
    print(f"Checking: {config_file}")
    
    if not config_file.exists():
        print("❌ Error: Config file missing")
        return False
    
    try:
        with open(config_file) as f:
            config = json.load(f)
        print("✅ Config válido con parámetros:")
        print(f"  Start: {config['start_time']}")
        print(f"  End: {config['end_time']}")
        print(f"  Networks: {config['networks']}")
        return True
    except Exception as e:
        print(f"❌ Error reading config: {str(e)}")
        return False

def log_stations_details(out_dir):
    stations_file = out_dir / "stations.json"
    print("\n[STATIONS DETAILS]")
    
    if not stations_file.exists():
        print("❌ Error: Stations file missing")
        return 0
    
    try:
        with open(stations_file) as f:
            stations = json.load(f)
        print(f"✅ Estaciones descargadas: {len(stations)}")
        if stations:
            print("  Ejemplo de estación:")
            print(f"  {stations[0]['network']}.{stations[0]['station']} "
                f"({stations[0]['latitude']}, {stations[0]['longitude']})")
        return len(stations)
    except Exception as e:
        print(f"❌ Error reading stations: {str(e)}")
        return 0

def log_download_details(out_dir):
    input_csv = out_dir / "input_data.csv"
    waveform_dir = out_dir / "waveforms"
    print("\n[DOWNLOAD DETAILS]")
    
    counts = {"stations": 0, "files": 0}
    
    if input_csv.exists():
        try:
            df = pd.read_csv(input_csv)
            counts["stations"] = len(df)
            print(f"✅ Estaciones para descargar: {len(df)}")
        except Exception as e:
            print(f"❌ Error reading input_data.csv: {str(e)}")
    
    if waveform_dir.exists():
        try:
            counts["files"] = len(list(waveform_dir.glob("*.mseed")))
            print(f"Archivos sísmicos descargados: {counts['files']}")
        except Exception as e:
            print(f"❌ Error counting waveforms: {str(e)}")
    
    return counts

def log_phasenet_details(out_dir):
    picks_file = out_dir / "picks.csv"
    print("\n[PHASENET DETAILS]")
    
    if not picks_file.exists():
        print("❌ Error: Picks file missing")
        return 0
    
    try:
        df = pd.read_csv(picks_file)
        print(f"✅ Picks detectados: {len(df)}")
        print("  Ejemplo de picks:")
        print("  Columnas:", df.columns.tolist())
        print(f"  Primer pick: {df.iloc[0]['timestamp']} "
            f"({df.iloc[0]['type']} @ {df.iloc[0]['station_id']})")
        return len(df)
    except Exception as e:
        print(f"❌ Error reading picks: {str(e)}")
        return 0

def log_gamma_details(out_dir):
    events_file = out_dir / "events.csv"
    print("\n[GAMMA DETAILS]")
    
    if not events_file.exists():
        print("❌ Error: Events file missing")
        return 0
    
    try:
        df = pd.read_csv(events_file)
        print(f"✅ Eventos agrupados: {len(df)}")
        print("  Columnas:", df.columns.tolist())
        if not df.empty:
            print(f"  Primer evento: {df.iloc[0]['event_id']} "
                f"con {df.iloc[0]['num_picks']} picks")
        return len(df)
    except Exception as e:
        print(f"❌ Error reading events: {str(e)}")
        return 0

def log_location_details(out_dir):
    output_dir = out_dir / "output"
    print("\n[LOCATION DETAILS]")
    
    results = {
        "total": 0,
        "good": 0,
        "bad": 0,
        "headers": {}
    }
    
    for quality in ['hyp_all.csv', 'hyp_good.csv', 'hyp_bad.csv']:
        file_path = output_dir / quality
        try:
            if file_path.exists():
                df = pd.read_csv(file_path, header=None)
                count = len(df)
                q_type = quality.split('_')[1].split('.')[0]
                results[q_type] = count
                results["total"] += count
                results["headers"][q_type] = df.iloc[0].values.tolist()[:5] if not df.empty else []
                print(f"  {quality}: {count} eventos")
        except Exception as e:
            print(f"❌ Error reading {quality}: {str(e)}")
    
    print("  Ejemplo de header (hyp_good.csv):")
    print("  ", results['headers'].get('good', []))
    return results

# =================================================================
# PROCESSING FUNCTIONS
# =================================================================

def run_step(cmd, step):
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print("[LOG] Paso ejecutado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {step} ERROR:")
        print(e.output)
        print(f"\nCommand failed: {' '.join(e.cmd)}")
        return False

def process_window(start, end, out_dir):
    metrics = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    config = out_dir / "config.json"
    stations = out_dir / "stations.json"
    picks = out_dir / "picks.csv"
    
    steps = [
        (RUN_CONFIG, 
         [
            sys.executable, str(SCRIPTS["config"]),
            "--start", start.isoformat(),
            "--end", end.isoformat(),
            "--output", str(config),
            "--center", str(CENTER[0]),  # Longitud como argumento separado
            str(CENTER[1]),              # Latitud como argumento separado
            "--deg", str(DEG),
            "--networks", ",".join(NETWORKS),
            "--channels", CHANNELS,
            "--client", CLIENT,
            "--region", REGION
         ], 
         "Config"),
        
        (RUN_DL, 
         [
            sys.executable, str(SCRIPTS["stations"]),
            "--config", str(config),
            "--output_dir", str(out_dir),
            "--plot"
         ], 
         "Stations"),
        
        (RUN_DL, 
         [
            sys.executable, str(SCRIPTS["download"]),
            "--config", str(config),
            "--output_dir", str(out_dir)
         ], 
         "Data Download"),
        
        (RUN_PHASENET, 
         [
            sys.executable, str(SCRIPTS["phasenet"]),
            "--model", str(MODEL_DIR),
            "--data_dir", str(out_dir/"waveforms"),
            "--data_list", str(out_dir/"input_data.csv"),
            "--stations", str(stations),
            "--result_dir", str(out_dir),
            "--format", "mseed_array",
            "--amplitude"
         ], 
         "PhaseNet"),
        
        (RUN_GAMMA, 
         [
            sys.executable, str(SCRIPTS["gamma"]),
            "--config", str(config),
            "--picks", str(picks),
            "--stations", str(stations),
            "--output_dir", str(out_dir)
         ], 
         "Gamma"),
        
        (RUN_LOC, 
         [
            sys.executable, str(SCRIPTS["location"]),
            "--date_dir", str(out_dir),
            "--vmodel_dir", str(VMODELS),
            "--hypo_bin", str(HYPO_BIN)
         ], 
         "Location")
    ]
    
    for condition, cmd, name in steps:
        if condition:
            print(f"\n{'='*30} {name.upper()} {'='*30}")
            success = run_step(cmd, name)
            
            if not success:
                return False
            
            if name == "Config":
                metrics["config"] = log_config_details(out_dir)
            elif name == "Stations":
                metrics["stations"] = log_stations_details(out_dir)
            elif name == "Data Download":
                metrics["download"] = log_download_details(out_dir)
            elif name == "PhaseNet":
                metrics["picks"] = log_phasenet_details(out_dir)
            elif name == "Gamma":
                metrics["events"] = log_gamma_details(out_dir)
            elif name == "Location":
                metrics["locations"] = log_location_details(out_dir)
    
    print("\nRESUMEN FINAL DE VENTANA:")
    print(f"• Estaciones: {metrics.get('stations', 0)}")
    print(f"• Picks: {metrics.get('picks', 0)}")
    print(f"• Eventos: {metrics.get('events', 0)}")
    if 'locations' in metrics:
        loc = metrics['locations']
        print(f"• Localizaciones: {loc.get('good', 0)} buenas, {loc.get('bad', 0)} malas")
    
    return True

def merge_results():
    files = list(DATA_ROOT.glob("**/output/hyp_good.csv"))
    if not files: 
        print("No hay archivos para consolidar")
        return
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, header=None).iloc[:, :5]
            df.columns = ["time", "lat", "lon", "depth", "mag"]
            dfs.append(df)
            print(f"Merging: {f}")
        except Exception as e:
            print(f"Skipped {f}: {e}")
    
    if dfs:
        consolidated = pd.concat(dfs)
        output_path = DATA_ROOT/"consolidated_hyp_good.csv"
        consolidated.to_csv(output_path, index=False)
        print(f"\n✅ Catálogo consolidado guardado en: {output_path}")
        print(f"Total de eventos: {len(consolidated)}")

# =================================================================
# MAIN EXECUTION
# =================================================================

def main():
    current = datetime.fromisoformat(START)
    end_time = datetime.fromisoformat(END)
    
    while current < end_time:
        window_end = min(current + timedelta(hours=HOUR_STEP), end_time)
        date_str = current.strftime("%Y%m%dT%H%M%S")
        output_dir = DATA_ROOT / date_str
        
        print(f"\n{'='*50}\nProcessing: {current} - {window_end}\n{'='*50}")
        if process_window(current, window_end, output_dir):
            print(f"\n✅ Completed: {output_dir}")
        else:
            print(f"\n❌ Failed: {output_dir}")
        current = window_end
    
    if MERGE: 
        print("\n\nIniciando consolidación de resultados...")
        merge_results()

if __name__ == "__main__":
    main()
