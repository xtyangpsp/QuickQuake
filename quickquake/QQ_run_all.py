import os
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# =================================================================
# MODIFIABLE CONFIGURATION - ADJUST THESE VALUES
# =================================================================

# Time configuration
START_TIME_STR = "2021-09-25T00:00:00"
END_TIME_STR = "2021-09-25T06:00:00"
INCREMENT_HOURS = 2

# Directory configuration
DATA_ROOT = Path("./data")
PHASENET_MODEL = "/home/elizabeth/soft/src/QuakeFlow/PhaseNet/model/190703-214543/"
VMODEL_DIR = Path("/home/elizabeth/soft/my_scripts/My_quakeFlow_1/vmodels")
HYPO_BIN = "/home/elizabeth/hyp1.40/source/hyp1.40"

# Conda environments
CONDA_PATHS = {
    "quakeflow": "/home/elizabeth/anaconda3/envs/quakeflow/bin/python",
    "phasenet": "/home/elizabeth/anaconda3/envs/phasenet/bin/python",
    "hypoinv": "/home/elizabeth/anaconda3/envs/hypoinv/bin/python"
}

# Scripts
SCRIPT_DIR_ROOT = Path("/home/elizabeth/soft/my_scripts/My_quakeFlow_1")
SCRIPTS = {
    "generate_config": SCRIPT_DIR_ROOT/"QQ_config.py",
    "download_stations": SCRIPT_DIR_ROOT/"QQ_dl_stations.py",
    "data_download": SCRIPT_DIR_ROOT/"QQ _dl_data.py",
    "phasenet_predict": SCRIPT_DIR_ROOT/"QQ_predict.py",
    "gamma_association": SCRIPT_DIR_ROOT/"QQ_gamma.py",
    "localizacion": SCRIPT_DIR_ROOT/"QQ_ location.py"
}

# Control flags (Modify only these True/False values!)
RUN_GENERATE_CONFIG = True
RUN_DOWNLOAD = True
RUN_PHASENET = True
RUN_GAMMA = True
RUN_LOCATION = True
MERGE_RESULTS = True 

# =================================================================
# DO NOT MODIFY BELOW THIS LINE
# =================================================================

def create_directory(base_path, date_str):
    """Creates directory structure for a time interval."""
    dir_path = base_path / date_str
    (dir_path / "waveforms").mkdir(parents=True, exist_ok=True)
    return dir_path

def run_step(condition, command, step_name, output_file=None):
    """Executes a workflow step if the condition is met."""
    if not condition:
        print(f"[-] Skipping step: {step_name}")
        return True
    
    if output_file and output_file.exists():
        print(f"[i] File already exists: {output_file}. Skipping step.")
        return True

    try:
        print(f"[+] Running: {step_name}")
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] Error in {step_name}: {str(e)}")
        return False

def process_interval(start, end, output_dir):
    """Processes a time interval."""
    # Step 1: Generate configuration
    config_file = output_dir / "config.json"
    if not run_step(
        RUN_GENERATE_CONFIG,
        [
            CONDA_PATHS["quakeflow"],
            str(SCRIPTS["generate_config"]),
            "--start", start.isoformat(),
            "--end", end.isoformat(),
            "--output", str(config_file)
        ],
        "Configuration generation",
        config_file
    ):
        return False

    # Step 2: Download stations
    stations_file = output_dir / "stations.json"
    if RUN_DOWNLOAD:
        if not run_step(
            True,
            [
                CONDA_PATHS["quakeflow"],
                str(SCRIPTS["download_stations"]),
                "--config", str(config_file),
                "--output_dir", str(output_dir),
                "--plot"
            ],
            "Download stations",
            stations_file
        ):
            return False

        # Step 3: Download seismic data
        if not run_step(
            True,
            [
                CONDA_PATHS["quakeflow"],
                str(SCRIPTS["data_download"]),
                "--config", str(config_file),
                "--output_dir", str(output_dir)
            ],
            "Seismic data download"
        ):
            return False

    # Step 4: Run PhaseNet
    picks_file = output_dir / "picks.csv"
    if not run_step(
        RUN_PHASENET,
        [
            CONDA_PATHS["phasenet"],
            str(SCRIPTS["phasenet_predict"]),
            "--model", PHASENET_MODEL,
            "--data_dir", str(output_dir / "waveforms"),
            "--data_list", str(output_dir / "input_data.csv"),
            "--stations", str(stations_file),
            "--result_dir", str(output_dir),
            "--format", "mseed_array",
            "--amplitude"
        ],
        "Detection with PhaseNet",
        picks_file
    ):
        return False

    # Step 5: Gamma Association
    gamma_file = output_dir / "gamma_catalog.csv"
    if not run_step(
        RUN_GAMMA,
        [
            CONDA_PATHS["quakeflow"],
            str(SCRIPTS["gamma_association"]),
            "--config", str(config_file),
            "--picks", str(picks_file),
            "--stations", str(stations_file),
            "--output_dir", str(output_dir)
        ],
        "Gamma Association",
        gamma_file
    ):
        return False

    # Step 6: Localization
    if not run_step(
        RUN_LOCATION,
        [
            CONDA_PATHS["hypoinv"],
            str(SCRIPTS["localizacion"]),
            "--date_dir", str(output_dir),
            "--vmodel_dir", str(VMODEL_DIR),
            "--hypo_bin", HYPO_BIN
        ],
        "Hypoinverse Localization"
    ):
        return False

    return True

def merge_hyp_good(data_root, output_filename="consolidated_hyp_good.csv"):
    """Une todos los archivos hyp_good.csv de las subcarpetas."""
    # Busca recursivamente todos los archivos hyp_good.csv
    hyp_good_files = list(data_root.glob("**/output/hyp_good.csv"))
    
    if not hyp_good_files:
        print("[!] No se encontraron archivos hyp_good.csv para unir.")
        return

    # Lista para almacenar los DataFrames
    dfs = []

    # Leer y procesar cada archivo
    for file in hyp_good_files:
        try:
            # Leer el archivo CSV
            df = pd.read_csv(file, header=None)  # Lee sin encabezado
            print(f"[+] Archivo leído: {file}")

            # Verificar que el archivo tenga al menos 5 columnas
            if df.shape[1] >= 5:
                # Seleccionar solo las primeras 5 columnas
                df = df.iloc[:, :5]
                # Asignar nombres a las columnas
                df.columns = ["time", "latitude", "longitude", "depth", "magnitude"]
                dfs.append(df)
            else:
                print(f"[!] Archivo {file} no tiene suficientes columnas. Se omitirá.")
        except Exception as e:
            print(f"[!] Error al leer {file}: {str(e)}")

    # Verificar si hay datos para unir
    if not dfs:
        print("[!] No hay datos válidos para unir.")
        return

    # Concatenar todos los DataFrames verticalmente
    consolidated_df = pd.concat(dfs, ignore_index=True)

    # Guardar el archivo consolidado
    output_path = data_root / output_filename
    consolidated_df.to_csv(output_path, index=False)
    print(f"\n[✔] Archivo consolidado guardado en: {output_path}")

def main():
    start = datetime.fromisoformat(START_TIME_STR)
    end = datetime.fromisoformat(END_TIME_STR)
    
    current = start
    while current < end:
        interval_end = min(current + timedelta(hours=INCREMENT_HOURS), end)
        date_str = current.strftime("%Y%m%dT%H%M%S")
        output_dir = create_directory(DATA_ROOT, date_str)
        
        print(f"\n{'='*50}\nProcesando: {current} - {interval_end}\n{'='*50}")
        if process_interval(current, interval_end, output_dir):
            print(f"\n[✔] Proceso completado: {output_dir}")
        else:
            print(f"\n[✖] Error en: {output_dir}")
        
        current = interval_end

    # Etapa adicional: Unir resultados
    if MERGE_RESULTS:
        print("\n\n" + "="*50)
        print("Iniciando consolidación de hyp_good.csv...")
        merge_hyp_good(DATA_ROOT)
if __name__ == "__main__":
    main()
    