#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# =================================================================
# CONFIGURACIÓN PRINCIPAL - CAMBIA ESTOS VALORES SEGÚN NECESITES
# =================================================================

# Configuración de tiempo
START_TIME_STR = "2021-09-25T00:00:00"
END_TIME_STR = "2021-09-25T06:00:00"
INCREMENT_HOURS = 2

# --- Configuración de rutas relativas ---
# Directorio base del proyecto (raíz de QuickQuake)
BASE_DIR = Path(__file__).parent.parent  # Ajusta según ubicación real de run_all.py

# Directorios clave
DATA_ROOT = BASE_DIR / "data"  # Datos sísmicos y resultados
PHASENET_MODEL = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"  # Modelo PhaseNet
VMODEL_DIR = BASE_DIR / "quickquake/vmodels"  # Modelos de velocidad para HypoInverse
HYPO_BIN = BASE_DIR / "dependencies/hyp1.40/src/hyp1.40"  # Binario de HypoInverse (debe estar compilado)

# Scripts (ubicados en quickquake/)
SCRIPTS = {
    "generate_config": BASE_DIR / "quickquake/QQ_config.py",
    "download_stations": BASE_DIR / "quickquake/QQ_dl_stations.py",
    "data_download": BASE_DIR / "quickquake/QQ_dl_data.py",
    "phasenet_predict": BASE_DIR / "quickquake/QQ_predict.py",
    "gamma_association": BASE_DIR / "quickquake/QQ_gamma.py",
    "localizacion": BASE_DIR / "quickquake/QQ_location.py"
}

# Flags de control (¡Solo modifica estos True/False!)
RUN_GENERATE_CONFIG = True
RUN_DOWNLOAD = True
RUN_PHASENET = True
RUN_GAMMA = True
RUN_LOCATION = True
MERGE_RESULTS = True

# =================================================================
# NO MODIFICAR A PARTIR DE AQUÍ
# =================================================================

def create_directory(base_path, date_str):
    """Crea la estructura de directorios para un intervalo de tiempo."""
    dir_path = base_path / date_str
    (dir_path / "waveforms").mkdir(parents=True, exist_ok=True)
    return dir_path

def process_interval(start, end, output_dir):
    """Procesa un intervalo de tiempo."""
    # Paso 1: Generar configuración
    config_file = output_dir / "config.json"
    if RUN_GENERATE_CONFIG:
        command = [
            sys.executable,  # Usa el Python del ambiente actual (QuickQuake)
            str(SCRIPTS["generate_config"]),
            "--start", start.isoformat(),
            "--end", end.isoformat(),
            "--output", str(config_file)
        ]
        print("[+] Ejecutando: Generación de configuración")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error en generación de configuración: {str(e)}")
            return False

    # Paso 2: Descargar estaciones
    stations_file = output_dir / "stations.json"
    if RUN_DOWNLOAD:
        # Descargar estaciones
        command = [
            sys.executable,
            str(SCRIPTS["download_stations"]),
            "--config", str(config_file),
            "--output_dir", str(output_dir),
            "--plot"
        ]
        print("[+] Ejecutando: Descarga de estaciones")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error en descarga de estaciones: {str(e)}")
            return False

        # Paso 3: Descargar datos sísmicos
        command = [
            sys.executable,
            str(SCRIPTS["data_download"]),
            "--config", str(config_file),
            "--output_dir", str(output_dir)
        ]
        print("[+] Ejecutando: Descarga de datos sísmicos")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error en descarga de datos: {str(e)}")
            return False

    # Paso 4: Ejecutar PhaseNet
    picks_file = output_dir / "picks.csv"
    if RUN_PHASENET:
        command = [
            sys.executable,
            str(SCRIPTS["phasenet_predict"]),
            "--model", str(PHASENET_MODEL),
            "--data_dir", str(output_dir / "waveforms"),
            "--data_list", str(output_dir / "input_data.csv"),
            "--stations", str(stations_file),
            "--result_dir", str(output_dir),
            "--format", "mseed_array",
            "--amplitude"
        ]
        print("[+] Ejecutando: Detección con PhaseNet")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error en PhaseNet: {str(e)}")
            return False

    # Paso 5: Asociación Gamma
    gamma_file = output_dir / "gamma_catalog.csv"
    if RUN_GAMMA:
        command = [
            sys.executable,
            str(SCRIPTS["gamma_association"]),
            "--config", str(config_file),
            "--picks", str(picks_file),
            "--stations", str(stations_file),
            "--output_dir", str(output_dir)
        ]
        print("[+] Ejecutando: Asociación Gamma")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error en Gamma: {str(e)}")
            return False

    # Paso 6: Localización con HypoInverse
    if RUN_LOCATION:
        command = [
            sys.executable,
            str(SCRIPTS["localizacion"]),
            "--date_dir", str(output_dir),
            "--vmodel_dir", str(VMODEL_DIR),
            "--hypo_bin", str(HYPO_BIN)
        ]
        print("[+] Ejecutando: Localización HypoInverse")
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error en HypoInverse: {str(e)}")
            return False

    return True

def merge_hyp_good(data_root, output_filename="consolidated_hyp_good.csv"):
    """Combina todos los archivos hyp_good.csv en un solo catálogo."""
    hyp_good_files = list(data_root.glob("**/output/hyp_good.csv"))
    
    if not hyp_good_files:
        print("[!] No se encontraron archivos hyp_good.csv.")
        return

    dfs = []
    for file in hyp_good_files:
        try:
            df = pd.read_csv(file, header=None)
            print(f"[+] Leyendo: {file}")
            if df.shape[1] >= 5:
                df = df.iloc[:, :5]
                df.columns = ["time", "latitude", "longitude", "depth", "magnitude"]
                dfs.append(df)
            else:
                print(f"[!] {file} tiene menos de 5 columnas. Omitiendo.")
        except Exception as e:
            print(f"[!] Error leyendo {file}: {str(e)}")

    if not dfs:
        print("[!] No hay datos para unir.")
        return

    consolidated_df = pd.concat(dfs, ignore_index=True)
    output_path = data_root / output_filename
    consolidated_df.to_csv(output_path, index=False)
    print(f"\n[✔] Catálogo consolidado en: {output_path}")

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
            print(f"\n[✔] Proceso exitoso: {output_dir}")
        else:
            print(f"\n[✖] Fallo en: {output_dir}")
        
        current = interval_end

    if MERGE_RESULTS:
        print("\n\n" + "="*50)
        print("Consolidando hyp_good.csv...")
        merge_hyp_good(DATA_ROOT)

if __name__ == "__main__":
    main()
