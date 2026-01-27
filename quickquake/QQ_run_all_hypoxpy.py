#!/usr/bin/env python3
"""
QuickQuake: run pipeline in time chunks (Config -> Stations -> Download -> PhaseNet -> GaMMA),
then merge ALL GaMMA outputs into data/merged/, and finally prepare ONE global station list
for HypoXPy in data/merged/input/GAMMA_station_list.json.

Optionally: run HypoXPy (HypoInverse + HypoDD) as a final relocation step.
"""

import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import shutil

# =================================================================
# CONFIGURATION
# =================================================================

START = "2021-09-19T15:00:00"
END = "2021-09-19T17:00:00"
HOUR_STEP = 1

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"

CENTER = (-161.8903, 55.4133)
DEG = 1.0
NETWORKS = ["AV"]
CHANNELS = "BHZ,BHN,BHE,SHZ,SHN,SHE"
CLIENT = "IRIS"
REGION = "pavlof"

SCRIPTS = {
    "config": BASE_DIR / "quickquake/QQ_config.py",
    "stations": BASE_DIR / "quickquake/QQ_dl_stations.py",
    "download": BASE_DIR / "quickquake/QQ_dl_data.py",
    "phasenet": BASE_DIR / "quickquake/QQ_predict.py",
    "gamma": BASE_DIR / "quickquake/QQ_gamma.py",
    "location": BASE_DIR / "quickquake/QQ_location_hypoxpy.py",  # <-- NUEVO
}

RUN_CONFIG = False
RUN_DL = False
RUN_PHASENET =  False
RUN_GAMMA = False

# Merge oficial (TU merge) oj
RUN_MERGE_GAMMA = False

# NUEVO: Localización / relocalización (HypoXPy)
RUN_LOCATION = True
LOCATION_BINPATH = "/home/elizabeth/bin"  # <-- AJUSTA si cambia
LOCATION_NAMEBASE = "GAMMA"
# Ejemplos opcionales:
# LOCATION_EXTRA_ARGS = ["--cleanup"]
LOCATION_EXTRA_ARGS = []


# =================================================================
# HELPERS
# =================================================================

def run_step(cmd, step, cwd=None):
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"{step} error: {e}")
        return False


def process_window(start, end, out_dir):
    """
    Corre SOLO hasta GaMMA dentro de cada carpeta/chunk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    config = out_dir / "config.json"
    stations = out_dir / "stations.json"
    picks = out_dir / "picks.csv"

    steps = [
        (RUN_CONFIG, [sys.executable, str(SCRIPTS["config"]),
                      "--start", start.isoformat(),
                      "--end", end.isoformat(),
                      "--output", str(config),
                      "--center=" + f"{CENTER[0]},{CENTER[1]}",
                      "--deg", str(DEG),
                      "--networks", ",".join(NETWORKS),
                      "--channels", CHANNELS,
                      "--client", CLIENT,
                      "--region", REGION], "Config"),

        (RUN_DL, [sys.executable, str(SCRIPTS["stations"]),
                  "--config", str(config),
                  "--output_dir", str(out_dir),
                  "--plot"], "Stations"),

        (RUN_DL, [sys.executable, str(SCRIPTS["download"]),
                  "--config", str(config),
                  "--output_dir", str(out_dir)], "Data Download"),

        (RUN_PHASENET, [sys.executable, str(SCRIPTS["phasenet"]),
                        "--model", str(MODEL_DIR),
                        "--data_dir", str(out_dir / "waveforms"),
                        "--data_list", str(out_dir / "input_data.csv"),
                        "--stations", str(stations),
                        "--result_dir", str(out_dir),
                        "--format", "mseed_array",
                        "--amplitude"], "PhaseNet"),

        (RUN_GAMMA, [sys.executable, str(SCRIPTS["gamma"]),
                     "--config", str(config),
                     "--picks", str(picks),
                     "--stations", str(stations),
                     "--output_dir", str(out_dir)], "Gamma"),
    ]

    for condition, cmd, name in steps:
        if condition:
            print(f"Running: {name}")
            if not run_step(cmd, name, cwd=BASE_DIR):
                return False

    return True


# =================================================================
# YOUR OFFICIAL MERGE (INTEGRATED)
# =================================================================

def merge_gamma_picks_with_event_id(base_dir: Path, out_dir: Path) -> Path:
    pattern = "**/gamma_picks.csv"
    out_path = out_dir / "gammapicks_id.csv"

    files = sorted(base_dir.glob(pattern))
    print(f"Encontré {len(files)} archivos gamma_picks.csv")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            window_id = f.parent.name

            if "event_idx" not in df.columns:
                raise ValueError(f"Falta 'event_idx' en {f}. Columnas: {list(df.columns)}")

            mask = df["event_idx"].notna()
            df["event_id"] = pd.NA
            df.loc[mask, "event_id"] = (
                window_id + "_" + df.loc[mask, "event_idx"].astype(float).astype(int).astype(str)
            )

            frames.append(df)

        except Exception as e:
            print(f"[Error leyendo {f}]: {e}")

    if not frames:
        raise RuntimeError("No se pudo leer ningún gamma_picks.csv. Revisa DATA_ROOT y estructura de carpetas.")

    merged = pd.concat(frames, ignore_index=True)
    n_unq = merged["event_id"].nunique(dropna=True)
    print(f"Total picks: {len(merged)} | event_id únicos: {n_unq}")

    merged.to_csv(out_path, index=False)
    print(f"Guardado: {out_path}")
    return out_path


def merge_gamma_catalog_with_event_id(base_dir: Path, out_dir: Path) -> Path:
    pattern = "**/gamma_catalog.csv"
    out_path = out_dir / "gammacatalog_id.csv"

    files = sorted(base_dir.glob(pattern))
    print(f"Encontré {len(files)} archivos gamma_catalog.csv")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            window_id = f.parent.name

            if "event_index" not in df.columns:
                raise ValueError(f"Falta 'event_index' en {f}. Columnas: {list(df.columns)}")

            mask = df["event_index"].notna()
            df["event_id"] = pd.NA
            df.loc[mask, "event_id"] = (
                window_id + "_" + df.loc[mask, "event_index"].astype(float).astype(int).astype(str)
            )

            frames.append(df)

        except Exception as e:
            print(f"[Error leyendo {f}]: {e}")

    if not frames:
        raise RuntimeError("No se pudo leer ningún gamma_catalog.csv. Revisa DATA_ROOT y estructura de carpetas.")

    merged = pd.concat(frames, ignore_index=True)
    n_unq = merged["event_id"].nunique(dropna=True)
    print(f"Total eventos (filas): {len(merged)} | event_id únicos: {n_unq}")

    merged.to_csv(out_path, index=False)
    print(f"Guardado: {out_path}")
    return out_path


def run_merge_block(first_chunk_dir: Path):
    """
    Corre el merge oficial de GaMMA y prepara data/merged/input para HypoXPy.
    """
    merged_dir = DATA_ROOT / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # 1) Merge oficial (tus CSV con event_id)
    picks_out = merge_gamma_picks_with_event_id(DATA_ROOT, merged_dir)
    catalog_out = merge_gamma_catalog_with_event_id(DATA_ROOT, merged_dir)

    # 2) Crear carpetas para HypoXPy dentro de merged/
    indir = merged_dir / "input"
    outdir = merged_dir / "output"
    indir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # 3) Copiar station list (primer chunk REAL de este run)
    src_station = first_chunk_dir / "stations.json"
    if not src_station.exists():
        raise FileNotFoundError(f"No existe stations.json en el primer chunk del run: {src_station}")

    dst_station = indir / "GAMMA_station_list.json"
    shutil.copy2(src_station, dst_station)

    print("\n✅ Merge terminado (sin renombrar CSV).")
    print(f" - Picks merged  : {picks_out}")
    print(f" - Catalog merged: {catalog_out}")
    print(f" - Station list  : {dst_station}")
    print(f" - HypoXPy work dirs: {indir}  |  {outdir}")


# =================================================================
# NUEVO: RUN LOCATION BLOCK (HYPoxPY)
# =================================================================

def run_location_block():
    """
    Llama QQ_location_hypoxpy.py como subproceso.
    Depende de que el merge ya haya generado:
      data/merged/gammacatalog_id.csv
      data/merged/gammapicks_id.csv
      data/merged/input/GAMMA_station_list.json
    """
    merged_dir = DATA_ROOT / "merged"

    required = [
        merged_dir / "gammacatalog_id.csv",
        merged_dir / "gammapicks_id.csv",
        merged_dir / "input" / f"{LOCATION_NAMEBASE}_station_list.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan inputs para HypoXPy. No puedo correr localización.\n"
            + "\n".join([f" - {p}" for p in missing])
        )

    cmd = [
        sys.executable,
        str(SCRIPTS["location"]),
        "--binpath", str(LOCATION_BINPATH),
        "--namebase", str(LOCATION_NAMEBASE),
    ] + list(LOCATION_EXTRA_ARGS)

    print("\n" + "=" * 50)
    print("Running: HypoXPy relocation (HypoInverse + HypoDD)")
    print("CMD:", " ".join(cmd))
    print("=" * 50 + "\n")

    ok = run_step(cmd, "HypoXPy relocation", cwd=BASE_DIR)
    if not ok:
        raise RuntimeError("Falló el step de HypoXPy relocation.")


# =================================================================
# MAIN
# =================================================================

def main():
    current = datetime.fromisoformat(START)
    end_time = datetime.fromisoformat(END)

    first_chunk_dir = None  # primer chunk exitoso del run

    while current < end_time:
        window_end = min(current + timedelta(hours=HOUR_STEP), end_time)
        date_str = current.strftime("%Y%m%dT%H%M%S")
        output_dir = DATA_ROOT / date_str

        print(f"\n{'='*50}\nProcessing: {current} - {window_end}\n{'='*50}")

        ok = process_window(current, window_end, output_dir)
        if ok:
            print(f"Completed: {output_dir}")
            if first_chunk_dir is None:
                first_chunk_dir = output_dir

        current = window_end

    # Merge oficial
    if RUN_MERGE_GAMMA:
        if first_chunk_dir is None:
            raise RuntimeError("No hubo ningún chunk exitoso; no puedo hacer merge ni copiar stations.json.")
        run_merge_block(first_chunk_dir)

    # NUEVO: HypoXPy relocation
    if RUN_LOCATION:
        run_location_block()


if __name__ == "__main__":
    main()
