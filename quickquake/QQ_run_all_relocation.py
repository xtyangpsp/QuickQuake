import subprocess
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# =================================================================
# CONFIGURATION
# =================================================================

START = "2021-09-19T15:00:00"
END = "2021-09-19T17:00:00"
HOUR_STEP = 1

BASE_DIR = Path(__file__).parent.parent
DATA_ROOT = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "dependencies/PhaseNet/model/190703-214543"

CENTER = (-161.8903, 55.4133)
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
    # "location": BASE_DIR / "quickquake/QQ_location.py"  # ya no se usa aquí
}

RUN_CONFIG = True
RUN_DL = True
RUN_PHASENET = True
RUN_GAMMA = True

# Nuevo: merge oficial (TU merge)
RUN_MERGE_GAMMA = True

# =================================================================
# HELPERS
# =================================================================

def run_step(cmd, step):
    try:
        subprocess.run(cmd, check=True)
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
            if not run_step(cmd, name):
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


def run_merge_block():
    """
    Corre el merge oficial de GaMMA y guarda en data/_MERGED/
    """
    out_dir = DATA_ROOT / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    picks_out = merge_gamma_picks_with_event_id(DATA_ROOT, out_dir)
    catalog_out = merge_gamma_catalog_with_event_id(DATA_ROOT, out_dir)

    print("\nMerge terminado.")
    print(f" - Picks  : {picks_out}")
    print(f" - Catalog: {catalog_out}")


# =================================================================
# MAIN
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
            print(f"Completed: {output_dir}")

        current = window_end

    # FUERA DEL LOOP: merge oficial (TU merge)
    if RUN_MERGE_GAMMA:
        run_merge_block()

    # Próximo paso (después): aquí vendrá HypoXPy relocate() usando esos 2 CSV

if __name__ == "__main__":
    main()
