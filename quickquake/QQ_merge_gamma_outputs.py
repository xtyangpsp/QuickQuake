#!/usr/bin/env python3
"""
Merge GaMMA outputs across chunks:
  - merge gamma_picks.csv   -> gammapicks_id.csv (adds event_id)
  - merge gamma_catalog.csv -> gammacatalog_id.csv (adds event_id)
  - merge stations.json (unique keys, same format) -> merged/input/GAMMA_station_list.json

Creates:
  data/merged/
    gammapicks_id.csv
    gammacatalog_id.csv
    input/
      GAMMA_station_list.json
    output/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd


# ============================================================
# Merge: gamma_picks.csv -> gammapicks_id.csv
# ============================================================

def merge_gamma_picks_with_event_id(base_dir: Path, out_dir: Path) -> Path:
    pattern = "**/gamma_picks.csv"
    out_path = out_dir / "gammapicks_id.csv"

    files = sorted(base_dir.glob(pattern))
    print(f"[merge picks] Encontré {len(files)} archivos gamma_picks.csv")

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
            print(f"[merge picks][WARN] Error leyendo {f}: {e}")

    if not frames:
        raise RuntimeError("No se pudo leer ningún gamma_picks.csv. Revisa base_dir y estructura de carpetas.")

    merged = pd.concat(frames, ignore_index=True)
    n_unq = merged["event_id"].nunique(dropna=True)
    print(f"[merge picks] Total picks: {len(merged)} | event_id únicos: {n_unq}")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[merge picks] Guardado: {out_path}")
    return out_path


# ============================================================
# Merge: gamma_catalog.csv -> gammacatalog_id.csv
# ============================================================

def merge_gamma_catalog_with_event_id(base_dir: Path, out_dir: Path) -> Path:
    pattern = "**/gamma_catalog.csv"
    out_path = out_dir / "gammacatalog_id.csv"

    files = sorted(base_dir.glob(pattern))
    print(f"[merge catalog] Encontré {len(files)} archivos gamma_catalog.csv")

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
            print(f"[merge catalog][WARN] Error leyendo {f}: {e}")

    if not frames:
        raise RuntimeError("No se pudo leer ningún gamma_catalog.csv. Revisa base_dir y estructura de carpetas.")

    merged = pd.concat(frames, ignore_index=True)
    n_unq = merged["event_id"].nunique(dropna=True)
    print(f"[merge catalog] Total eventos (filas): {len(merged)} | event_id únicos: {n_unq}")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[merge catalog] Guardado: {out_path}")
    return out_path


# ============================================================
# Merge: stations.json (unique by key, same dict format)
# ============================================================

def merge_stations_json_unique(
    base_dir: Path,
    out_json: Path,
    pattern: str = "**/stations.json",
    prefer: str = "first",          # "first" o "last"
    check_conflicts: bool = True,
    tol: float = 1e-6
) -> Tuple[Path, int, int, List[str]]:

    files = sorted(base_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No encontré stations.json con patrón '{pattern}' dentro de {base_dir}")

    merged: Dict[str, Any] = {}
    conflicts: List[str] = []

    def nearly_equal(a, b) -> bool:
        try:
            return abs(float(a) - float(b)) <= tol
        except Exception:
            return a == b

    def same_station_meta(old: Dict[str, Any], new: Dict[str, Any]) -> bool:
        keys_to_check = ["longitude", "latitude", "elevation(m)", "unit", "component", "response"]
        for k in keys_to_check:
            if k not in old and k not in new:
                continue
            if k not in old or k not in new:
                return False

            if isinstance(old[k], list) or isinstance(new[k], list):
                if old[k] != new[k]:
                    return False
            else:
                if not nearly_equal(old[k], new[k]):
                    return False
        return True

    for f in files:
        try:
            d = json.loads(f.read_text())
        except Exception as e:
            conflicts.append(f"[WARN] No pude leer {f}: {e}")
            continue

        if not isinstance(d, dict):
            conflicts.append(f"[WARN] {f} no es un dict JSON. Saltando.")
            continue

        for sid, meta in d.items():
            if sid not in merged:
                merged[sid] = meta
                continue

            if check_conflicts and not same_station_meta(merged[sid], meta):
                conflicts.append(f"[CONFLICT] {sid} difiere. Mantengo '{prefer}'. Archivo: {f}")
                if prefer == "last":
                    merged[sid] = meta
            else:
                if prefer == "last":
                    merged[sid] = meta

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged, indent=2, sort_keys=True))

    print(f"[merge stations] Encontré {len(files)} stations.json")
    print(f"[merge stations] Estaciones únicas: {len(merged)}")
    print(f"[merge stations] Guardado: {out_json}")

    if conflicts:
        print(f"[merge stations][WARN] Conflictos/avisos: {len(conflicts)} (muestro hasta 10)")
        for msg in conflicts[:10]:
            print("  " + msg)

    return out_json, len(files), len(merged), conflicts


# ============================================================
# Orquestador del merge (lo que llama RunAll)
# ============================================================

def run_merge_block(
    data_root: Path,
    merged_dir: Optional[Path] = None,
    namebase: str = "GAMMA",
) -> Tuple[Path, Path, Path]:
    """
    Produce:
      merged_dir/gammapicks_id.csv
      merged_dir/gammacatalog_id.csv
      merged_dir/input/{namebase}_station_list.json
    """

    data_root = data_root.resolve()
    merged_dir = (merged_dir or (data_root / "merged")).resolve()

    merged_dir.mkdir(parents=True, exist_ok=True)
    indir = merged_dir / "input"
    outdir = merged_dir / "output"
    indir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Merge CSVs con event_id
    picks_out = merge_gamma_picks_with_event_id(data_root, merged_dir)
    catalog_out = merge_gamma_catalog_with_event_id(data_root, merged_dir)

    # 2) Merge stations único (SIN duplicar, mismo formato)
    station_out = indir / f"{namebase}_station_list.json"
    merge_stations_json_unique(
        base_dir=data_root,
        out_json=station_out,
        pattern="**/stations.json",
        prefer="first",
        check_conflicts=True,
    )

    print("\n✅ Merge terminado.")
    print(f" - Picks merged   : {picks_out}")
    print(f" - Catalog merged : {catalog_out}")
    print(f" - Station list   : {station_out}")
    print(f" - HypoXPy dirs   : {indir} | {outdir}")

    return picks_out, catalog_out, station_out


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Merge GaMMA outputs across chunks and prepare HypoXPy inputs.")
    p.add_argument("--data_root", type=str, required=True,
                   help="Directorio base donde están los chunks (ej: .../QuickQuake/data)")
    p.add_argument("--merged_dir", type=str, default=None,
                   help="Directorio destino merged (default: <data_root>/merged)")
    p.add_argument("--namebase", type=str, default="GAMMA",
                   help="Prefijo para station_list.json (default: GAMMA)")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    merged_dir = Path(args.merged_dir) if args.merged_dir else None
    run_merge_block(data_root=data_root, merged_dir=merged_dir, namebase=args.namebase)


if __name__ == "__main__":
    main()
