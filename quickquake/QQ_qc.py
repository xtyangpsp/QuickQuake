#!/usr/bin/env python3
"""
QQ_qc.py

QC por curva de energía vs velocidad usando percentiles:
pc_ratio_energy = P(signal_percentile) / P(noise_percentile)

Versión optimizada pero fiel al original:
- Mantiene la lógica científica del original
- NO cambia el cálculo del ratio
- NO cambia el preprocesado por evento
- Acelera indexando picks por event_id
- Reusa v_grid
- Reusa stream cache por bloque de archivos
- Separa eventos calculados de eventos no calculados

- Input:
    merged/output/{namebase}_hypodd_catalog.csv
    merged/input/{namebase}_picks_cleaned.csv
    merged/input/{namebase}_station_list.json

- Output:
    1) merged/output/{namebase}_hypodd_catalog_qcfiltered.csv
       (catálogo base de HypoDD + pc_ratio_energy, solo eventos que pasaron)

    2) merged/output/{namebase}_hypodd_catalog_qcrejected.csv
       (catálogo base de HypoDD + pc_ratio_energy, solo eventos con ratio calculado que NO pasaron)

    3) merged/output/{namebase}_hypodd_catalog_qcskipped.csv
       (catálogo base de HypoDD + columnas QC, eventos para los que NO se pudo calcular ratio)

    4) (opcional) merged/output/qc_plots_{namebase}/qc_eventXXXX_YYYYmmddTHHMMSS.png
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import re
from datetime import datetime
from bisect import bisect_right
import argparse

import numpy as np
import pandas as pd
import obspy
from obspy import read, UTCDateTime
from obspy.signal.filter import envelope
from obspy.geodetics.base import gps2dist_azimuth


# ============================================================
# FIXED DEFAULTS (NOT user-controlled)
# ============================================================

PRE_S = 10.0
POST_S = 40.0

FREQMIN = 1.0
FREQMAX = 15.0

VMIN_SIGNAL = 2.0
VMAX_SIGNAL = 8.0

PAD_NEIGHBORS = True
GEOM_CORRECTION = False
EPS = 1e-12

MAX_EVENTS = 0  # 0 = all (no user control)

MIN_TOTAL_STATIONS = 4

CHUNK_RE = re.compile(r"^\d{8}T\d{6}$")


# -------------------------
# Chunk index
# -------------------------

def build_chunk_index(data_root: Path):
    chunks = []
    for p in data_root.iterdir():
        if p.is_dir() and CHUNK_RE.match(p.name):
            dt = datetime.strptime(p.name, "%Y%m%dT%H%M%S")
            chunks.append((UTCDateTime(dt), p))
    chunks.sort(key=lambda x: x[0])
    if not chunks:
        raise RuntimeError(f"No chunk folders found in {data_root}")
    return [t for t, _ in chunks], [d for _, d in chunks]


def chunk_idx_for_time(t: UTCDateTime, times: List[UTCDateTime]) -> int:
    i = bisect_right(times, t) - 1
    return 0 if i < 0 else (len(times) - 1 if i >= len(times) else i)


def chunk_dirs_for_window(
    t_start: UTCDateTime,
    t_end: UTCDateTime,
    times: List[UTCDateTime],
    dirs: List[Path],
    pad_neighbors: bool = True,
) -> List[Path]:
    i0 = chunk_idx_for_time(t_start, times)
    i1 = chunk_idx_for_time(t_end, times)
    a, b = i0, i1
    if pad_neighbors:
        a = max(0, a - 1)
        b = min(len(dirs) - 1, b + 1)
    return dirs[a:b + 1]


def waveform_files_for_window(
    t_start: UTCDateTime,
    t_end: UTCDateTime,
    times: List[UTCDateTime],
    dirs: List[Path],
    pad_neighbors: bool = True,
) -> List[Path]:
    chunk_dirs = chunk_dirs_for_window(t_start, t_end, times, dirs, pad_neighbors=pad_neighbors)
    files, seen = [], set()
    for cd in chunk_dirs:
        wdir = cd / "waveforms"
        if not wdir.exists():
            continue
        for fp in sorted(wdir.glob("*.mseed")):
            if fp not in seen:
                seen.add(fp)
                files.append(fp)
    return files


# -------------------------
# Stream cache
# -------------------------

class StreamCache:
    def __init__(self):
        self._key = None
        self._st = None

    def get_stream(
        self,
        t_start: UTCDateTime,
        t_end: UTCDateTime,
        times: List[UTCDateTime],
        dirs: List[Path],
        pad_neighbors: bool = True,
    ):
        files = waveform_files_for_window(t_start, t_end, times, dirs, pad_neighbors=pad_neighbors)
        if not files:
            return None, []

        key = tuple(str(f) for f in files)
        if self._key == key and self._st is not None:
            return self._st, list(files)

        st = None
        for fp in files:
            try:
                s = read(str(fp))
                st = s if st is None else (st + s)
            except Exception:
                pass

        if st is None:
            self._key = None
            self._st = None
            return None, list(files)

        st.merge(fill_value="interpolate")
        self._key = key
        self._st = st
        return st, list(files)


# -------------------------
# Helpers
# -------------------------

def stations_by_hypo_distance_km(
    stations: pd.DataFrame,
    ev_lat: float,
    ev_lon: float,
    ev_depth_km: float,
):
    hypo_km = []
    for _, r in stations.iterrows():
        d_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, float(r["lat"]), float(r["lon"]))
        horiz_km = d_m / 1000.0
        vert_km = float(ev_depth_km) + float(r.get("elev_km", 0.0))
        hypo_km.append(float(np.hypot(horiz_km, vert_km)))

    tmp = stations.copy()
    tmp["hypo_km"] = hypo_km
    tmp = tmp.sort_values("hypo_km")
    return dict(zip(tmp["id"], tmp["hypo_km"]))


def best_Z_channel_for_station(st: obspy.Stream, net_sta: str) -> Optional[str]:
    prefs = ["BHZ", "EHZ", "SHZ", "HHZ"]
    chans = {tr.stats.channel for tr in st if f"{tr.stats.network}.{tr.stats.station}" == net_sta}
    for c in prefs:
        if c in chans:
            return c
    return None


def window_energy_stat(tr: obspy.Trace, t_abs: UTCDateTime, win_len_s: float, energy_type: str = "squared_median",) -> float:
    sr = float(tr.stats.sampling_rate)
    n = tr.data.size
    if n == 0 or not np.isfinite(sr) or sr <= 0:
        return 0.0

    half = 0.5 * float(win_len_s)
    t0_win = t_abs - half

    i0 = int((t0_win - tr.stats.starttime) * sr)
    nsamp = int(max(1, round(float(win_len_s) * sr)))
    i1 = i0 + nsamp

    a = max(i0, 0)
    b = min(i1, n)
    if b <= a:
        return 0.0

    x = tr.data[a:b].astype(np.float64, copy=False)
    if x.size == 0:
        return 0.0

    if energy_type == "squared_median":
        val = np.median(x ** 2)

    elif energy_type == "envelope_median":
        env = envelope(x)
        val = np.median(env)

    else:
        raise ValueError(
            f"Unsupported energy_type='{energy_type}'. "
            "Use 'squared_median' or 'envelope_median'."
        )

    if not np.isfinite(val) or val <= 0.0:
        return 0.0

    return float(val)


def safe_percentile(x: np.ndarray, p: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, float(p)))


def percentile_rank(value: float, sample: np.ndarray) -> float:
    s = sample[np.isfinite(sample)]
    if s.size == 0 or not np.isfinite(value):
        return float("nan")
    return float(100.0 * np.mean(s <= float(value)))


def build_event_pick_index(picks_clean: pd.DataFrame):
    event_to_station_ids: Dict[int, List[str]] = {}
    grp = picks_clean.groupby("event_id_mapped", sort=False)["id"].unique()
    for eid, ids in grp.items():
        try:
            event_to_station_ids[int(eid)] = [str(x) for x in ids if pd.notna(x)]
        except Exception:
            continue
    return event_to_station_ids


def build_skip_result(eid: int, reason: str):
    return {
        "event_id": int(eid),
        "pc_ratio_energy": np.nan,
        "qc_status": "skipped",
        "qc_reason": str(reason),
    }


# -------------------------
# Paths + IO
# -------------------------

def resolve_paths(data_root: Path, namebase: str):
    merged_in = data_root / "merged" / "input"
    merged_out = data_root / "merged" / "output"

    picks_csv = merged_in / f"{namebase}_picks_cleaned.csv"
    stations_json = merged_in / f"{namebase}_station_list.json"

    catalog_csv = merged_out / f"{namebase}_hypodd_catalog.csv"
    if not catalog_csv.exists():
        raise FileNotFoundError(f"Expected HypoDD catalog not found: {catalog_csv}")

    qc_filtered_catalog_csv = merged_out / f"{namebase}_hypodd_catalog_qcfiltered.csv"
    qc_rejected_catalog_csv = merged_out / f"{namebase}_hypodd_catalog_qcrejected.csv"
    qc_skipped_catalog_csv = merged_out / f"{namebase}_hypodd_catalog_qcskipped.csv"

    return (
        catalog_csv,
        picks_csv,
        stations_json,
        qc_filtered_catalog_csv,
        qc_rejected_catalog_csv,
        qc_skipped_catalog_csv,
    )


def load_inputs(catalog_csv: Path, picks_csv: Path, stations_json: Path):
    catalog_raw = pd.read_csv(catalog_csv)

    catalog = catalog_raw.copy()
    picks = pd.read_csv(picks_csv)

    catalog["time"] = pd.to_datetime(catalog["time"], utc=True, errors="coerce", format="mixed")
    picks["timestamp"] = pd.to_datetime(picks["timestamp"], utc=True, errors="coerce", format="mixed")

    catalog = catalog.dropna(subset=["time"]).reset_index(drop=True)
    picks = picks.dropna(subset=["timestamp"]).reset_index(drop=True)

    catalog["event_id"] = pd.to_numeric(catalog["event_id"], errors="coerce")
    picks["event_id_mapped"] = pd.to_numeric(picks["event_id_mapped"], errors="coerce")

    catalog = catalog.dropna(subset=["event_id"]).reset_index(drop=True)
    picks = picks.dropna(subset=["event_id_mapped"]).reset_index(drop=True)

    station_df = pd.read_json(stations_json).T
    station_df.index.name = "id"
    station_df["id"] = station_df.index.astype(str)
    station_df["elev_km"] = station_df.get("elevation(m)", 0.0).astype(float) / 1000.0

    stations = (
        station_df.rename(columns={"latitude": "lat", "longitude": "lon"})
        .reset_index(drop=True)[["id", "lat", "lon", "elev_km"]]
        .dropna(subset=["lat", "lon"])
        .drop_duplicates(subset=["id"])
        .reset_index(drop=True)
    )

    return catalog_raw, catalog, picks, stations


# -------------------------
# QC core (single event)
# -------------------------

def qc_one_event(
    row_event,
    event_to_station_ids: Dict[int, List[str]],
    stations: pd.DataFrame,
    times: List[UTCDateTime],
    dirs: List[Path],
    cache: StreamCache,
    *,
    v_grid: np.ndarray,
    winlen: float,
    energy_type: str,
    noise_percentile: float,
    signal_percentile: float,
    make_plot: bool,
    plot_dir: Path,
    min_total_stations: int,
    min_valid_stations_per_v: int,
):
    if not (0.0 < float(noise_percentile) < 100.0 and 0.0 < float(signal_percentile) < 100.0):
        raise ValueError("noise_percentile y signal_percentile deben estar en (0, 100).")
    if float(signal_percentile) <= float(noise_percentile):
        raise ValueError("signal_percentile debe ser > noise_percentile.")

    t0 = UTCDateTime(row_event["time"].to_pydatetime())
    ev_lat = float(row_event["latitude"])
    ev_lon = float(row_event["longitude"])
    ev_depth_km = float(row_event.get("depth_km", 0.0))
    eid = int(row_event["event_id"])

    t_start = t0 - float(PRE_S)
    t_end = t0 + float(POST_S)

    stations_with_picks = event_to_station_ids.get(eid, [])
    if len(stations_with_picks) == 0:
        return build_skip_result(eid, "no_picks")

    if len(stations_with_picks) < int(min_total_stations):
        return build_skip_result(eid, "too_few_stations_with_picks")

    dist_map_all = stations_by_hypo_distance_km(stations, ev_lat, ev_lon, ev_depth_km)
    dist_map = {sid: dist_map_all[sid] for sid in stations_with_picks if sid in dist_map_all}
    if len(dist_map) < int(min_total_stations):
        return build_skip_result(eid, "too_few_stations_in_metadata")

    st_raw, files = cache.get_stream(t_start, t_end, times, dirs, pad_neighbors=PAD_NEIGHBORS)
    if st_raw is None:
        return build_skip_result(eid, "no_waveforms_loaded")

    prepped: Dict[str, Tuple[obspy.Trace, float]] = {}
    for sid, dkm in dist_map.items():
        parts = sid.split(".")
        if len(parts) < 2:
            continue
        net, sta = parts[0], parts[1]

        chz = best_Z_channel_for_station(st_raw, f"{net}.{sta}")
        if chz is None:
            continue

        tr_list = st_raw.select(network=net, station=sta, channel=chz)
        if not tr_list:
            continue

        tr = tr_list[0].copy()
        tr.trim(t_start, t_end, pad=True, fill_value=0)
        tr.detrend("demean")
        tr.taper(max_percentage=0.05, type="cosine")

        sr = float(tr.stats.sampling_rate)
        nyq = 0.5 * sr
        fmin = max(0.001, min(float(FREQMIN), nyq * 0.99))
        fmax = max(fmin + 0.001, min(float(FREQMAX), nyq * 0.99))
        if fmax > fmin:
            tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

        prepped[sid] = (tr, float(dkm))

    if len(prepped) < int(min_total_stations):
        return build_skip_result(eid, "too_few_prepped_stations")

    req = int(min(min_valid_stations_per_v, len(prepped)))
    energies = np.full_like(v_grid, np.nan, dtype=np.float64)
    half = 0.5 * float(winlen)

    for i, v_test in enumerate(v_grid):
        inv_v = 1.0 / max(float(v_test), 1e-6)
        Ev_sum = 0.0
        n_valid = 0

        for _, (tr, dkm) in prepped.items():
            t_abs = t0 + (dkm * inv_v)

            if (t_abs - half) < t_start or (t_abs + half) > t_end:
                continue

            
            E = window_energy_stat(
                tr,
                t_abs,
                win_len_s=float(winlen),
                energy_type=energy_type,
            )
            if not np.isfinite(E) or E <= 0.0:
                continue

            if GEOM_CORRECTION:
                E *= (dkm ** 2)

            Ev_sum += E
            n_valid += 1

        if n_valid >= req:
            energies[i] = Ev_sum / float(n_valid)

    if not np.any(np.isfinite(energies)) or float(np.nanmax(energies)) <= 0.0:
        return build_skip_result(eid, "no_valid_energy_curve")

    sig_mask = (v_grid >= float(VMIN_SIGNAL)) & (v_grid <= float(VMAX_SIGNAL)) & np.isfinite(energies)
    if not np.any(sig_mask):
        return build_skip_result(eid, "no_valid_signal_band")

    # Mantener exactamente la lógica del original
    e_noise = safe_percentile(energies, float(noise_percentile))
    e_signal = safe_percentile(energies, float(signal_percentile))
    if not np.isfinite(e_noise) or e_noise <= 0.0:
        return build_skip_result(eid, "invalid_noise_percentile")
    if not np.isfinite(e_signal) or e_signal <= 0.0:
        return build_skip_result(eid, "invalid_signal_percentile")

    ratio = float(e_signal / (e_noise + float(EPS)))

    if make_plot:
        i_peak = int(np.nanargmax(np.where(sig_mask, energies, -np.inf)))
        best_v = float(v_grid[i_peak])

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(v_grid, energies, "k-", lw=2)

        highlight_w = 0.15
        x0 = best_v - 0.5 * highlight_w
        x1 = best_v + 0.5 * highlight_w
        ax.axvspan(x0, x1, color="red", alpha=0.18, zorder=0)
        ax.axvline(best_v, color="red", lw=1.2, alpha=0.8)

        ax.axhline(e_noise, color="tab:orange", lw=2.0, label=f"P{noise_percentile:.0f}={e_noise:.2e}")
        ax.axhline(e_signal, color="tab:cyan", lw=2.0, label=f"P{signal_percentile:.0f}={e_signal:.2e}")

        ax.set_title(
            f"event_id={eid} | {t0.strftime('%Y-%m-%d %H:%M:%S')} | "
            f"P{signal_percentile:.0f}/P{noise_percentile:.0f}={ratio:.2f}"
        )
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Stacked Energy")
        ax.grid(ls="-", alpha=0.15)
        ax.legend()
        plt.tight_layout()

        plot_dir.mkdir(parents=True, exist_ok=True)
        out_png = plot_dir / f"qc_event{eid}_{t0.strftime('%Y%m%dT%H%M%S')}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

    return {
        "event_id": eid,
        "pc_ratio_energy": float(ratio),
        "qc_status": "computed",
        "qc_reason": "computed",
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="QuickQuake QC: velocity energy curve + percentile ratio filter")

    ap.add_argument("--data_root", required=True, type=str, help="QuickQuake data root (contains merged/)")
    ap.add_argument("--namebase", required=True, type=str, help="Namebase, e.g., GAMMA")

    ap.add_argument("--vmin_curve", type=float, default=2.0)
    ap.add_argument("--vmax_curve", type=float, default=8.0)
    ap.add_argument("--vsteps_curve", type=int, default=150)
    ap.add_argument("--winlen", type=float, default=0.5)
    ap.add_argument(
        "--energy_type",
        type=str,
        default="squared_median",
        choices=["squared_median", "envelope_median"],
        help="Per-trace energy definition within the short window around the predicted arrival.",
    )
    ap.add_argument("--noise_percentile", type=float, default=50.0)
    ap.add_argument("--signal_percentile", type=float, default=90.0)
    ap.add_argument(
        "--min_ratio",
        type=float,
        default=2.0,
        help="Keep events with pc_ratio_energy >= min_ratio",
    )

    ap.add_argument("--make_plot", action="store_true", help="Save QC plots to merged/output/qc_plots_{namebase}/")
    ap.add_argument(
        "--max_plots",
        type=int,
        default=50,
        help="Max number of plots to save when --make_plot is enabled (default: 50)",
    )
    ap.add_argument(
        "--min_total_stations",
        type=int,
        default=MIN_TOTAL_STATIONS,
        help="Minimum number of unique stations required to run QC for an event (event gating).",
    )
    ap.add_argument(
        "--min_valid_stations_per_v",
        type=int,
        default=None,
        help="Minimum number of stations contributing at each test velocity. "
             "If not provided, defaults to --min_total_stations.",
    )

    args = ap.parse_args()
    min_total_stations = int(args.min_total_stations)
    min_valid_stations_per_v = (
        int(args.min_valid_stations_per_v)
        if args.min_valid_stations_per_v is not None
        else min_total_stations
    )

    data_root = Path(args.data_root)
    namebase = str(args.namebase)

    (
        catalog_csv,
        picks_csv,
        stations_json,
        qc_filtered_catalog_csv,
        qc_rejected_catalog_csv,
        qc_skipped_catalog_csv,
    ) = resolve_paths(data_root, namebase)

    catalog_raw, catalog, picks_clean, stations = load_inputs(catalog_csv, picks_csv, stations_json)
    event_to_station_ids = build_event_pick_index(picks_clean)

    times, dirs = build_chunk_index(data_root)
    cache = StreamCache()

    plot_dir = data_root / "merged" / "output" / f"qc_plots_{namebase}"
    plots_left = int(args.max_plots)
    v_grid = np.flip(np.linspace(float(args.vmin_curve), float(args.vmax_curve), int(args.vsteps_curve)))

    rows = []
    n_total = len(catalog) if int(MAX_EVENTS) <= 0 else min(len(catalog), int(MAX_EVENTS))

    for i in range(n_total):
        row_event = catalog.iloc[i]
        do_plot = bool(args.make_plot) and (plots_left > 0)

        
        r = qc_one_event(
            row_event,
            event_to_station_ids,
            stations,
            times,
            dirs,
            cache,
            v_grid=v_grid,
            winlen=args.winlen,
            energy_type=args.energy_type,
            noise_percentile=args.noise_percentile,
            signal_percentile=args.signal_percentile,
            make_plot=do_plot,
            plot_dir=plot_dir,
            min_total_stations=min_total_stations,
            min_valid_stations_per_v=min_valid_stations_per_v,
        )

        rows.append(r)

        if do_plot and r.get("qc_status") == "computed":
            plots_left -= 1

        if (i + 1) % 500 == 0 or (i + 1) == n_total:
            print(f"[QC] processed {i + 1}/{n_total} events")

    qc_df = pd.DataFrame(rows)

    raw_out = catalog_raw.copy()
    raw_out["_event_id_num"] = pd.to_numeric(raw_out["event_id"], errors="coerce").astype("Int64")

    if not qc_df.empty:
        qc_merge = qc_df.copy()
        qc_merge["event_id"] = pd.to_numeric(qc_merge["event_id"], errors="coerce").astype("Int64")
        qc_merge["pc_ratio_energy"] = pd.to_numeric(qc_merge["pc_ratio_energy"], errors="coerce").round(2)

        raw_out = raw_out.merge(
            qc_merge[["event_id", "pc_ratio_energy", "qc_status", "qc_reason"]],
            left_on="_event_id_num",
            right_on="event_id",
            how="left",
            suffixes=("", "_qc"),
        )
    else:
        raw_out["pc_ratio_energy"] = np.nan
        raw_out["qc_status"] = "skipped"
        raw_out["qc_reason"] = "no_qc_rows_produced"

    if "event_id_qc" in raw_out.columns:
        raw_out = raw_out.drop(columns=["event_id_qc"])

    if "event_id_y" in raw_out.columns:
        raw_out = raw_out.drop(columns=["event_id_y"])

    if "event_id_x" in raw_out.columns:
        raw_out = raw_out.rename(columns={"event_id_x": "event_id"})

    raw_out = raw_out.drop(columns=["_event_id_num"])

    raw_out["qc_status"] = raw_out["qc_status"].fillna("skipped")
    raw_out["qc_reason"] = raw_out["qc_reason"].fillna("missing_after_merge")

    passed_mask = (
        raw_out["pc_ratio_energy"].notna()
        & (raw_out["pc_ratio_energy"] >= float(args.min_ratio))
    )

    rejected_mask = (
        raw_out["pc_ratio_energy"].notna()
        & (raw_out["pc_ratio_energy"] < float(args.min_ratio))
    )

    skipped_mask = raw_out["pc_ratio_energy"].isna()

    filtered = raw_out[passed_mask].copy()
    rejected = raw_out[rejected_mask].copy()
    skipped = raw_out[skipped_mask].copy()

    # Quitar columnas de diagnóstico en filtered y rejected
    cols_to_drop_final = [c for c in ["qc_status", "qc_reason"] if c in filtered.columns]
    if cols_to_drop_final:
        filtered = filtered.drop(columns=cols_to_drop_final)

    cols_to_drop_final = [c for c in ["qc_status", "qc_reason"] if c in rejected.columns]
    if cols_to_drop_final:
        rejected = rejected.drop(columns=cols_to_drop_final)

    filtered.to_csv(qc_filtered_catalog_csv, index=False)
    rejected.to_csv(qc_rejected_catalog_csv, index=False)
    skipped.to_csv(qc_skipped_catalog_csv, index=False)

    print(f"\nWrote filtered catalog: {qc_filtered_catalog_csv}")
    print(f"Wrote rejected catalog: {qc_rejected_catalog_csv}")
    print(f"Wrote skipped catalog:  {qc_skipped_catalog_csv}")

    if bool(args.make_plot):
        print(f"Wrote up to {int(args.max_plots)} plots in: {plot_dir}")

    print(f"\nKept {len(filtered)} / {len(raw_out)} events with pc_ratio_energy >= {args.min_ratio}")
    print(f"Rejected by ratio {len(rejected)} / {len(raw_out)} events")
    print(f"Skipped (no QC computed) {len(skipped)} / {len(raw_out)} events")

    if len(skipped) > 0:
        print("\nTop skipped reasons:")
        print(skipped["qc_reason"].value_counts().to_string())


if __name__ == "__main__":
    main()