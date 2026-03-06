#!/usr/bin/env python3
"""
QQ_qc.py

QC por curva de energía vs velocidad usando percentiles:
Percentile_energy_ratio = P(signal_percentile) / P(noise_percentile)

- Input:
    merged/output/{namebase}_hypodd_catalog.csv
    merged/input/{namebase}_picks_cleaned.csv
    merged/input/{namebase}_station_list.json
- Output:
    1) merged/output/{namebase}_qc_metrics.csv
    2) merged/output/{namebase}_hypodd_catalog_qcfiltered.csv
       (mismo HypoDD + 1 columna extra, filtrado)
    3) (opcional) merged/output/qc_plots_{namebase}/qc_eventXXXX_YYYYmmddTHHMMSS.png
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
    return dirs[a : b + 1]


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


def window_l2_energy_fast(tr: obspy.Trace, t_abs: UTCDateTime, win_len_s: float) -> float:
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
    return float(np.dot(x, x))


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

    qc_metrics_csv = merged_out / f"{namebase}_qc_metrics.csv"
    qc_filtered_catalog_csv = merged_out / f"{namebase}_hypodd_catalog_qcfiltered.csv"

    return catalog_csv, picks_csv, stations_json, qc_metrics_csv, qc_filtered_catalog_csv


def load_inputs(catalog_csv: Path, picks_csv: Path, stations_json: Path):
    catalog = pd.read_csv(catalog_csv)
    picks = pd.read_csv(picks_csv)

    catalog["time"] = pd.to_datetime(catalog["time"], utc=True, errors="coerce")
    picks["timestamp"] = pd.to_datetime(picks["timestamp"], utc=True, errors="coerce")

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

    return catalog, picks, stations


# -------------------------
# QC core (single event)
# -------------------------

def qc_one_event(
    row_event,
    picks_clean: pd.DataFrame,
    stations: pd.DataFrame,
    times: List[UTCDateTime],
    dirs: List[Path],
    cache: StreamCache,
    *,
    vmin_curve: float,
    vmax_curve: float,
    vsteps_curve: int,
    winlen: float,
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

    ev_picks = picks_clean[picks_clean["event_id_mapped"] == eid]
    if ev_picks.empty:
        return None

    stations_with_picks = ev_picks["id"].unique().tolist()
    if len(stations_with_picks) < int(min_total_stations):
        return None

    dist_map_all = stations_by_hypo_distance_km(stations, ev_lat, ev_lon, ev_depth_km)
    dist_map = {sid: dist_map_all[sid] for sid in stations_with_picks if sid in dist_map_all}
    if len(dist_map) < int(min_total_stations):
        return None

    st_raw, files = cache.get_stream(t_start, t_end, times, dirs, pad_neighbors=PAD_NEIGHBORS)
    if st_raw is None:
        return None

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
        return None
    req = int(min(min_valid_stations_per_v, len(prepped))) 

    v_grid = np.flip(np.linspace(float(vmin_curve), float(vmax_curve), int(vsteps_curve)))
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

            E = window_l2_energy_fast(tr, t_abs, win_len_s=float(winlen))
            if not np.isfinite(E) or E <= 0.0:
                continue

            if GEOM_CORRECTION:
                E *= (dkm ** 2)

            Ev_sum += E
            n_valid += 1

         
        if n_valid >= req:
            energies[i] = Ev_sum / float(n_valid)

    if not np.any(np.isfinite(energies)) or float(np.nanmax(energies)) <= 0.0:
        return None

    sig_mask = (v_grid >= float(VMIN_SIGNAL)) & (v_grid <= float(VMAX_SIGNAL)) & np.isfinite(energies)
    if not np.any(sig_mask):
        return None

    i_peak = int(np.nanargmax(np.where(sig_mask, energies, -np.inf)))
    best_v = float(v_grid[i_peak])
    peak = float(energies[i_peak])
    peak_pctl_rank = percentile_rank(peak, energies)

    e_noise = safe_percentile(energies, float(noise_percentile))
    e_signal = safe_percentile(energies, float(signal_percentile))
    if not np.isfinite(e_noise) or e_noise <= 0.0:
        return None
    if not np.isfinite(e_signal) or e_signal <= 0.0:
        return None

    ratio = float(e_signal / (e_noise + float(EPS)))

    if make_plot:
        import matplotlib
        matplotlib.use("Agg")  # seguro en subprocess / headless
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(v_grid, energies, "k-", lw=2)

        highlight_w = 0.15
        x0 = best_v - 0.5 * highlight_w
        x1 = best_v + 0.5 * highlight_w
        ax.axvspan(x0, x1, color="red", alpha=0.18, zorder=0)
        ax.axvline(best_v, color="red", lw=1.2, alpha=0.8)

        ax.axhline(e_noise,  color="tab:orange", lw=2.0, label=f"P{noise_percentile:.0f}={e_noise:.2e}")
        ax.axhline(e_signal, color="tab:cyan",   lw=2.0, label=f"P{signal_percentile:.0f}={e_signal:.2e}")

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
        "time": row_event["time"],
        "best_v_km_s": best_v,
        "peak_energy": peak,
        "peak_percentile_rank": float(peak_pctl_rank),
        "noise_percentile": float(noise_percentile),
        "noise_energy": float(e_noise),
        "signal_percentile": float(signal_percentile),
        "signal_ref_energy": float(e_signal),
        "Percentile_energy_ratio": float(ratio),
        "winlen_s": float(winlen),
        "num_stations": int(len(prepped)),
        "min_valid_stations_per_v": int(req),
        "min_total_stations": int(min_total_stations),
        "n_files": int(len(files)),
    }


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="QuickQuake QC: velocity energy curve + percentile ratio filter")

    ap.add_argument("--data_root", required=True, type=str, help="QuickQuake data root (contains merged/)")
    ap.add_argument("--namebase", required=True, type=str, help="Namebase, e.g., GAMMA")

    # USER-CONTROLLED knobs (ONLY these)
    ap.add_argument("--vmin_curve", type=float, default=2.0)
    ap.add_argument("--vmax_curve", type=float, default=8.0)
    ap.add_argument("--vsteps_curve", type=int, default=150)
    ap.add_argument("--winlen", type=float, default=0.5)
    ap.add_argument("--noise_percentile", type=float, default=50.0)
    ap.add_argument("--signal_percentile", type=float, default=90.0)
    ap.add_argument("--min_ratio", type=float, default=2.0,
                    help="Keep events with Percentile_energy_ratio >= min_ratio")

    ap.add_argument("--make_plot", action="store_true", help="Save QC plots to merged/output/qc_plots_{namebase}/")
    ap.add_argument("--max_plots", type=int, default=50,
                    help="Max number of plots to save when --make_plot is enabled (default: 50)")
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

    catalog_csv, picks_csv, stations_json, qc_metrics_csv, qc_filtered_catalog_csv = resolve_paths(data_root, namebase)
    catalog, picks_clean, stations = load_inputs(catalog_csv, picks_csv, stations_json)

    times, dirs = build_chunk_index(data_root)
    cache = StreamCache()

    plot_dir = data_root / "merged" / "output" / f"qc_plots_{namebase}"
    plots_left = int(args.max_plots)

    rows = []
    n_total = len(catalog) if int(MAX_EVENTS) <= 0 else min(len(catalog), int(MAX_EVENTS))

    for i in range(n_total):
        row_event = catalog.iloc[i]

        do_plot = bool(args.make_plot) and (plots_left > 0)

        r = qc_one_event(
            row_event,
            picks_clean,
            stations,
            times,
            dirs,
            cache,
            vmin_curve=args.vmin_curve,
            vmax_curve=args.vmax_curve,
            vsteps_curve=args.vsteps_curve,
            winlen=args.winlen,
            noise_percentile=args.noise_percentile,
            signal_percentile=args.signal_percentile,
            make_plot=do_plot,
            plot_dir=plot_dir,
            min_total_stations=min_total_stations,
            min_valid_stations_per_v=min_valid_stations_per_v,
            
        )
        if r is not None:
            rows.append(r)
            if do_plot:
                plots_left -= 1

       

    qc_df = pd.DataFrame(rows)
    qc_df.to_csv(qc_metrics_csv, index=False)

    # merge + filter catalog (append 1 column, then filter)
    cat = catalog.copy()
    cat["event_id"] = pd.to_numeric(cat["event_id"], errors="coerce").astype("Int64")

    if not qc_df.empty:
        qc_df["event_id"] = pd.to_numeric(qc_df["event_id"], errors="coerce").astype("Int64")
        merged = cat.merge(
            qc_df[["event_id", "Percentile_energy_ratio"]],
            on="event_id",
            how="left",
        )
    else:
        merged = cat.copy()
        merged["Percentile_energy_ratio"] = np.nan

    filtered = merged[
        merged["Percentile_energy_ratio"].notna()
        & (merged["Percentile_energy_ratio"] >= float(args.min_ratio))
    ].copy()

    filtered.to_csv(qc_filtered_catalog_csv, index=False)

    print(f"\nWrote QC metrics: {qc_metrics_csv}")
    print(f"Wrote filtered catalog: {qc_filtered_catalog_csv}")
    if bool(args.make_plot):
        print(f"Wrote up to {int(args.max_plots)} plots in: {plot_dir}")

    if not qc_df.empty:
        print("\nQC preview:")
        print(qc_df.head(20).to_string(index=False))
        print(f"\nKept {len(filtered)} / {len(merged)} events with Percentile_energy_ratio >= {args.min_ratio}")
    else:
        print("\nNo QC rows produced (qc_df empty). Check inputs and waveforms.")


if __name__ == "__main__":
    main()
