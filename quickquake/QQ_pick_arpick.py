#!/usr/bin/env python3
"""
QuickQuake - lightweight P/S pick refinement with ObsPy ar_pick

Goal
----
Read merged GaMMA picks from:
    data/merged/gammapicks_id.csv

and write ONE compatible output file, by default:
    data/merged/gammapicks_id_arpick.csv

This script keeps the original column structure as much as possible and only
updates the `timestamp` column for associated P/S picks when a local ar_pick
repick is successful and passes simple sanity thresholds.

Design choices
--------------
- Runs AFTER QQ_merge_gamma_outputs.py and BEFORE QQ_location.py
- Uses existing merged `event_id` convention from merge script
- Processes only associated picks that have a valid `event_id`
- Groups by (event_id, station id) so ar_pick is run once per station/event pair
- Leaves original pick untouched if:
    * waveform is missing
    * Z/N/E are incomplete
    * ar_pick fails
    * dt exceeds the allowed threshold
- Writes only one new CSV by default
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime, read
from obspy.signal.trigger import ar_pick

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# I/O helpers
# ============================================================

def ensure_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Refine merged GaMMA picks with ObsPy ar_pick and write one compatible CSV."
    )
    ap.add_argument("--data_root", type=str, required=True,
                    help="QuickQuake data root, e.g. /path/to/QuickQuake/data")
    ap.add_argument("--namebase", type=str, default="GAMMA",
                    help="Kept for consistency with pipeline naming (default: GAMMA)")
    ap.add_argument("--picks_in", type=str, default=None,
                    help="Optional input picks CSV. Default: <data_root>/merged/gammapicks_id.csv")
    ap.add_argument("--picks_out", type=str, default=None,
                    help="Optional output picks CSV. Default: <data_root>/merged/gammapicks_id_arpick.csv")

    # local windows around existing picks
    ap.add_argument("--pre_p", type=float, default=1.0,
                    help="Seconds before original P pick to include in local repick window")
    ap.add_argument("--post_p", type=float, default=2.0,
                    help="Seconds after original P pick to include in local repick window")
    ap.add_argument("--pre_s", type=float, default=1.5,
                    help="Seconds before original S pick to include in local repick window")
    ap.add_argument("--post_s", type=float, default=3.0,
                    help="Seconds after original S pick to include in local repick window")

    # acceptance thresholds
    ap.add_argument("--max_dt_p", type=float, default=0.30,
                    help="Maximum allowed absolute correction for P picks in seconds")
    ap.add_argument("--max_dt_s", type=float, default=0.50,
                    help="Maximum allowed absolute correction for S picks in seconds")

    # preprocessing for ar_pick
    ap.add_argument("--freqmin", type=float, default=1.0,
                    help="Bandpass low corner before ar_pick")
    ap.add_argument("--freqmax", type=float, default=15.0,
                    help="Bandpass high corner before ar_pick")

    # ar_pick parameters
    ap.add_argument("--lta_p", type=float, default=1.0)
    ap.add_argument("--sta_p", type=float, default=0.1)
    ap.add_argument("--lta_s", type=float, default=2.0)
    ap.add_argument("--sta_s", type=float, default=0.2)
    ap.add_argument("--m_p", type=int, default=2)
    ap.add_argument("--m_s", type=int, default=8)
    ap.add_argument("--l_p", type=float, default=0.1)
    ap.add_argument("--l_s", type=float, default=0.2)

    # behavior
    ap.add_argument("--keep_debug_cols", action="store_true",
                    help="If set, append a few lightweight diagnostic columns to the output CSV")
    return ap.parse_args()


# ============================================================
# waveform helpers
# ============================================================

class StreamCache:
    """
    Cache one merged stream per chunk/window folder, because merged event_id is
    built from the chunk folder name in QQ_merge_gamma_outputs.py.
    """
    def __init__(self):
        self._cache: Dict[str, Optional[obspy.Stream]] = {}

    def get_chunk_stream(self, chunk_dir: Path) -> Optional[obspy.Stream]:
        key = str(chunk_dir.resolve())
        if key in self._cache:
            return self._cache[key]

        wdir = chunk_dir / "waveforms"
        if not wdir.exists():
            self._cache[key] = None
            return None

        files = sorted(wdir.glob("*.mseed"))
        if not files:
            self._cache[key] = None
            return None

        st = None
        for fp in files:
            try:
                s = read(str(fp))
                st = s if st is None else (st + s)
            except Exception:
                pass

        if st is None:
            self._cache[key] = None
            return None

        try:
            st.merge(fill_value="interpolate")
        except Exception:
            pass

        self._cache[key] = st
        return st


def parse_sid(sid: str) -> Tuple[str, str, str, str]:
    """
    Expected sid style from your pipeline:
        NET.STA.LOC.PREFIX
    e.g. AV.DT1..BH
    """
    parts = str(sid).split(".")
    if len(parts) < 4:
        raise ValueError(f"Station id '{sid}' does not look like NET.STA.LOC.PREFIX")
    net, sta, loc, chprefix = parts[0], parts[1], parts[2], parts[3]
    return net, sta, loc, chprefix


def pick_trace_by_component(st: obspy.Stream, sid: str, comp: str) -> Optional[obspy.Trace]:
    """
    Return one trace for requested component (Z/N/E), preferring the original
    channel prefix when possible.
    """
    net, sta, loc, chprefix = parse_sid(sid)

    candidates = [
        f"{chprefix}{comp}",
        f"BH{comp}",
        f"EH{comp}",
        f"SH{comp}",
        f"HH{comp}",
    ]

    # exact location first
    for ch in candidates:
        tr = st.select(network=net, station=sta, location=loc, channel=ch)
        if len(tr) > 0:
            return tr[0].copy()

    # fallback: same net/station and any channel ending in comp
    tr = st.select(network=net, station=sta, channel=f"*{comp}")
    if len(tr) > 0:
        return tr[0].copy()

    return None


def prepare_three_components(
    st_chunk: obspy.Stream,
    sid: str,
    t_start: UTCDateTime,
    t_end: UTCDateTime,
    freqmin: float,
    freqmax: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, UTCDateTime]]:
    tr_z = pick_trace_by_component(st_chunk, sid, "Z")
    tr_n = pick_trace_by_component(st_chunk, sid, "N")
    tr_e = pick_trace_by_component(st_chunk, sid, "E")

    if tr_z is None or tr_n is None or tr_e is None:
        return None

    trs = [tr_z, tr_n, tr_e]
    for tr in trs:
        tr.trim(t_start, t_end, pad=True, fill_value=0)
        tr.detrend("demean")
        tr.taper(max_percentage=0.05, type="cosine")

    # resample to lowest sampling rate if needed
    srs = [float(tr.stats.sampling_rate) for tr in trs]
    sr = min(srs)
    for i, tr in enumerate(trs):
        if abs(float(tr.stats.sampling_rate) - sr) > 1e-6:
            tr.resample(sr)

    # ensure same length
    nmin = min(len(tr.data) for tr in trs)
    if nmin <= 5:
        return None

    trs = [tr.slice(tr.stats.starttime, tr.stats.starttime + (nmin - 1) / sr) for tr in trs]
    trs = [tr.copy() for tr in trs]

    nyq = 0.5 * sr
    fmin = max(0.001, min(float(freqmin), nyq * 0.99))
    fmax = max(fmin + 0.001, min(float(freqmax), nyq * 0.99))

    if fmax > fmin:
        for tr in trs:
            tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

    a = trs[0].data.astype(np.float64)
    b = trs[1].data.astype(np.float64)
    c = trs[2].data.astype(np.float64)

    if not (np.any(np.isfinite(a)) and np.any(np.isfinite(b)) and np.any(np.isfinite(c))):
        return None

    return a, b, c, sr, trs[0].stats.starttime


# ============================================================
# repick core
# ============================================================

def infer_chunk_id_from_event_id(event_id: str) -> Optional[str]:
    """
    merge script builds event_id as:
        <window_id>_<event_idx>
    where window_id = chunk folder name
    """
    if pd.isna(event_id):
        return None
    s = str(event_id)
    if "_" not in s:
        return None
    return s.rsplit("_", 1)[0]


def infer_chunk_id_from_timestamp(ts: pd.Timestamp) -> Optional[str]:
    """
    Fallback only if needed. This is less exact than event_id-based routing.
    """
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y%m%dT%H0000")


def normalize_type(val: str) -> str:
    s = str(val).strip().lower()
    if s in {"p", "P"}:
        return "p"
    if s in {"s", "S"}:
        return "s"
    return s


def choose_group_window(g: pd.DataFrame, pre_p: float, post_p: float, pre_s: float, post_s: float) -> Tuple[UTCDateTime, UTCDateTime]:
    """
    Build one local waveform window per (event_id, station id).
    """
    tmins = []
    tmaxs = []

    for _, r in g.iterrows():
        t = UTCDateTime(pd.Timestamp(r["timestamp"]).to_pydatetime())
        ph = normalize_type(r["type"])
        if ph == "p":
            tmins.append(t - float(pre_p))
            tmaxs.append(t + float(post_p))
        elif ph == "s":
            tmins.append(t - float(pre_s))
            tmaxs.append(t + float(post_s))

    if not tmins:
        # should not happen because caller filters P/S
        t0 = UTCDateTime(pd.Timestamp(g.iloc[0]["timestamp"]).to_pydatetime())
        return t0 - 1.0, t0 + 2.0

    return min(tmins), max(tmaxs)


def run_arpick_on_group(
    g: pd.DataFrame,
    data_root: Path,
    cache: StreamCache,
    *,
    pre_p: float,
    post_p: float,
    pre_s: float,
    post_s: float,
    max_dt_p: float,
    max_dt_s: float,
    freqmin: float,
    freqmax: float,
    lta_p: float,
    sta_p: float,
    lta_s: float,
    sta_s: float,
    m_p: int,
    m_s: int,
    l_p: float,
    l_s: float,
):
    """
    g = one station / one event.
    Returns a dict indexed by original row index:
        {
          row_idx: {
             "new_timestamp": ...,
             "dt": ...,
             "used": bool,
             "status": ...
          },
          ...
        }
    """
    out = {}
    sid = str(g.iloc[0]["id"])
    event_id = g.iloc[0]["event_id"]

    chunk_id = infer_chunk_id_from_event_id(event_id)
    if chunk_id is None:
        for idx in g.index:
            out[idx] = {"new_timestamp": g.loc[idx, "timestamp"], "dt": np.nan, "used": False, "status": "no_chunk_id"}
        return out

    chunk_dir = data_root / chunk_id
    st_chunk = cache.get_chunk_stream(chunk_dir)
    if st_chunk is None:
        for idx in g.index:
            out[idx] = {"new_timestamp": g.loc[idx, "timestamp"], "dt": np.nan, "used": False, "status": "no_waveforms"}
        return out

    t_start, t_end = choose_group_window(g, pre_p, post_p, pre_s, post_s)

    prepared = prepare_three_components(
        st_chunk=st_chunk,
        sid=sid,
        t_start=t_start,
        t_end=t_end,
        freqmin=freqmin,
        freqmax=freqmax,
    )
    if prepared is None:
        for idx in g.index:
            out[idx] = {"new_timestamp": g.loc[idx, "timestamp"], "dt": np.nan, "used": False, "status": "missing_3c"}
        return out

    a, b, c, sr, trace_start = prepared

    try:
        p_rel, s_rel = ar_pick(
            a, b, c, sr,
            float(freqmin), float(freqmax),
            float(lta_p), float(sta_p),
            float(lta_s), float(sta_s),
            int(m_p), int(m_s),
            float(l_p), float(l_s),
            s_pick=True,
        )
    except Exception:
        for idx in g.index:
            out[idx] = {"new_timestamp": g.loc[idx, "timestamp"], "dt": np.nan, "used": False, "status": "arpick_failed"}
        return out

    p_abs = trace_start + float(p_rel) if np.isfinite(p_rel) else None
    s_abs = trace_start + float(s_rel) if np.isfinite(s_rel) else None

    for idx, r in g.iterrows():
        old_ts = pd.Timestamp(r["timestamp"])
        old_utc = UTCDateTime(old_ts.to_pydatetime())
        ph = normalize_type(r["type"])

        if ph == "p" and p_abs is not None:
            new_utc = p_abs
            dt = float(new_utc - old_utc)
            ok = abs(dt) <= float(max_dt_p)
            out[idx] = {
                "new_timestamp": pd.Timestamp(new_utc.datetime, tz="UTC"),
                "dt": dt,
                "used": bool(ok),
                "status": "updated" if ok else "p_dt_too_large",
            }
        elif ph == "s" and s_abs is not None:
            new_utc = s_abs
            dt = float(new_utc - old_utc)
            ok = abs(dt) <= float(max_dt_s)
            out[idx] = {
                "new_timestamp": pd.Timestamp(new_utc.datetime, tz="UTC"),
                "dt": dt,
                "used": bool(ok),
                "status": "updated" if ok else "s_dt_too_large",
            }
        else:
            out[idx] = {
                "new_timestamp": old_ts,
                "dt": np.nan,
                "used": False,
                "status": "phase_not_repicked",
            }

    return out


# ============================================================
# main
# ============================================================

def main():
    args = parse_args()

    data_root = Path(args.data_root).resolve()
    merged_dir = data_root / "merged"

    picks_in = Path(args.picks_in).resolve() if args.picks_in else (merged_dir / "gammapicks_id.csv")
    picks_out = Path(args.picks_out).resolve() if args.picks_out else (merged_dir / "gammapicks_id_arpick.csv")

    ensure_file(picks_in, "input merged picks CSV")

    picks = pd.read_csv(picks_in)
    if "timestamp" not in picks.columns:
        raise ValueError(f"Missing 'timestamp' column in {picks_in}")
    if "id" not in picks.columns:
        raise ValueError(f"Missing 'id' column in {picks_in}")
    if "type" not in picks.columns:
        raise ValueError(f"Missing 'type' column in {picks_in}")
    if "event_id" not in picks.columns:
        raise ValueError(f"Missing 'event_id' column in {picks_in}")

    # preserve original order
    picks = picks.copy()
    picks["timestamp"] = pd.to_datetime(picks["timestamp"], utc=True, errors="coerce", format="mixed")

    out_df = picks.copy()

    if args.keep_debug_cols:
        out_df["timestamp_original"] = out_df["timestamp"]
        out_df["dt_arpick_s"] = np.nan
        out_df["arpick_used"] = False
        out_df["arpick_status"] = "not_processed"

    mask_assoc = out_df["event_id"].notna()
    mask_phase = out_df["type"].astype(str).str.lower().isin(["p", "s"])
    todo = out_df[mask_assoc & mask_phase].copy()

    cache = StreamCache()

    n_groups = 0
    n_rows_considered = 0
    n_rows_updated = 0

    grouped = todo.groupby(["event_id", "id"], sort=False)

    for (_, _), g in grouped:
        n_groups += 1
        n_rows_considered += len(g)

        result = run_arpick_on_group(
            g=g,
            data_root=data_root,
            cache=cache,
            pre_p=args.pre_p,
            post_p=args.post_p,
            pre_s=args.pre_s,
            post_s=args.post_s,
            max_dt_p=args.max_dt_p,
            max_dt_s=args.max_dt_s,
            freqmin=args.freqmin,
            freqmax=args.freqmax,
            lta_p=args.lta_p,
            sta_p=args.sta_p,
            lta_s=args.lta_s,
            sta_s=args.sta_s,
            m_p=args.m_p,
            m_s=args.m_s,
            l_p=args.l_p,
            l_s=args.l_s,
        )

        for idx, r in result.items():
            if bool(r["used"]):
                out_df.at[idx, "timestamp"] = r["new_timestamp"]
                n_rows_updated += 1

            if args.keep_debug_cols:
                out_df.at[idx, "dt_arpick_s"] = r["dt"]
                out_df.at[idx, "arpick_used"] = bool(r["used"])
                out_df.at[idx, "arpick_status"] = str(r["status"])

        if n_groups % 200 == 0:
            print(f"[ar_pick] processed {n_groups} station-event groups")

    # keep output compatible
    # by default preserve original columns only
    if not args.keep_debug_cols:
        out_df = out_df[picks.columns.tolist()]

    picks_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(
        picks_out,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )

    print("\n✅ ar_pick refinement finished.")
    print(f"Input picks : {picks_in}")
    print(f"Output picks: {picks_out}")
    print(f"Groups processed         : {n_groups}")
    print(f"P/S rows considered      : {n_rows_considered}")
    print(f"Rows updated by ar_pick  : {n_rows_updated}")
    if n_rows_considered > 0:
        print(f"Update fraction          : {100.0 * n_rows_updated / n_rows_considered:.2f}%")

if __name__ == "__main__":
    main()