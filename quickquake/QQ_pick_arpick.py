#!/usr/bin/env python3
"""
QuickQuake - lightweight P/S pick refinement with ObsPy ar_pick

Robust version for large runs on macOS / long catalogs:
- keeps the SAME repick logic and SAME acceptance rules
- isolates processing by chunk in short-lived subprocesses
- avoids long-lived native-memory accumulation/corruption

Default behavior:
    input : <data_root>/merged/gammapicks_id.csv
    output: <data_root>/merged/gammapicks_id_arpick.csv
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime, read
from obspy.signal.trigger import ar_pick

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# basic helpers
# ============================================================

def ensure_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def normalize_type(val: str) -> str:
    s = str(val).strip().lower()
    if s == "p":
        return "p"
    if s == "s":
        return "s"
    return s


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


def parse_sid(sid: str) -> Tuple[str, str, str, str]:
    """
    Expected sid style:
        NET.STA.LOC.PREFIX
    e.g. AV.DT1..BH
    """
    parts = str(sid).split(".")
    if len(parts) < 4:
        raise ValueError(f"Station id '{sid}' does not look like NET.STA.LOC.PREFIX")
    net, sta, loc, chprefix = parts[0], parts[1], parts[2], parts[3]
    return net, sta, loc, chprefix


# ============================================================
# argument parsing
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Refine merged GaMMA picks with ObsPy ar_pick."
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
    ap.add_argument("--pre_p", type=float, default=1.0)
    ap.add_argument("--post_p", type=float, default=2.0)
    ap.add_argument("--pre_s", type=float, default=1.5)
    ap.add_argument("--post_s", type=float, default=3.0)

    # acceptance thresholds
    ap.add_argument("--max_dt_p", type=float, default=0.30)
    ap.add_argument("--max_dt_s", type=float, default=0.50)

    # preprocessing for ar_pick
    ap.add_argument("--freqmin", type=float, default=1.0)
    ap.add_argument("--freqmax", type=float, default=15.0)

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
                    help="If set, append lightweight diagnostic columns to the final CSV")
    ap.add_argument("--log_every", type=int, default=200,
                    help="Print progress every N station-event groups in worker mode")

    # internal worker mode
    ap.add_argument("--worker_chunk_id", type=str, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--worker_out", type=str, default=None,
                    help=argparse.SUPPRESS)

    return ap.parse_args()


# ============================================================
# waveform helpers
# ============================================================

def load_chunk_stream(chunk_dir: Path) -> Optional[obspy.Stream]:
    wdir = chunk_dir / "waveforms"
    if not wdir.exists():
        return None

    files = sorted(wdir.glob("*.mseed"))
    if not files:
        return None

    st = None
    for fp in files:
        try:
            s = read(str(fp))
            st = s if st is None else (st + s)
        except Exception:
            pass

    if st is None or len(st) == 0:
        return None

    try:
        st.merge(fill_value="interpolate")
    except Exception:
        pass

    return st


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

    for tr in trs:
        if abs(float(tr.stats.sampling_rate) - sr) > 1e-6:
            tr.resample(sr)

    # ensure same length
    nmin = min(len(tr.data) for tr in trs)
    if nmin <= 5:
        return None

    trs = [tr.slice(tr.stats.starttime, tr.stats.starttime + (nmin - 1) / sr) for tr in trs]

    nyq = 0.5 * sr
    fmin = max(0.001, min(float(freqmin), nyq * 0.99))
    fmax = max(fmin + 0.001, min(float(freqmax), nyq * 0.99))

    if fmax > fmin:
        for tr in trs:
            tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)

    # keep dtype/logic same as before
    a = trs[0].data.astype(np.float64)
    b = trs[1].data.astype(np.float64)
    c = trs[2].data.astype(np.float64)
    trace_start = trs[0].stats.starttime

    if not (np.any(np.isfinite(a)) and np.any(np.isfinite(b)) and np.any(np.isfinite(c))):
        return None

    del tr_z, tr_n, tr_e, trs
    return a, b, c, sr, trace_start


# ============================================================
# repick core
# ============================================================

def choose_group_window(
    g: pd.DataFrame,
    pre_p: float,
    post_p: float,
    pre_s: float,
    post_s: float
) -> Tuple[UTCDateTime, UTCDateTime]:
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
        t0 = UTCDateTime(pd.Timestamp(g.iloc[0]["timestamp"]).to_pydatetime())
        return t0 - 1.0, t0 + 2.0

    return min(tmins), max(tmaxs)


def run_arpick_on_group(
    g: pd.DataFrame,
    st_chunk: obspy.Stream,
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
            out[idx] = {
                "row_index": int(idx),
                "new_timestamp": g.loc[idx, "timestamp"],
                "dt": np.nan,
                "used": False,
                "status": "missing_3c",
            }
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
            out[idx] = {
                "row_index": int(idx),
                "new_timestamp": g.loc[idx, "timestamp"],
                "dt": np.nan,
                "used": False,
                "status": "arpick_failed",
            }
        del a, b, c
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
                "row_index": int(idx),
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
                "row_index": int(idx),
                "new_timestamp": pd.Timestamp(new_utc.datetime, tz="UTC"),
                "dt": dt,
                "used": bool(ok),
                "status": "updated" if ok else "s_dt_too_large",
            }
        else:
            out[idx] = {
                "row_index": int(idx),
                "new_timestamp": old_ts,
                "dt": np.nan,
                "used": False,
                "status": "phase_not_repicked",
            }

    del a, b, c
    return out


# ============================================================
# dataframe helpers
# ============================================================

def load_and_prepare_picks(picks_in: Path) -> pd.DataFrame:
    ensure_file(picks_in, "input merged picks CSV")

    picks = pd.read_csv(picks_in)

    required_cols = ["id", "timestamp", "type", "event_idx", "event_id"]
    missing = [c for c in required_cols if c not in picks.columns]
    if missing:
        raise ValueError(f"Missing required columns in {picks_in}: {missing}")

    picks = picks.copy()
    picks["timestamp"] = pd.to_datetime(picks["timestamp"], utc=True, errors="coerce", format="mixed")
    return picks


def build_todo_from_picks(picks: pd.DataFrame) -> pd.DataFrame:
    event_idx_num = pd.to_numeric(picks["event_idx"], errors="coerce")
    phase_norm = picks["type"].astype(str).str.strip().str.lower()

    mask_assoc = picks["event_id"].notna() & event_idx_num.notna() & (event_idx_num >= 0)
    mask_phase = phase_norm.isin(["p", "s"])

    todo = picks.loc[mask_assoc & mask_phase].copy()
    todo["chunk_id"] = todo["event_id"].astype(str).str.rsplit("_", n=1).str[0]
    todo = todo.sort_values(["chunk_id", "event_id", "id"], kind="stable")
    return todo


# ============================================================
# worker mode: process ONE chunk in a fresh subprocess
# ============================================================

def worker_main(args):
    data_root = Path(args.data_root).resolve()
    merged_dir = data_root / "merged"
    picks_in = Path(args.picks_in).resolve() if args.picks_in else (merged_dir / "gammapicks_id.csv")

    if args.worker_chunk_id is None:
        raise ValueError("worker mode requires --worker_chunk_id")
    if args.worker_out is None:
        raise ValueError("worker mode requires --worker_out")

    chunk_id = args.worker_chunk_id
    worker_out = Path(args.worker_out).resolve()

    picks = load_and_prepare_picks(picks_in)
    todo = build_todo_from_picks(picks)
    todo = todo.loc[todo["chunk_id"] == chunk_id].copy()

    rows_out = []

    if len(todo) == 0:
        pd.DataFrame(columns=["row_index", "new_timestamp", "dt", "used", "status"]).to_csv(worker_out, index=False)
        return

    chunk_dir = data_root / chunk_id
    st_chunk = load_chunk_stream(chunk_dir)

    n_groups = 0

    if st_chunk is None:
        grouped = todo.groupby(["event_id", "id"], sort=False)
        for _, g in grouped:
            n_groups += 1
            for idx in g.index:
                rows_out.append({
                    "row_index": int(idx),
                    "new_timestamp": picks.loc[idx, "timestamp"],
                    "dt": np.nan,
                    "used": False,
                    "status": "no_waveforms",
                })
        pd.DataFrame(rows_out).to_csv(worker_out, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
        return

    grouped = todo.groupby(["event_id", "id"], sort=False)

    for _, g in grouped:
        n_groups += 1

        result = run_arpick_on_group(
            g=g,
            st_chunk=st_chunk,
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

        rows_out.extend(result.values())

        if n_groups % int(args.log_every) == 0:
            print(f"[ar_pick worker:{chunk_id}] processed {n_groups} station-event groups", flush=True)

        del result, g
        if n_groups % 50 == 0:
            gc.collect()

    pd.DataFrame(rows_out).to_csv(
        worker_out,
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f",
    )


# ============================================================
# parent mode: orchestrate all chunk workers and merge output
# ============================================================

def parent_main(args):
    data_root = Path(args.data_root).resolve()
    merged_dir = data_root / "merged"

    picks_in = Path(args.picks_in).resolve() if args.picks_in else (merged_dir / "gammapicks_id.csv")
    picks_out = Path(args.picks_out).resolve() if args.picks_out else (merged_dir / "gammapicks_id_arpick.csv")

    picks = load_and_prepare_picks(picks_in)
    todo = build_todo_from_picks(picks)

    out_df = picks.copy()

    if args.keep_debug_cols:
        out_df["timestamp_original"] = out_df["timestamp"]
        out_df["dt_arpick_s"] = np.nan
        out_df["arpick_used"] = False
        out_df["arpick_status"] = "not_processed"

    chunk_ids = todo["chunk_id"].dropna().astype(str).drop_duplicates().tolist()

    if len(chunk_ids) == 0:
        if args.keep_debug_cols:
            debug_cols = ["timestamp_original", "dt_arpick_s", "arpick_used", "arpick_status"]
            base_cols = [c for c in picks.columns]
            out_df = out_df[base_cols + debug_cols]
        else:
            out_df = out_df[picks.columns.tolist()]

        picks_out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(picks_out, index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
        print("No associated P/S picks to process.")
        return

    tmp_root = Path(tempfile.mkdtemp(prefix="qq_arpick_chunks_"))

    n_rows_considered = 0
    n_rows_updated = 0

    try:
        for i, chunk_id in enumerate(chunk_ids, start=1):
            tmp_csv = tmp_root / f"{i:06d}_{chunk_id}_updates.csv"

            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--data_root", str(data_root),
                "--namebase", str(args.namebase),
                "--picks_in", str(picks_in),
                "--worker_chunk_id", str(chunk_id),
                "--worker_out", str(tmp_csv),
                "--pre_p", str(args.pre_p),
                "--post_p", str(args.post_p),
                "--pre_s", str(args.pre_s),
                "--post_s", str(args.post_s),
                "--max_dt_p", str(args.max_dt_p),
                "--max_dt_s", str(args.max_dt_s),
                "--freqmin", str(args.freqmin),
                "--freqmax", str(args.freqmax),
                "--lta_p", str(args.lta_p),
                "--sta_p", str(args.sta_p),
                "--lta_s", str(args.lta_s),
                "--sta_s", str(args.sta_s),
                "--m_p", str(args.m_p),
                "--m_s", str(args.m_s),
                "--l_p", str(args.l_p),
                "--l_s", str(args.l_s),
                "--log_every", str(args.log_every),
            ]

            print(f"[ar_pick parent] chunk {i}/{len(chunk_ids)} -> {chunk_id}", flush=True)

            subprocess.run(cmd, check=True)

            upd = pd.read_csv(tmp_csv)
            if len(upd) == 0:
                continue

            upd["row_index"] = pd.to_numeric(upd["row_index"], errors="coerce").astype("Int64")
            upd["used"] = upd["used"].astype(str).str.lower().isin(["true", "1", "yes"])
            upd["new_timestamp"] = pd.to_datetime(upd["new_timestamp"], utc=True, errors="coerce", format="mixed")

            n_rows_considered += len(upd)

            used = upd["used"].fillna(False)
            if used.any():
                used_rows = upd.loc[used].copy()

                for _, r in used_rows.iterrows():
                    idx = int(r["row_index"])
                    out_df.at[idx, "timestamp"] = r["new_timestamp"]
                    n_rows_updated += 1

            if args.keep_debug_cols:
                for _, r in upd.iterrows():
                    idx = int(r["row_index"])
                    out_df.at[idx, "dt_arpick_s"] = r["dt"]
                    out_df.at[idx, "arpick_used"] = bool(r["used"])
                    out_df.at[idx, "arpick_status"] = str(r["status"])

            del upd
            gc.collect()

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "ar_pick worker failed. "
            "This now means the failure is isolated to a specific chunk worker, "
            "not the entire long-lived parent process."
        ) from e
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if args.keep_debug_cols:
        debug_cols = ["timestamp_original", "dt_arpick_s", "arpick_used", "arpick_status"]
        base_cols = [c for c in picks.columns]
        out_df = out_df[base_cols + debug_cols]
    else:
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
    print(f"Chunks processed         : {len(chunk_ids)}")
    print(f"P/S rows considered      : {n_rows_considered}")
    print(f"Rows updated by ar_pick  : {n_rows_updated}")
    if n_rows_considered > 0:
        print(f"Update fraction          : {100.0 * n_rows_updated / n_rows_considered:.2f}%")


# ============================================================
# entry point
# ============================================================

def main():
    args = parse_args()

    if args.worker_chunk_id is not None:
        worker_main(args)
    else:
        parent_main(args)


if __name__ == "__main__":
    main()