#!/usr/bin/env python3
"""
QuickQuake - faster P/S pick refinement with ObsPy ar_pick

Key ideas:
- keeps the SAME scientific core:
    * same ar_pick call
    * same acceptance thresholds
    * same "keep original pick if refinement fails" philosophy
- reads the large merged picks CSV only once in the parent
- parallelizes by batches of chunk_id using multiprocessing spawn
- handles exceptions at group level so one bad group does not kill a batch
- processes chunk -> station -> event for speed
- preprocesses each station ONCE per chunk (select, detrend, taper, resample, filter)
- then extracts small windows per event from the cached station arrays

Default behavior:
    input : <data_root>/merged/gammapicks_id.csv
    output: <data_root>/merged/gammapicks_id_arpick.csv
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

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
    s = str(event_id).strip()
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


def make_original_result_rows(g: pd.DataFrame, status: str) -> List[Dict[str, Any]]:
    rows = []
    for idx in g.index:
        rows.append({
            "row_index": int(idx),
            "new_timestamp": g.loc[idx, "timestamp"],
            "dt": np.nan,
            "used": False,
            "status": status,
        })
    return rows


def pool_kwargs(processes: int, maxtasksperchild: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"processes": max(1, int(processes))}
    if maxtasksperchild is not None and int(maxtasksperchild) > 0:
        kwargs["maxtasksperchild"] = int(maxtasksperchild)
    return kwargs


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

    # preprocessing / ar_pick frequency args
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
    ap.add_argument("--log_every", type=int, default=5000,
                    help="Print progress every N station-event groups inside a worker")
    ap.add_argument("--nproc", type=int, default=max(1, min(4, (os.cpu_count() or 4) - 1)),
                    help="Number of worker processes for batch-level parallelism")
    ap.add_argument("--chunks_per_batch", type=int, default=12,
                    help="How many chunk_ids to send per batch worker")
    ap.add_argument("--maxtasksperchild", type=int, default=0,
                    help="Recycle worker after N tasks. Use 0 or negative to disable recycling.")
    ap.add_argument("--batch_timeout_s", type=int, default=7200,
                    help="Timeout in seconds for a batch")
    ap.add_argument("--chunk_timeout_s", type=int, default=3600,
                    help="Timeout in seconds for a chunk fallback")
    ap.add_argument("--group_timeout_s", type=int, default=600,
                    help="Timeout in seconds for a group fallback")

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

    for ch in candidates:
        tr = st.select(network=net, station=sta, location=loc, channel=ch)
        if len(tr) > 0:
            return tr[0].copy()

    tr = st.select(network=net, station=sta, channel=f"*{comp}")
    if len(tr) > 0:
        return tr[0].copy()

    return None


def prepare_station_cache(
    st_chunk: obspy.Stream,
    sid: str,
    freqmin: float,
    freqmax: float,
) -> Optional[Dict[str, Any]]:
    """
    Prepare one station once for the whole chunk:
    - select Z/N/E
    - demean/taper
    - resample to common rate
    - bandpass once
    - store arrays + common start/end metadata
    """
    tr_z = pick_trace_by_component(st_chunk, sid, "Z")
    tr_n = pick_trace_by_component(st_chunk, sid, "N")
    tr_e = pick_trace_by_component(st_chunk, sid, "E")

    if tr_z is None or tr_n is None or tr_e is None:
        return None

    trs = [tr_z, tr_n, tr_e]

    for tr in trs:
        tr.detrend("demean")
        tr.taper(max_percentage=0.05, type="cosine")

    srs = [float(tr.stats.sampling_rate) for tr in trs]
    sr = min(srs)

    for tr in trs:
        if abs(float(tr.stats.sampling_rate) - sr) > 1e-6:
            tr.resample(sr)

    t0 = max(tr.stats.starttime for tr in trs)
    t1 = min(tr.stats.endtime for tr in trs)
    if t1 <= t0:
        return None

    trs = [tr.slice(t0, t1) for tr in trs]

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

    a = trs[0].data.astype(np.float64)
    b = trs[1].data.astype(np.float64)
    c = trs[2].data.astype(np.float64)

    if not (np.any(np.isfinite(a)) and np.any(np.isfinite(b)) and np.any(np.isfinite(c))):
        return None

    return {
        "a": a,
        "b": b,
        "c": c,
        "sr": sr,
        "trace_start": trs[0].stats.starttime,
        "npts": len(a),
    }


def extract_window_from_cache(
    cache: Dict[str, Any],
    t_start: UTCDateTime,
    t_end: UTCDateTime,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, UTCDateTime]]:
    """
    Extract a padded time window from a preprocessed station cache.
    """
    sr = float(cache["sr"])
    trace_start = cache["trace_start"]
    npts = int(cache["npts"])

    a_full = cache["a"]
    b_full = cache["b"]
    c_full = cache["c"]

    win_n = int(round((t_end - t_start) * sr)) + 1
    if win_n <= 5:
        return None

    out_a = np.zeros(win_n, dtype=np.float64)
    out_b = np.zeros(win_n, dtype=np.float64)
    out_c = np.zeros(win_n, dtype=np.float64)

    i0_full = int(np.floor((t_start - trace_start) * sr))
    i1_full = i0_full + win_n

    src0 = max(0, i0_full)
    src1 = min(npts, i1_full)

    if src1 <= src0:
        return None

    dst0 = src0 - i0_full
    dst1 = dst0 + (src1 - src0)

    out_a[dst0:dst1] = a_full[src0:src1]
    out_b[dst0:dst1] = b_full[src0:src1]
    out_c[dst0:dst1] = c_full[src0:src1]

    if not (np.any(np.isfinite(out_a)) and np.any(np.isfinite(out_b)) and np.any(np.isfinite(out_c))):
        return None

    return out_a, out_b, out_c, sr, t_start


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
    station_cache: Dict[str, Any],
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
    out = {}

    t_start, t_end = choose_group_window(g, pre_p, post_p, pre_s, post_s)

    prepared = extract_window_from_cache(
        cache=station_cache,
        t_start=t_start,
        t_end=t_end,
    )

    if prepared is None:
        for row in make_original_result_rows(g, "missing_window"):
            out[row["row_index"]] = row
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
        for row in make_original_result_rows(g, "arpick_failed"):
            out[row["row_index"]] = row
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
            out[int(idx)] = {
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
            out[int(idx)] = {
                "row_index": int(idx),
                "new_timestamp": pd.Timestamp(new_utc.datetime, tz="UTC"),
                "dt": dt,
                "used": bool(ok),
                "status": "updated" if ok else "s_dt_too_large",
            }
        else:
            out[int(idx)] = {
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
    todo["chunk_id"] = todo["event_id"].apply(infer_chunk_id_from_event_id)

    todo = todo.loc[todo["chunk_id"].notna()].copy()
    todo["chunk_id"] = todo["chunk_id"].astype(str).str.strip()

    # Keep only chunk ids with expected format YYYYMMDDTHHMMSS
    todo = todo.loc[todo["chunk_id"].str.match(r"^\d{8}T\d{6}$", na=False)].copy()

    todo = todo.sort_values(["chunk_id", "id", "event_id"], kind="stable")
    return todo


def split_list(seq: List[str], batch_size: int) -> List[List[str]]:
    if batch_size <= 0:
        batch_size = 1
    return [seq[i:i + batch_size] for i in range(0, len(seq), batch_size)]


# ============================================================
# worker function for multiprocessing
# ============================================================

def process_batch(batch_df: pd.DataFrame, data_root_str: str, worker_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Faster worker:
    - processes chunk by chunk
    - inside each chunk, groups by station first
    - preprocesses each station only once
    - then runs ar_pick for each event of that station
    """
    if len(batch_df) == 0:
        return []

    data_root = Path(data_root_str)

    batch_df = batch_df.copy()
    batch_df["timestamp"] = pd.to_datetime(batch_df["timestamp"], utc=True, errors="coerce", format="mixed")
    batch_df = batch_df.sort_values(["chunk_id", "id", "event_id"], kind="stable")

    results: List[Dict[str, Any]] = []
    n_groups = 0

    grouped_by_chunk = batch_df.groupby("chunk_id", sort=False)

    for chunk_id, chunk_df in grouped_by_chunk:
        try:
            st_chunk = load_chunk_stream(data_root / str(chunk_id))
        except Exception as e:
            print(f"[ar_pick worker] failed loading chunk stream | chunk={chunk_id} | err={e}", flush=True)
            st_chunk = None

        if st_chunk is None:
            for _, g in chunk_df.groupby(["event_id", "id"], sort=False):
                results.extend(make_original_result_rows(g, "no_waveforms"))
            continue

        grouped_by_sid = chunk_df.groupby("id", sort=False)

        for sid, sid_df in grouped_by_sid:
            try:
                station_cache = prepare_station_cache(
                    st_chunk=st_chunk,
                    sid=str(sid),
                    freqmin=worker_params["freqmin"],
                    freqmax=worker_params["freqmax"],
                )
            except Exception as e:
                print(f"[ar_pick worker] station cache failed | chunk={chunk_id} | station={sid} | err={e}", flush=True)
                station_cache = None

            if station_cache is None:
                for _, g in sid_df.groupby("event_id", sort=False):
                    results.extend(make_original_result_rows(g, "missing_3c"))
                continue

            for _, g in sid_df.groupby("event_id", sort=False):
                n_groups += 1

                try:
                    res = run_arpick_on_group(
                        g=g,
                        station_cache=station_cache,
                        pre_p=worker_params["pre_p"],
                        post_p=worker_params["post_p"],
                        pre_s=worker_params["pre_s"],
                        post_s=worker_params["post_s"],
                        max_dt_p=worker_params["max_dt_p"],
                        max_dt_s=worker_params["max_dt_s"],
                        freqmin=worker_params["freqmin"],
                        freqmax=worker_params["freqmax"],
                        lta_p=worker_params["lta_p"],
                        sta_p=worker_params["sta_p"],
                        lta_s=worker_params["lta_s"],
                        sta_s=worker_params["sta_s"],
                        m_p=worker_params["m_p"],
                        m_s=worker_params["m_s"],
                        l_p=worker_params["l_p"],
                        l_s=worker_params["l_s"],
                    )
                    results.extend(res.values())
                    del res

                except Exception as e:
                    event_id = str(g.iloc[0]["event_id"]) if len(g) else "unknown_event"
                    print(
                        f"[ar_pick worker] ERROR processing group | chunk={chunk_id} | event_id={event_id} | station={sid}: {e}",
                        flush=True
                    )
                    results.extend(make_original_result_rows(g, "group_exception"))

                if n_groups % int(worker_params["log_every"]) == 0:
                    print(f"[ar_pick worker] processed {n_groups} station-event groups", flush=True)

                if n_groups % 100 == 0:
                    gc.collect()

            del station_cache
            gc.collect()

        del st_chunk
        gc.collect()

    return results


# ============================================================
# parent helpers
# ============================================================

def apply_updates_to_out_df(out_df: pd.DataFrame, upd: pd.DataFrame, keep_debug_cols: bool):
    n_rows_considered = 0
    n_rows_updated = 0

    if len(upd) == 0:
        return n_rows_considered, n_rows_updated

    upd = upd.copy()
    upd["row_index"] = pd.to_numeric(upd["row_index"], errors="coerce").astype("Int64")
    upd["used"] = upd["used"].astype(bool)
    upd["new_timestamp"] = pd.to_datetime(upd["new_timestamp"], utc=True, errors="coerce", format="mixed")

    n_rows_considered = len(upd)

    used = upd["used"].fillna(False)
    if used.any():
        used_rows = upd.loc[used]
        for _, r in used_rows.iterrows():
            idx = int(r["row_index"])
            out_df.at[idx, "timestamp"] = r["new_timestamp"]
            n_rows_updated += 1

    if keep_debug_cols:
        for _, r in upd.iterrows():
            idx = int(r["row_index"])
            out_df.at[idx, "dt_arpick_s"] = r["dt"]
            out_df.at[idx, "arpick_used"] = bool(r["used"])
            out_df.at[idx, "arpick_status"] = str(r["status"])

    return n_rows_considered, n_rows_updated


def fallback_chunk_group_by_group(
    chunk_df: pd.DataFrame,
    data_root: Path,
    worker_params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Safest fallback:
    process one group at a time in an isolated short-lived subprocess via spawn Pool(1).
    If a group crashes native code, only that tiny task dies and the original picks are preserved.
    """
    if len(chunk_df) == 0:
        return pd.DataFrame(columns=["row_index", "new_timestamp", "dt", "used", "status"])

    chunk_id = str(chunk_df["chunk_id"].iloc[0])
    print(f"[ar_pick parent] fallback group-by-group for chunk {chunk_id}", flush=True)

    all_rows = []
    grouped = list(chunk_df.groupby(["event_id", "id"], sort=False))
    ctx = mp.get_context("spawn")

    for j, (_, g) in enumerate(grouped, start=1):
        try:
            with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
                ar = pool.apply_async(process_batch, (g.copy(), str(data_root), worker_params))
                group_rows = ar.get(timeout=int(worker_params["group_timeout_s"]))
                all_rows.extend(group_rows)

        except Exception:
            event_id = str(g.iloc[0]["event_id"])
            sid = str(g.iloc[0]["id"])
            print(f"[ar_pick parent] skipped crashing group | chunk={chunk_id} | event_id={event_id} | station={sid}", flush=True)
            all_rows.extend(make_original_result_rows(g, "group_crashed"))

        if j % 100 == 0:
            print(f"[ar_pick parent] fallback chunk {chunk_id}: processed {j} groups", flush=True)

    return pd.DataFrame(all_rows, columns=["row_index", "new_timestamp", "dt", "used", "status"])


# ============================================================
# main
# ============================================================

def main():
    args = parse_args()

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

    if len(todo) == 0:
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

    chunk_ids = todo["chunk_id"].dropna().astype(str).drop_duplicates().tolist()
    batches = split_list(chunk_ids, max(1, int(args.chunks_per_batch)))

    worker_params = {
        "pre_p": args.pre_p,
        "post_p": args.post_p,
        "pre_s": args.pre_s,
        "post_s": args.post_s,
        "max_dt_p": args.max_dt_p,
        "max_dt_s": args.max_dt_s,
        "freqmin": args.freqmin,
        "freqmax": args.freqmax,
        "lta_p": args.lta_p,
        "sta_p": args.sta_p,
        "lta_s": args.lta_s,
        "sta_s": args.sta_s,
        "m_p": args.m_p,
        "m_s": args.m_s,
        "l_p": args.l_p,
        "l_s": args.l_s,
        "log_every": args.log_every,
        "group_timeout_s": args.group_timeout_s,
    }

    ctx = mp.get_context("spawn")

    n_rows_considered = 0
    n_rows_updated = 0

    batch_payloads = []
    for i, batch_chunk_ids in enumerate(batches, start=1):
        batch_df = todo.loc[todo["chunk_id"].isin(batch_chunk_ids)].copy()
        batch_payloads.append((i, batch_chunk_ids, batch_df))

    with ctx.Pool(**pool_kwargs(processes=args.nproc, maxtasksperchild=args.maxtasksperchild)) as pool:
        async_results = []
        for i, batch_chunk_ids, batch_df in batch_payloads:
            print(f"[ar_pick parent] batch {i}/{len(batch_payloads)} -> chunks={batch_chunk_ids}", flush=True)
            ar = pool.apply_async(process_batch, (batch_df, str(data_root), worker_params))
            async_results.append((i, batch_chunk_ids, batch_df, ar))

        pool.close()

        for i, batch_chunk_ids, batch_df, ar in async_results:
            try:
                batch_rows = ar.get(timeout=int(args.batch_timeout_s))
                upd = pd.DataFrame(batch_rows, columns=["row_index", "new_timestamp", "dt", "used", "status"])

            except Exception as e:
                print(f"[ar_pick parent] batch {i} failed: {e}", flush=True)
                print(f"[ar_pick parent] entering fallback for batch {i}", flush=True)

                upd_parts = []

                for chunk_id in batch_chunk_ids:
                    chunk_df = todo.loc[todo["chunk_id"] == chunk_id].copy()

                    try:
                        with ctx.Pool(processes=1, maxtasksperchild=1) as chunk_pool:
                            ar_chunk = chunk_pool.apply_async(process_batch, (chunk_df, str(data_root), worker_params))
                            chunk_rows = ar_chunk.get(timeout=int(args.chunk_timeout_s))

                        chunk_upd = pd.DataFrame(chunk_rows, columns=["row_index", "new_timestamp", "dt", "used", "status"])

                    except Exception as e_chunk:
                        print(f"[ar_pick parent] chunk {chunk_id} failed: {e_chunk}", flush=True)
                        chunk_upd = fallback_chunk_group_by_group(
                            chunk_df=chunk_df,
                            data_root=data_root,
                            worker_params=worker_params,
                        )

                    upd_parts.append(chunk_upd)

                    del chunk_df, chunk_upd
                    gc.collect()

                if len(upd_parts) > 0:
                    upd = pd.concat(upd_parts, ignore_index=True)
                else:
                    upd = pd.DataFrame(columns=["row_index", "new_timestamp", "dt", "used", "status"])

            c_considered, c_updated = apply_updates_to_out_df(
                out_df=out_df,
                upd=upd,
                keep_debug_cols=args.keep_debug_cols,
            )
            n_rows_considered += c_considered
            n_rows_updated += c_updated

            del upd
            gc.collect()

        pool.join()

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
    print(f"Batches processed        : {len(batches)}")
    print(f"P/S rows considered      : {n_rows_considered}")
    print(f"Rows updated by ar_pick  : {n_rows_updated}")
    if n_rows_considered > 0:
        print(f"Update fraction          : {100.0 * n_rows_updated / n_rows_considered:.2f}%")


if __name__ == "__main__":
    main()