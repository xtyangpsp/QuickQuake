# qq_helpers.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bisect import bisect_right

import numpy as np
import pandas as pd
import matplotlib.dates as mdates

import obspy
from obspy import read, UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth


# =========================
# SCHEMAS (CANONICAL NAMES)
# =========================

@dataclass(frozen=True)
class CatalogCols:
    time: str = "datetime"
    lat: str = "latitude"
    lon: str = "longitude"
    depth_km: str = "depth_km"
    mag: str = "magnitude"
    event_id: str = "event_id"

@dataclass(frozen=True)
class PicksCols:
    station_id: str = "id"
    time: str = "timestamp"
    phase: str = "type"
    prob: str = "prob"
    event_id: str = "event_id"
    event_id_mapped: str = "event_id_mapped"

@dataclass(frozen=True)
class StationCols:
    station_id: str = "id"
    lat: str = "lat"
    lon: str = "lon"
    elev_km: str = "elev_km"


# =========================
# SMALL UTILS
# =========================

def _rename_first_match(df: pd.DataFrame, target: str, candidates: List[str]) -> pd.DataFrame:
    """
    If target exists -> do nothing.
    Else: find first candidate that exists and rename it -> target.
    If none found -> leave as is (caller decides if it's fatal).
    """
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: target})
    return df

def _require_cols(df: pd.DataFrame, required: List[str], name: str = "DataFrame") -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name} missing required columns: {missing}\n"
            f"Columns present: {list(df.columns)}"
        )

def _coerce_utc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


# =========================
# NORMALIZERS (RENAMING HERE ONLY)
# =========================

def normalize_catalog(catalog: pd.DataFrame, catc: CatalogCols = CatalogCols()) -> pd.DataFrame:
    df = catalog.copy()

    df = _rename_first_match(df, catc.time, ["time", "origin_time", "otime", "t0"])
    df = _rename_first_match(df, catc.lat,  ["lat", "Lat", "LAT"])
    df = _rename_first_match(df, catc.lon,  ["lon", "Lon", "LON"])
    df = _rename_first_match(df, catc.depth_km, ["depth", "depthKm", "Depth_km"])
    df = _rename_first_match(df, catc.mag,  ["mag", "Mag", "MAG"])
    df = _rename_first_match(df, catc.event_id, ["event_idx", "eventid", "eid"])

    _require_cols(df, [catc.time, catc.lat, catc.lon], name="catalog")

    df = _coerce_utc(df, catc.time)
    # Force event_id to nullable integer (for consistent comparisons)
    if catc.event_id in df.columns:
        df[catc.event_id] = pd.to_numeric(df[catc.event_id], errors="coerce").astype("Int64")


    # Default depth
    if catc.depth_km not in df.columns:
        df[catc.depth_km] = 0.0

    return df

def normalize_picks(picks: pd.DataFrame, pkc: PicksCols = PicksCols()) -> pd.DataFrame:
    df = picks.copy()

    df = _rename_first_match(df, pkc.time, ["time", "pick_time", "datetime", "t"])
    df = _rename_first_match(df, pkc.station_id, ["station_id", "station", "sta", "sid"])
    df = _rename_first_match(df, pkc.phase, ["phase", "phase_type", "pick_type"])

    _require_cols(df, [pkc.station_id, pkc.time, pkc.phase], name="picks")

    df = _coerce_utc(df, pkc.time)

    # Force event id columns to nullable integer (event_id / event_id_mapped)
    for c in [pkc.event_id, pkc.event_id_mapped]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df




# =========================
# LOADERS
# =========================

def load_stations_json(stations_json: Path) -> pd.DataFrame:
    """
    Return stations with canonical columns: id, lat, lon, elev_km
    """
    station_df = pd.read_json(stations_json).T
    station_df.index.name = "id"
    station_df["elev_km"] = station_df.get("elevation(m)", 0.0).astype(float) / 1000.0

    stations = (station_df
                .rename(columns={"latitude": "lat", "longitude": "lon"})
                .reset_index()[["id", "lat", "lon", "elev_km"]]
                .dropna(subset=["lat", "lon"]))
    return stations


def load_inputs(catalog_csv: Path, picks_csv: Path, stations_json: Path,
                catc: CatalogCols = CatalogCols(),
                pkc: PicksCols = PicksCols()) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalog = pd.read_csv(catalog_csv)
    picks   = pd.read_csv(picks_csv)
    stations = load_stations_json(stations_json)

    catalog = normalize_catalog(catalog, catc=catc)
    picks   = normalize_picks(picks, pkc=pkc)

    return catalog, picks, stations






# =========================
# WAVEFORMS: INDEX + CACHE
# =========================

class WaveformIndex:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("t0").reset_index(drop=True)
        self._t0 = self.df["t0"].to_numpy(dtype=np.float64)
        self._t1 = self.df["t1"].to_numpy(dtype=np.float64)

    @staticmethod
    def build(roots: List[Path], glob_pattern: str = "**/waveforms/*.mseed", quiet: bool = True) -> "WaveformIndex":
        rows = []
        for root in roots:
            if not root.exists():
                continue
            for fp in root.glob(glob_pattern):
                if not fp.is_file():
                    continue
                try:
                    st = read(str(fp), headonly=True)
                    if not st:
                        continue
                    t0 = min(tr.stats.starttime for tr in st)
                    t1 = max(tr.stats.endtime   for tr in st)
                    rows.append({"path": str(fp), "t0": float(t0.timestamp), "t1": float(t1.timestamp)})
                except Exception as e:
                    if not quiet:
                        print(f"[index warn] {fp}: {e}")

        if not rows:
            raise RuntimeError("No se encontraron archivos .mseed (o no se pudieron leer headers).")

        return WaveformIndex(pd.DataFrame(rows))

    def save(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_pickle(out_path)

    @staticmethod
    def load(in_path: Path) -> "WaveformIndex":
        df = pd.read_pickle(in_path)
        return WaveformIndex(df)

    def files_for_window(self, t_start: UTCDateTime, t_end: UTCDateTime, pad_neighbors: int = 1) -> List[Path]:
        ts = float(t_start.timestamp)
        te = float(t_end.timestamp)

        j = bisect_right(self._t0, te)
        if j <= 0:
            return []

        mask = self._t1[:j] >= ts
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return []

        i0 = int(idx.min())
        i1 = int(idx.max())

        i0p = max(i0 - pad_neighbors, 0)
        i1p = min(i1 + pad_neighbors, len(self.df) - 1)

        return [Path(p) for p in self.df.loc[i0p:i1p, "path"].tolist()]


class StreamCache:
    def __init__(self):
        self._key = None
        self._st = None

    def get_stream(self, t_start: UTCDateTime, t_end: UTCDateTime, wf_index: WaveformIndex, pad_neighbors: int = 1):
        files = wf_index.files_for_window(t_start, t_end, pad_neighbors=pad_neighbors)
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


def get_trimmed_stream(cache: StreamCache, t_start: UTCDateTime, t_end: UTCDateTime,
                       wf_index: WaveformIndex, pad_neighbors: int = 1):
    st, files = cache.get_stream(t_start, t_end, wf_index, pad_neighbors=pad_neighbors)
    if st is None:
        return None, files

    stw = st.copy()
    stw.trim(t_start, t_end, pad=True, fill_value=0)
    stw.merge(fill_value="interpolate")
    return stw, files


# =========================
# PICKS: BEST FILTER (EVENT_ID OR TIME WINDOW)
# =========================

def picks_for_event(row_event: pd.Series, picks: pd.DataFrame,
                    t_start: UTCDateTime, t_end: UTCDateTime,
                    catc: CatalogCols = CatalogCols(),
                    pkc: PicksCols = PicksCols(),
                    prefer_event_id: bool = True,
                    min_prob: Optional[float] = None) -> pd.DataFrame:
    pw = picks.copy()

    used_id_match = False

    if prefer_event_id and (catc.event_id in row_event.index):
        ev_id = row_event[catc.event_id]

        if pd.notna(ev_id):
            # 1) Try event_id_mapped first
            pw_by = pd.DataFrame()
            if pkc.event_id_mapped in pw.columns:
                pw_by = pw[pw[pkc.event_id_mapped] == ev_id]

            # 2) If no match, try event_id
            if pw_by.empty and (pkc.event_id in pw.columns):
                pw_by = pw[pw[pkc.event_id] == ev_id]

            if not pw_by.empty:
                pw = pw_by
                used_id_match = True

    # 3) If no id match, fall back to time window
    if not used_id_match:
        pw = pw[(pw[pkc.time] >= pd.Timestamp(t_start.datetime, tz="UTC")) &
                (pw[pkc.time] <= pd.Timestamp(t_end.datetime,   tz="UTC"))]

    if min_prob is not None and (pkc.prob in pw.columns):
        pw = pw[pw[pkc.prob] >= float(min_prob)]

    return pw



def add_mpl_xnum(pw: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    pw = pw.copy()
    pw["xnum"] = mdates.date2num(pw[time_col].dt.tz_convert("UTC").dt.tz_localize(None))
    return pw


# =========================
# DISTANCES
# =========================

def stations_by_hypo_distance_km(stations: pd.DataFrame, ev_lat: float, ev_lon: float, ev_depth_km: float,
                                 stc: StationCols = StationCols()) -> Tuple[List[str], Dict[str, float], float]:
    if stc.elev_km not in stations.columns:
        st = stations.copy()
        st[stc.elev_km] = 0.0
    else:
        st = stations

    hypo = []
    for _, r in st.iterrows():
        d_m, _, _ = gps2dist_azimuth(ev_lat, ev_lon, float(r[stc.lat]), float(r[stc.lon]))
        horiz_km = d_m / 1000.0
        vert_km  = float(ev_depth_km) + float(r[stc.elev_km])
        hypo.append(float(np.hypot(horiz_km, vert_km)))

    tmp = st.copy()
    tmp["hypo_km"] = hypo
    tmp = tmp.sort_values("hypo_km")

    order = tmp[stc.station_id].tolist()
    dist_map = dict(zip(tmp[stc.station_id], tmp["hypo_km"]))
    dmax = float(tmp["hypo_km"].max()) if len(tmp) else 0.0
    return order, dist_map, dmax


# =========================
# TRACE: ENERGY
# =========================

def window_l2_energy(trace: obspy.Trace, t_abs: UTCDateTime, win_len: float) -> float:
    half = win_len / 2.0
    tr_win = trace.copy()
    tr_win.trim(t_abs - half, t_abs + half, pad=True, fill_value=0)
    data = tr_win.data.astype(float)
    return float(np.sum(data**2)) if data.size else 0.0


def _select_vertical_trace_for_sid(st, sid: str):
    """
    sid: 'NET.STA.LOC.PREFIX'  e.g. 'AV.DT1..BH'
    Return (trace_copy, chosen_channel) or (None, None)
    """
    parts = sid.split(".")
    if len(parts) < 4:
        return None, None

    net, sta, loc, chprefix = parts[0], parts[1], parts[2], parts[3]

    # 1) Try exact prefix (BH -> BHZ)
    candidates = [f"{chprefix}Z", "BHZ", "EHZ", "SHZ", "HHZ"]

    for ch in candidates:
        tr = st.select(network=net, station=sta, location=loc, channel=ch)
        if len(tr) > 0:
            return tr[0].copy(), tr[0].stats.channel

    # 2) Any vertical channel
    tr = st.select(network=net, station=sta, location=loc, channel="*Z")
    if len(tr) > 0:
        return tr[0].copy(), tr[0].stats.channel

    # 3) Any channel with the prefix (BH*)
    tr = st.select(network=net, station=sta, location=loc, channel=f"{chprefix}*")
    if len(tr) > 0:
        return tr[0].copy(), tr[0].stats.channel

    return None, None


def plot_event_moveout_picks_only(
    row_event: pd.Series,
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cache: "StreamCache",
    wf_index: "WaveformIndex",
    # window
    pre_s: float = 10.0,
    post_s: float = 30.0,
    min_prob: Optional[float] = None,
    # filter
    freqmin: Optional[float] = None,
    freqmax: Optional[float] = None,
    corners: int = 4,
    zerophase: bool = True,
    # pick style
    p_style: str = "vline",     # "vline" | "marker"
    s_style: str = "vline",
    p_color: str = "black",
    s_color: str = "red",
    marker_size: float = 4.0,
    line_width: float = 1.2,
    line_height_km: Optional[float] = None,
    # wiggles
    wiggle_frac_of_spacing: float = 0.35,
    y_pad_km: Optional[float] = None,
    # output
    save_png: Optional[Path] = None,
    figsize: Tuple[float, float] = (10, 6),
):
    """
    Moveout (wiggles + picks) using only stations that have picks for this event.
    X = UTC time (matplotlib datenums), Y = hypocentral distance (km).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timezone

    catc = CatalogCols()
    pkc = PicksCols()

    # -------- event and window --------
    t0 = UTCDateTime(row_event[catc.time].to_pydatetime())
    t_start = t0 - float(pre_s)
    t_end   = t0 + float(post_s)

    ev_lat = float(row_event[catc.lat])
    ev_lon = float(row_event[catc.lon])
    ev_depth_km = float(row_event.get(catc.depth_km, 0.0))

    # -------- event picks (id-mapped first) --------
    pw = picks_for_event(
        row_event=row_event,
        picks=picks,
        t_start=t_start,
        t_end=t_end,
        catc=catc,
        pkc=pkc,
        prefer_event_id=True,
        min_prob=min_prob,
    )
    if pw.empty:
        print("[moveout] No picks for this event/window.")
        return None, None, {"files": [], "stations_used": []}

    # -------- only stations with picks --------
    stations_with_picks = sorted(pw[pkc.station_id].dropna().unique().tolist())

    # -------- distances --------
    order_all, dist_map_all, _ = stations_by_hypo_distance_km(
        stations=stations,
        ev_lat=ev_lat,
        ev_lon=ev_lon,
        ev_depth_km=ev_depth_km,
        stc=StationCols(),
    )

    # Keep stations with picks and valid distance
    stations_with_picks = [sid for sid in stations_with_picks if sid in dist_map_all]
    if not stations_with_picks:
        print("[moveout] Picks exist, but none of their station ids are in dist_map.")
        return None, None, {"files": [], "stations_used": []}

    # Sort by distance (near -> far)
    picked_ids = sorted(stations_with_picks, key=lambda sid: dist_map_all.get(sid, np.inf))

    # -------- waveforms (index + cache) --------
    st, files = get_trimmed_stream(cache, t_start, t_end, wf_index, pad_neighbors=1)
    if st is None or len(st) == 0:
        print("[moveout] No waveforms in window.")
        return None, None, {"files": [str(f) for f in files], "stations_used": []}

    # -------- base x-axis --------
    t0_num = mdates.date2num(t0.datetime.replace(tzinfo=None))

    # -------- helper: prep trace -> (d_norm, x_abs) --------
    def prep_trace(tr):
        tf = tr.copy()
        tf.detrend("demean")
        tf.taper(max_percentage=0.05, type="cosine")

        if (freqmin is not None) and (freqmax is not None):
            sr = float(tf.stats.sampling_rate)
            nyq = 0.5 * sr
            fmin = max(0.001, min(float(freqmin), nyq * 0.99))
            fmax = max(fmin + 0.001, min(float(freqmax), nyq * 0.99))
            if fmax > fmin:
                tf.filter(
                    "bandpass",
                    freqmin=fmin,
                    freqmax=fmax,
                    corners=int(corners),
                    zerophase=bool(zerophase),
                )

        d = tf.data.astype(float)
        if d.size == 0:
            return None, None

        m = float(np.max(np.abs(d)))
        if (not np.isfinite(m)) or m == 0.0:
            return None, None

        d = d / m  # normalize for moveout
        x = t0_num + tf.times(reftime=t0) / 86400.0
        return d, x

    # -------- wiggle scale based on station spacing --------
    dvals = np.array([dist_map_all[sid] for sid in picked_ids], dtype=float)
    spac = np.diff(dvals)
    spac = spac[spac > 0]
    med = float(np.median(spac)) if spac.size else 1.0

    amp_km = float(wiggle_frac_of_spacing) * med
    vline_h = float(line_height_km) if line_height_km is not None else 0.8 * med
    ypad = float(y_pad_km) if y_pad_km is not None else 0.5 * med

    # -------- plot --------
    fig, ax = plt.subplots(figsize=figsize)

    # Cache for marker placement on wiggles
    wig_cache = {}   # sid -> (x_arr, d_arr, y0)
    used_y = []

    # Plot only stations with picks
    for sid in picked_ids:
        tr, chosen_ch = _select_vertical_trace_for_sid(st, sid)
        if tr is None:
            continue

        d, x = prep_trace(tr)
        if d is None:
            continue

        y0 = float(dist_map_all[sid])
        ax.plot(x, y0 + amp_km * d, lw=0.6, alpha=0.85, zorder=1)
        wig_cache[sid] = (x, d, y0)
        used_y.append(y0)

    if not used_y:
        print("[moveout] No usable traces for stations-with-picks (nothing plotted).")
        return None, None, {"files": [str(f) for f in files], "stations_used": []}

    # -------- keep picks only for plotted stations --------
    pw = pw[pw[pkc.station_id].isin(wig_cache.keys())].copy()
    if pw.empty:
        print("[moveout] Waveforms plotted, but no picks belong to those plotted stations.")
        # Wiggles can still be shown
    else:
        
        pw = add_mpl_xnum(pw, time_col=pkc.time)

    # -------- helper for marker on wiggle --------
    def samp_on_wiggle(sid, xval):
        if sid not in wig_cache:
            return None
        x_arr, d_arr, y0 = wig_cache[sid]
        if xval < x_arr[0] or xval > x_arr[-1]:
            return None
        return y0 + amp_km * np.interp(xval, x_arr, d_arr)

    # -------- draw picks --------
    def draw_phase(phase_letter: str, style: str, color: str):
        if pw.empty:
            return
        m = pw[pkc.phase].astype(str).str.upper().eq(phase_letter)
        if not m.any():
            return

        sids = pw.loc[m, pkc.station_id].to_numpy()
        xs   = pw.loc[m, "xnum"].to_numpy()

        label_once = True
        if style.lower() == "marker":
            ys = [samp_on_wiggle(s, x) for s, x in zip(sids, xs)]
            xp = [x for x, y in zip(xs, ys) if y is not None]
            yp = [y for y in ys if y is not None]
            ax.plot(
                xp, yp,
                linestyle="none",
                marker=("o" if phase_letter == "P" else "s"),
                ms=marker_size,
                color=color,
                label=(phase_letter if label_once else ""),
                zorder=3,
            )
        else:  # vline
            half = 0.5 * vline_h
            for s, x in zip(sids, xs):
                if s not in wig_cache:
                    continue
                y0 = wig_cache[s][2]
                ax.vlines(
                    x, y0 - half, y0 + half,
                    colors=color,
                    linewidth=line_width,
                    label=(phase_letter if label_once else ""),
                    zorder=3,
                )
                label_once = False

    draw_phase("P", p_style, p_color)
    draw_phase("S", s_style, s_color)

    # -------- formatting --------
    ax.set_ylim(min(used_y) - ypad, max(used_y) + ypad)
    ax.axvline(t0_num, ls="--", lw=0.8)

    ax.set_ylabel("Distance (km)")
    ax.set_xlabel("UTC time")

    # Title
    eid = row_event.get(catc.event_id, "")
    ax.set_title(f"Moveout | event_id={eid} | {row_event[catc.time]}")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=timezone.utc))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.tick_params(axis="x", labelrotation=0)
    ax.legend(loc="best")

    fig.tight_layout()

    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=400, bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] saved: {save_png}")
        return None, None, {"files": [str(f) for f in files], "stations_used": list(wig_cache.keys())}

    return fig, ax, {"files": [str(f) for f in files], "stations_used": list(wig_cache.keys())}

# SPECTROGRAMS


def plot_event_spectrograms_vertical_picks_only(
    row_event: pd.Series,
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cache: "StreamCache",
    wf_index: "WaveformIndex",
    # window around origin time
    pre_s: float = 30.0,
    post_s: float = 90.0,
    min_prob: Optional[float] = None,
    # STFT / spectrogram
    window_s: float = 2.0,
    overlap_frac: float = 0.8,
    freq_min: float = 0.0,
    freq_max: float = 20.0,
    # optional filter (also applied to the trace)
    bp_freqmin: Optional[float] = None,
    bp_freqmax: Optional[float] = None,
    corners: int = 4,
    zerophase: bool = True,
    # color scaling
    db_pmin: float = 5.0,
    db_pmax: float = 98.0,
    global_clim: bool = True,
    # style
    time_color: str = "navy",
    p_color: str = "black",
    s_color: str = "red",
    pick_lw: float = 1.2,
    origin_color: str = "red",
    origin_ls: str = "--",
    # output
    save_png: Optional[Path] = None,
    dpi: int = 300,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    One figure per event:
      - rows: stations with picks (for this event), sorted by hypocentral distance (near -> far)
      - col 0: Z trace (time)
      - col 1: Z spectrogram (STFT)
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # SciPy (preferred). Fallback if not available.
    try:
        from scipy.signal import spectrogram as scipy_spectrogram
        _HAS_SCIPY = True
    except Exception:
        _HAS_SCIPY = False

    catc = CatalogCols()
    pkc = PicksCols()

    # --- event and window ---
    t0 = UTCDateTime(row_event[catc.time].to_pydatetime())
    t_start = t0 - float(pre_s)
    t_end   = t0 + float(post_s)
    t0_num = mdates.date2num(t0.datetime.replace(tzinfo=None))

    ev_lat = float(row_event[catc.lat])
    ev_lon = float(row_event[catc.lon])
    ev_depth_km = float(row_event.get(catc.depth_km, 0.0))
    eid = row_event.get(catc.event_id, "")

    # --- event picks (id-mapped first) ---
    pw = picks_for_event(
        row_event=row_event,
        picks=picks,
        t_start=t_start,
        t_end=t_end,
        catc=catc,
        pkc=pkc,
        prefer_event_id=True,
        min_prob=min_prob,
    )
    if pw.empty:
        print("[spec] No picks for this event/window.")
        return None, None, {"files": [], "stations_used": []}

    # --- stations with picks ---
    stations_with_picks = sorted(pw[pkc.station_id].dropna().unique().tolist())

    # --- distances (to sort near -> far) ---
    _, dist_map_all, _ = stations_by_hypo_distance_km(
        stations=stations,
        ev_lat=ev_lat,
        ev_lon=ev_lon,
        ev_depth_km=ev_depth_km,
        stc=StationCols(),
    )

    stations_with_picks = [sid for sid in stations_with_picks if sid in dist_map_all]
    if not stations_with_picks:
        print("[spec] Picks exist, but none of their station ids are in dist_map.")
        return None, None, {"files": [], "stations_used": []}

    picked_ids = sorted(stations_with_picks, key=lambda sid: dist_map_all.get(sid, np.inf))
    nsta = len(picked_ids)

    # --- waveforms (index + cache) ---
    st, files = get_trimmed_stream(cache, t_start, t_end, wf_index, pad_neighbors=1)
    if st is None or len(st) == 0:
        print("[spec] No waveforms in window.")
        # Still build a "no data" figure
        st = None

    # --- prepare picks for drawing P/S lines ---
    pw = pw.copy()
    pw = pw[pw[pkc.station_id].isin(picked_ids)]
    if not pw.empty:
        pw = add_mpl_xnum(pw, time_col=pkc.time)

    # --- layout ---
    if figsize is None:
        # Height per station: ~2.3"
        figsize = (14.0, max(3.0, 2.3 * nsta))

    fig, axes = plt.subplots(
        nrows=nsta, ncols=2,
        figsize=figsize,
        sharex="col",
        gridspec_kw={"wspace": 0.15, "hspace": 0.25},
    )
    if nsta == 1:
        axes = np.array([axes])  # force shape (1,2)

    # --- helper: prep trace (detrend/taper + optional filter) ---
    def prep_trace_for_plot(tr):
        tf = tr.copy()
        tf.detrend("demean")
        tf.taper(max_percentage=0.05, type="cosine")

        if (bp_freqmin is not None) and (bp_freqmax is not None):
            sr = float(tf.stats.sampling_rate)
            nyq = 0.5 * sr
            fmin = max(0.001, min(float(bp_freqmin), nyq * 0.99))
            fmax = max(fmin + 0.001, min(float(bp_freqmax), nyq * 0.99))
            if fmax > fmin:
                tf.filter(
                    "bandpass",
                    freqmin=fmin,
                    freqmax=fmax,
                    corners=int(corners),
                    zerophase=bool(zerophase),
                )
        return tf

    # --- 1) precompute spectrograms (for global clim) ---
    spec_store = {}  # sid -> dict for plotting
    all_db_vals = []

    for sid in picked_ids:
        if st is None:
            spec_store[sid] = {"status": "no_data"}
            continue

        tr, chosen_ch = _select_vertical_trace_for_sid(st, sid)
        if tr is None or len(tr.data) == 0:
            spec_store[sid] = {"status": "no_data"}
            continue

        tr = tr.copy()
        tr.trim(starttime=t_start, endtime=t_end, pad=True, fill_value=0)
        if len(tr.data) == 0:
            spec_store[sid] = {"status": "no_data"}
            continue

        tf = prep_trace_for_plot(tr)
        fs = float(tf.stats.sampling_rate)

        nperseg = max(8, int(fs * float(window_s)))
        noverlap = int(nperseg * float(overlap_frac))
        noverlap = min(max(0, noverlap), nperseg - 1)

        if _HAS_SCIPY:
            f, t_seg, Sxx = scipy_spectrogram(
                tf.data.astype(float),
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                scaling="density",
                mode="psd",
            )
        else:
            # Simple fallback using matplotlib.specgram
            Pxx, f, t_seg, _ = axes[0, 1].specgram(
                tf.data.astype(float),
                NFFT=nperseg,
                Fs=fs,
                noverlap=noverlap,
                scale="dB",
                mode="psd",
            )
            # specgram returns dB in the plotted image; keep it as "Sxx_dB"
            Sxx = Pxx
            Sxx_dB = Sxx
            seg_times = [tf.stats.starttime + tt for tt in t_seg]
            seg_dates = mdates.date2num([stt.datetime.replace(tzinfo=None) for stt in seg_times])
            spec_store[sid] = {
                "status": "ok",
                "tr": tf,
                "chosen_ch": chosen_ch,
                "f": f,
                "t_seg": t_seg,
                "seg_dates": seg_dates,
                "Sxx_dB": Sxx_dB,
                "fs": fs,
            }
            all_db_vals.append(Sxx_dB[np.isfinite(Sxx_dB)].ravel())
            continue

        Sxx_dB = 10.0 * np.log10(Sxx + 1e-20)
        seg_times = [tf.stats.starttime + tt for tt in t_seg]
        seg_dates = mdates.date2num([stt.datetime.replace(tzinfo=None) for stt in seg_times])

        spec_store[sid] = {
            "status": "ok",
            "tr": tf,
            "chosen_ch": chosen_ch,
            "f": f,
            "t_seg": t_seg,
            "seg_dates": seg_dates,
            "Sxx_dB": Sxx_dB,
            "fs": fs,
        }
        all_db_vals.append(Sxx_dB[np.isfinite(Sxx_dB)].ravel())

    # Global clim (optional)
    vmin = vmax = None
    if global_clim:
        vals = np.concatenate(all_db_vals) if len(all_db_vals) else np.array([])
        if vals.size:
            vmin = float(np.percentile(vals, db_pmin))
            vmax = float(np.percentile(vals, db_pmax))

    # --- 2) plot each station (rows) ---
    stations_used = []
    last_im = None

    for r, sid in enumerate(picked_ids):
        ax_time = axes[r, 0]
        ax_spec = axes[r, 1]

        dist_km = float(dist_map_all.get(sid, np.nan))

        # Picks for this station
        pw_sid = pw[pw[pkc.station_id] == sid] if (pw is not None and not pw.empty) else pd.DataFrame()

        info = spec_store.get(sid, {"status": "no_data"})
        if info["status"] != "ok":
            ax_time.set_title(f"{sid} | {dist_km:.1f} km | NO DATA", color="gray", fontsize=11)
            ax_time.set_yticks([])
            ax_spec.set_yticks([])
            ax_time.axvline(t0_num, color=origin_color, linestyle=origin_ls, linewidth=1.1, alpha=0.85)
            ax_spec.axvline(t0_num, color=origin_color, linestyle=origin_ls, linewidth=1.1, alpha=0.85)
            continue

        tf = info["tr"]
        chosen_ch = info["chosen_ch"]
        f = info["f"]
        seg_dates = info["seg_dates"]
        Sxx_dB = info["Sxx_dB"]
        fs = float(info["fs"])

        stations_used.append(sid)

        # --- TIME TRACE ---
        times = tf.times("matplotlib")
        ax_time.plot(times, tf.data.astype(float), linewidth=0.8, color=time_color)
        ax_time.axvline(t0_num, color=origin_color, linestyle=origin_ls, linewidth=1.1, alpha=0.85)

        # P/S picks as lines
        if not pw_sid.empty:
            for _, rr in pw_sid.iterrows():
                ph = str(rr[pkc.phase]).upper()
                x = float(rr["xnum"])
                if ph == "P":
                    ax_time.axvline(x, color=p_color, linewidth=pick_lw, alpha=0.9)
                elif ph == "S":
                    ax_time.axvline(x, color=s_color, linewidth=pick_lw, alpha=0.9)

        ax_time.set_title(f"{sid} | {dist_km:.1f} km | {chosen_ch}", fontsize=11)
        ax_time.grid(True, linestyle=":", alpha=0.5)
        if r == nsta - 1:
            ax_time.set_xlabel("Time (UTC)")

        # --- SPECTROGRAM ---
        # Use pcolormesh; shading="auto" avoids manual bin issues
        im = ax_spec.pcolormesh(
            seg_dates,
            f,
            Sxx_dB,
            shading="auto",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        last_im = im

        ax_spec.axvline(t0_num, color=origin_color, linestyle=origin_ls, linewidth=1.1, alpha=0.85)

        # P/S picks on spectrogram
        if not pw_sid.empty:
            for _, rr in pw_sid.iterrows():
                ph = str(rr[pkc.phase]).upper()
                x = float(rr["xnum"])
                if ph == "P":
                    ax_spec.axvline(x, color=p_color, linewidth=pick_lw, alpha=0.9)
                elif ph == "S":
                    ax_spec.axvline(x, color=s_color, linewidth=pick_lw, alpha=0.9)

        ax_spec.set_ylim(float(freq_min), min(float(freq_max), fs / 2.0))
        if r == nsta - 1:
            ax_spec.set_xlabel("Time (UTC)")
        ax_spec.set_ylabel("Freq [Hz]")

    # --- global x formatting ---
    date_format = mdates.DateFormatter("%H:%M:%S")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)

    for r in range(nsta):
        for c in range(2):
            axes[r, c].xaxis.set_major_locator(locator)
            axes[r, c].xaxis.set_major_formatter(date_format)
            axes[r, c].tick_params(axis="x", labelrotation=25)

    # --- common labels ---
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 1].set_title("Spectrogram (Z)", fontsize=11)

    fig.suptitle(
        f"Event spectrograms, vertical Z) | event_id={eid} | {row_event[catc.time]} | window=[-{pre_s}s, +{post_s}s]",
        fontsize=12,
        y=0.995,
    )

    # One colorbar for the full figure
    # Reserve space on the right
    fig.tight_layout(rect=[0, 0, 0.90, 0.98])

    # Add a dedicated axis for the colorbar (single colorbar)
    if last_im is not None:
        cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])  # [left, bottom, width, height]
        cbar = fig.colorbar(last_im, cax=cax)
        cbar.set_label("PSD (dB re 1 count$^2$/Hz)")




    meta = {"files": [str(f) for f in files] if files else [], "stations_used": stations_used}

    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        print(f"[ok] saved: {save_png}")
        return None, None, meta

    return fig, axes, meta


# SEISMICITY 3D

def plot_catalog_3d_topo_seismicity(
    catalog: pd.DataFrame,
    *,
    catc: CatalogCols = CatalogCols(),
    mode: str = "time",              # "time" | "year_month"
    year: Optional[int] = None,      # required if mode="year_month"
    dem_res: str = "15s",
    topo_exagg: float = 5.0,
    topo_opacity: float = 0.98,
    colorscale: str = "Spectral",
    show: bool = False,
    renderer: Optional[str] = "notebook_connected",
    save_image: Optional[Path] = None,
    image_format: str = "svg",
    image_width: int = 1200,
    image_height: int = 900,
    image_engine: str = "kaleido",
):
    """
    3D catalog plot:
      - mode="year_month": filter by year and color by month
      - mode="time": color by time (old -> new)
      - DEM (pygmt) + Surface (plotly) + Scatter3d (plotly)
      - interactive if the renderer is set correctly
      - optional: fig.show() and fig.write_image()
    """
    import numpy as np
    import pandas as pd
    import pygmt
    import plotly.graph_objects as go

    # 1) Normalize to canonical columns (datetime/lat/lon/depth_km/magnitude if present)
    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()
    df[catc.time] = pd.to_datetime(df[catc.time], utc=True, errors="coerce")
    df = df.dropna(subset=[catc.time]).copy()

    mode = str(mode).lower().strip()
    if mode not in {"time", "year_month"}:
        raise ValueError("mode debe ser 'time' o 'year_month'.")

    # 2) Filter by year if needed
    if mode == "year_month":
        if year is None:
            raise ValueError("Para mode='year_month' debes pasar year=YYYY.")
        df = df[df[catc.time].dt.year == int(year)].copy()
        df["month"] = df[catc.time].dt.month.astype(int)

    if df.empty:
        raise ValueError("No hay eventos después del filtro (revisa year o datos NaN).")

    # 3) Bounding box (no padding)
    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    # 4) DEM using PyGMT
    grid = pygmt.datasets.load_earth_relief(dem_res, region=region)
    lon = grid["lon"].values
    lat = grid["lat"].values
    elev_km = grid.values / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)

    Z = -elev_km * float(topo_exagg)  # negative so the surface looks "above"

    surface = go.Surface(
        x=Lon,
        y=Lat,
        z=Z,
        colorscale="earth",
        showscale=False,
        opacity=float(topo_opacity),
        name="Topography",
    )

    # 5) Marker sizes from magnitude
    if catc.mag in df.columns:
        mags = pd.to_numeric(df[catc.mag], errors="coerce")
        mmin, mmax = float(mags.min()), float(mags.max())
        if np.isfinite(mmin) and np.isfinite(mmax) and (mmax > mmin):
            sizes = 1 + (mags - mmin) / (mmax - mmin) * 15
        else:
            sizes = np.full(len(df), 6.0)
    else:
        sizes = np.full(len(df), 6.0)

    # 6) Color values + colorbar
    if mode == "year_month":
        color_vals = df["month"].to_numpy()
        cmin, cmax = 1, 12
        cbar = dict(
            title=f"Month ({int(year)})",
            tickvals=list(range(1, 13)),
            ticktext=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        )
        title = f"Pavlof Volcano – {int(year)}"
    else:
        # Continuous time (old -> new)
        # Plotly needs numbers, so use epoch seconds
        tsec = (df[catc.time].astype("int64") / 1e9).to_numpy()
        color_vals = tsec
        cmin, cmax = float(np.min(tsec)), float(np.max(tsec))

        # Human-readable tick labels (quantiles)
        qs = np.linspace(0, 1, 6)
        tickvals = [float(np.quantile(tsec, q)) for q in qs]
        ticktext = [
            pd.to_datetime(v, unit="s", utc=True).strftime("%Y-%m-%d<br>%H:%M")
            for v in tickvals
        ]
        cbar = dict(title="Time (old → new)", tickvals=tickvals, ticktext=ticktext)
        title = f"Pavlof Volcano – time colored"

    # 7) Scatter3d (depth)
    z_depth = df[catc.depth_km] if catc.depth_km in df.columns else np.zeros(len(df))

    scatter = go.Scatter3d(
        x=df[catc.lon],
        y=df[catc.lat],
        z=z_depth,
        mode="markers",
        marker=dict(
            size=sizes,
            color=color_vals,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            colorbar=cbar,
            line=dict(width=0.2, color="black"),
            opacity=0.9,
        ),
        name="Earthquakes",
    )

    # 8) Camera settings
    r = np.sqrt(1.5**2 + 1.5**2 + 1.2**2)
    az_rad = np.deg2rad(310)
    el_rad = np.deg2rad(15)
    camera = {
        "center": {"x": 0, "y": 0, "z": 0},
        "eye": {
            "x": float(r * np.cos(el_rad) * np.cos(az_rad)),
            "y": float(r * np.cos(el_rad) * np.sin(az_rad)),
            "z": float(r * np.sin(el_rad)),
        },
        "up": {"x": 0, "y": 0, "z": 1},
    }

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Depth (km)",
            zaxis=dict(autorange="reversed"),
            aspectmode="auto",
            camera=camera,
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    fig = go.Figure(data=[surface, scatter], layout=layout)

    # 9) Show (optional)
    if show:
        if renderer is None:
            fig.show()
        else:
            fig.show(renderer=renderer)

    # 10) Save image (optional)
    if save_image is not None:
        fig.write_image(
            str(save_image),
            format=image_format,
            width=int(image_width),
            height=int(image_height),
            engine=image_engine,
        )

    return fig
