"""
Unified plotting and visualization helpers for QuickQuake/GAMMA seismic catalogs.

This file was consolidated from:
- qq_helpers.py
- qq_helpers_updated.py
- exploratory functions that were embedded in the notebook

Design intent:
- Keep the notebook thin.
- Put reusable logic here.
- Avoid duplicated helper files.
- Return fig/ax/meta objects instead of hiding state inside the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_right
from typing import Optional, Tuple, Sequence, Dict, Any, List

import numpy as np
import pandas as pd


# ============================================================
# SCHEMAS
# ============================================================

@dataclass(frozen=True)
class CatalogCols:
    time: str = "time"
    lat: str = "latitude"
    lon: str = "longitude"
    depth_km: str = "depth_km"
    mag: str = "magnitude"
    event_id: str = "event_id"


@dataclass(frozen=True)
class StationCols:
    station_id: str = "station"
    lat: str = "latitude"
    lon: str = "longitude"
    elev_km: str = "elev_km"


@dataclass(frozen=True)
class PicksCols:
    station_id: str = "id"
    time: str = "timestamp"
    phase: str = "type"
    prob: str = "prob"
    event_id: str = "event_id"
    event_id_mapped: str = "event_id_mapped"


# ============================================================
# FIXED COLOURS PER NETWORK
# ============================================================
# The same network always gets the same colour, in every figure and every project.
# Add a network here to fix its colour; any network not listed gets a colour from
# FALLBACK_COLOURS, chosen from its code (so it is also always the same).
NETWORK_COLOURS: Dict[str, str] = {
    "DK": "#000000",   # Danish national network (GEUS), permanent
    "GE": "#7f7f7f",   # GEOFON
    "IU": "#8c564b",   # Global Seismographic Network
    "3Z": "#1f77b4",   # Jakobshavn glacier arrays (ETH)
    "6H": "#d62728",   # GreenlandLakes2020 (WHOI)
    "YF": "#9467bd",   # Helheim (NYU)
    "9F": "#17becf",   # Isunnguata Sermia nodes
    "6R": "#2ca02c",   # SIIOS 2022
    "9C": "#bcbd22",   # SIIOS 2018
    "6D": "#e377c2",   # Bowdoin 2019
    "7Q": "#ff7f0e",   # GreenDrill
    "AV": "#1f77b4",   # Alaska Volcano Observatory (Pavlof example)
}
FALLBACK_COLOURS = ["#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354"]


def network_of(station_id: str) -> str:
    """Network code of a station id such as '3Z.SK01L..CH' -> '3Z'."""
    return str(station_id).split(".")[0]


def network_colour(network: str) -> str:
    """Fixed colour of a network (see NETWORK_COLOURS)."""
    if network in NETWORK_COLOURS:
        return NETWORK_COLOURS[network]
    return FALLBACK_COLOURS[sum(ord(ch) for ch in str(network)) % len(FALLBACK_COLOURS)]


def scatter_stations_by_network(ax, lons, lats, station_ids, *, size=125, marker="s", transform=None):
    """Draw stations with one fixed colour per network and one legend entry per network
    (with the number of stations of that network)."""
    ids = list(station_ids)
    nets = [network_of(i) for i in ids]
    lons, lats = np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    kw = {} if transform is None else {"transform": transform}
    for net in sorted(set(nets)):
        m = np.array([n == net for n in nets])
        ax.scatter(lons[m], lats[m], s=size, marker=marker, facecolor=network_colour(net),
                   edgecolor="w", linewidth=0.5, label=f"{net} ({int(m.sum())} stations)", **kw)


# ============================================================
# BASIC NORMALIZATION HELPERS
# ============================================================

def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_catalog(
    catalog: pd.DataFrame,
    *,
    catc: CatalogCols = CatalogCols(),
) -> pd.DataFrame:
    """
    Rename common column names to the canonical names defined in CatalogCols.
    Columns that already have the right name are left untouched.
    """
    if not isinstance(catalog, pd.DataFrame):
        raise TypeError("catalog must be a pandas.DataFrame")

    df = catalog.copy()

    rename_map: Dict[str, str] = {}

    time_src = _first_existing_column(
        df, [catc.time, "datetime", "origin_time", "date", "event_time"]
    )
    lat_src = _first_existing_column(
        df, [catc.lat, "lat", "latitude_deg", "y"]
    )
    lon_src = _first_existing_column(
        df, [catc.lon, "lon", "longitude_deg", "x"]
    )
    depth_src = _first_existing_column(
        df, [catc.depth_km, "depth", "depth_km", "z", "depth_km_final"]
    )
    mag_src = _first_existing_column(
        df, [catc.mag, "mag", "magnitude", "ml", "mw"]
    )
    event_src = _first_existing_column(
        df, [catc.event_id, "event_idx", "eventid", "eid", "id"]
    )

    if time_src and time_src != catc.time:
        rename_map[time_src] = catc.time
    if lat_src and lat_src != catc.lat:
        rename_map[lat_src] = catc.lat
    if lon_src and lon_src != catc.lon:
        rename_map[lon_src] = catc.lon
    if depth_src and depth_src != catc.depth_km:
        rename_map[depth_src] = catc.depth_km
    if mag_src and mag_src != catc.mag:
        rename_map[mag_src] = catc.mag
    if event_src and event_src != catc.event_id:
        rename_map[event_src] = catc.event_id

    if rename_map:
        df = df.rename(columns=rename_map)

    # conversiones suaves
    if catc.time in df.columns:
        df[catc.time] = pd.to_datetime(df[catc.time], utc=True, errors="coerce", format="mixed")
    if catc.lat in df.columns:
        df[catc.lat] = pd.to_numeric(df[catc.lat], errors="coerce")
    if catc.lon in df.columns:
        df[catc.lon] = pd.to_numeric(df[catc.lon], errors="coerce")
    if catc.depth_km in df.columns:
        df[catc.depth_km] = pd.to_numeric(df[catc.depth_km], errors="coerce")
    if catc.mag in df.columns:
        df[catc.mag] = pd.to_numeric(df[catc.mag], errors="coerce")
    if catc.event_id in df.columns:
        df[catc.event_id] = pd.to_numeric(df[catc.event_id], errors="coerce").astype("Int64")

    return df


# ============================================================
# I/O HELPERS
# ============================================================


def load_catalog_csv(path: Path | str, *, catc: Optional[CatalogCols] = None) -> pd.DataFrame:
    """
    Load a catalog CSV and normalize common QuickQuake/GAMMA/HypoDD columns.

    Parameters
    ----------
    path
        CSV file path.
    catc
        Optional catalog column schema.

    Returns
    -------
    pandas.DataFrame
        Catalog with canonical time/latitude/longitude/depth_km/magnitude columns
        when those columns can be inferred.
    """
    catc = catc or CatalogCols()
    df = pd.read_csv(path)
    return normalize_catalog(df, catc=catc)


def load_stations_json(path: Path | str) -> pd.DataFrame:
    """
    Load a QuickQuake/GAMMA station JSON into a DataFrame.

    Supported JSON shapes:
    - list[dict]
    - {"stations": list[dict]}
    - dict keyed by station id
    """
    import json

    with open(path, "r") as f:
        stations_obj = json.load(f)

    if isinstance(stations_obj, list):
        df = pd.DataFrame(stations_obj)
    elif isinstance(stations_obj, dict) and "stations" in stations_obj:
        df = pd.DataFrame(stations_obj["stations"])
    elif isinstance(stations_obj, dict):
        df = pd.DataFrame(stations_obj).T.reset_index().rename(columns={"index": "station"})
    else:
        raise ValueError("Formato de STATIONS_JSON no reconocido")

    return df


def ensure_dir(path: Path | str) -> Path:
    """Create an output directory if it does not exist and return it as a Path."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


# ============================================================
# TIME AND MAP DATA HELPERS
# ============================================================

def filter_catalog_time(
    catalog: pd.DataFrame,
    start,
    end,
    cols: Optional[CatalogCols] = None,
) -> pd.DataFrame:
    """
    Return the catalog between [start, end] (UTC). start/end can be str or datetime.
    """
    cols = cols or CatalogCols()
    df = normalize_catalog(catalog, catc=cols)

    if cols.time not in df.columns:
        raise KeyError(f"Time column '{cols.time}' not found in catalog.")

    t = pd.to_datetime(df[cols.time], utc=True, errors="coerce")
    start = pd.to_datetime(start, utc=True)
    end = pd.to_datetime(end, utc=True)

    m = (t >= start) & (t <= end)
    return df.loc[m].copy()


def filter_catalog_by_radius(
    catalog,
    center_lat,
    center_lon,
    radius_km,
    lat_col="latitude",
    lon_col="longitude",
    distance_col="r_km",
):
    """
    Keep the events whose horizontal distance to (center_lat, center_lon) is <= radius_km.

    Distance is the great-circle distance on the WGS84 ellipsoid (ObsPy
    gps2dist_azimuth), in km; depth is ignored. The distance of each kept event
    is stored in `distance_col` (km). Rows with missing coordinates are dropped.
    Example: filter_catalog_by_radius(cat, 56.9058, -158.209, 20.0) keeps the
    events within 20 km of Aniakchak.
    """
    from obspy.geodetics.base import gps2dist_azimuth

    if radius_km <= 0:
        raise ValueError("radius_km must be greater than zero.")

    df = catalog.copy()

    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

    df = df.dropna(subset=[lat_col, lon_col]).copy()

    # gps2dist_azimuth returns (distance in m, azimuth, back azimuth)
    df[distance_col] = [
        gps2dist_azimuth(
            latitude,
            longitude,
            center_lat,
            center_lon,
        )[0] / 1000.0
        for latitude, longitude in zip(
            df[lat_col],
            df[lon_col],
        )
    ]

    df = df[df[distance_col] <= radius_km].copy()

    return df.reset_index(drop=True)


def make_quakes_for_map(
    catalog: pd.DataFrame,
    start,
    end,
    catc: Optional[CatalogCols] = None,
) -> pd.DataFrame:
    """
    Exact column layout expected by the map functions:
      quakes_for_map['lon'], ['lat'], ['depth'], ['mag'], ['datetime']
    """
    catc = catc or CatalogCols()
    df = filter_catalog_time(catalog, start, end, cols=catc)

    out = pd.DataFrame({
        "lon": pd.to_numeric(df[catc.lon], errors="coerce"),
        "lat": pd.to_numeric(df[catc.lat], errors="coerce"),
        "depth": pd.to_numeric(df[catc.depth_km], errors="coerce").fillna(0.0)
                 if catc.depth_km in df.columns else np.zeros(len(df)),
        "mag": pd.to_numeric(df[catc.mag], errors="coerce")
               if catc.mag in df.columns else np.full(len(df), np.nan),
        "datetime": pd.to_datetime(df[catc.time], utc=True, errors="coerce"),
    }).dropna(subset=["lon", "lat", "datetime"])

    return out


def make_stations_wide_for_map(
    stations: pd.DataFrame,
    stc: Optional[StationCols] = None,
) -> pd.DataFrame:
    """
    WIDE format, so this access pattern works:
        stations_t = stations_wide.T
        stations_t['longitude'], stations_t['latitude']
    """
    stc = stc or StationCols()
    df = stations.copy()

    rename_map = {}

    if stc.station_id not in df.columns:
        src = _first_existing_column(df, [stc.station_id, "station_id", "id", "sta", "station"])
        if src is not None:
            rename_map[src] = stc.station_id

    if stc.lon not in df.columns:
        src = _first_existing_column(df, [stc.lon, "longitude", "lon", "x"])
        if src is not None:
            rename_map[src] = stc.lon

    if stc.lat not in df.columns:
        src = _first_existing_column(df, [stc.lat, "latitude", "lat", "y"])
        if src is not None:
            rename_map[src] = stc.lat

    if stc.elev_km not in df.columns:
        src = _first_existing_column(df, [stc.elev_km, "elevation_km", "elev_km", "elev", "elevation", "elevation(m)", "elevation_m"])
        if src is not None:
            rename_map[src] = stc.elev_km

    if rename_map:
        df = df.rename(columns=rename_map)

    # If station elevations come in meters, convert them to km for the canonical column.
    if stc.elev_km in df.columns:
        df[stc.elev_km] = pd.to_numeric(df[stc.elev_km], errors="coerce")
        if "elevation(m)" in stations.columns and stc.elev_km != "elevation(m)":
            df[stc.elev_km] = df[stc.elev_km] / 1000.0

    if stc.station_id not in df.columns:
        raise KeyError(f"stations must have a '{stc.station_id}' column. Columns={list(df.columns)}")
    if stc.lon not in df.columns or stc.lat not in df.columns:
        raise KeyError(f"stations must have '{stc.lon}' and '{stc.lat}' columns. Columns={list(df.columns)}")

    cols = [stc.lon, stc.lat]
    if stc.elev_km in df.columns:
        cols.append(stc.elev_km)

    wide = df.set_index(stc.station_id)[cols].T.copy()
    rename_index = {stc.lon: "longitude", stc.lat: "latitude"}
    if stc.elev_km in wide.index:
        rename_index[stc.elev_km] = "elev_km"
    wide = wide.rename(index=rename_index)

    return wide


def ellipse_lonlat_from_km(lon0, lat0, a_km, b_km, theta_deg=0.0, n=200):
    """
    Ellipse centered on (lon0, lat0), semi-axes a_km/b_km (km),
    rotated by theta_deg (degrees, counter-clockwise). Local km->deg approximation.
    """
    theta = np.deg2rad(float(theta_deg))
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(float(lat0)))

    if np.isclose(km_per_deg_lon, 0.0):
        raise ValueError("km_per_deg_lon is ~0. Check lat0 (too close to a pole).")

    t = np.linspace(0, 2 * np.pi, int(n))
    ex = float(a_km) * np.cos(t)
    ey = float(b_km) * np.sin(t)

    x0 = ex * np.cos(theta) - ey * np.sin(theta)
    y0 = ex * np.sin(theta) + ey * np.cos(theta)

    lon_ell = float(lon0) + x0 / km_per_deg_lon
    lat_ell = float(lat0) + y0 / km_per_deg_lat
    return lon_ell, lat_ell


# ============================================================
# 2D MAP VIEWS
# ============================================================

# ============================================================
# MAP BACKGROUND TILES (ESRI World Shaded Relief)
# ============================================================
# cartopy downloads map tiles in parallel with urlopen() and NO timeout: one stalled connection
# blocks the figure forever (seen with cartopy 0.23 and 0.25). This tile source is cartopy's
# GoogleTiles with the ESRI shaded-relief URL, plus a timeout and retries per tile and an
# on-disk cache, so each tile is downloaded once and later maps/runs read it from disk.

TILE_CACHE_DIR = Path.home() / ".quickquake" / "map_tiles"   # next to the earth_relief cache
TILE_TIMEOUT_S = 20                                         # seconds to wait for one tile
TILE_RETRIES = 3                                            # attempts per tile before giving up
_SHADED_RELIEF_CLASS = None


def _shaded_relief_tile_class():
    """
    cartopy tile source for ESRI World Shaded Relief (built once, cartopy imported lazily).

    Same behaviour as cartopy's GoogleWTS.get_image, plus: each download times out after
    TILE_TIMEOUT_S seconds and is retried TILE_RETRIES times; tiles are cached in
    TILE_CACHE_DIR (a tile that could not be downloaded is drawn light grey, as cartopy
    does, and is not cached).
    """
    global _SHADED_RELIEF_CLASS
    if _SHADED_RELIEF_CLASS is not None:
        return _SHADED_RELIEF_CLASS

    import io
    import time
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen
    from PIL import Image
    from cartopy.io import img_tiles as cimgt

    class ShadedReliefESRI(cimgt.GoogleTiles):
        def __init__(self, cache=TILE_CACHE_DIR, **kwargs):
            # cache: a folder, True (cartopy's default folder) or False/None (no cache)
            super().__init__(cache=False if cache is None else cache, **kwargs)

        def _image_url(self, tile):
            x, y, z = tile
            return (
                "https://server.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg"
            )

        def get_image(self, tile):
            if self.cache_path is not None:
                cached_file = self._cache_dir / ("_".join(str(i) for i in tile) + ".npy")
            else:
                cached_file = None

            if cached_file in self.cache:
                img = np.load(cached_file, allow_pickle=False)
            else:
                img = None
                for attempt in range(TILE_RETRIES):
                    try:
                        request = Request(self._image_url(tile), headers={"User-Agent": self.user_agent})
                        with urlopen(request, timeout=TILE_TIMEOUT_S) as fh:
                            img = Image.open(io.BytesIO(fh.read()))
                            img.load()
                        break
                    except (HTTPError, URLError, OSError) as err:   # OSError includes timeouts
                        if attempt == TILE_RETRIES - 1:
                            print(f"[tiles] could not download tile {tile}: {err}")
                        else:
                            time.sleep(2.0)
                if img is None:
                    # light-grey tile, as cartopy draws a failed download; not cached
                    img = Image.fromarray(np.full((256, 256, 3), (250, 250, 250), dtype=np.uint8))
                    cached_file = None
                img = img.convert(self.desired_tile_form)
                if cached_file is not None:
                    np.save(cached_file, img, allow_pickle=False)
                    self.cache.add(cached_file)

            return img, self.tileextent(tile), "lower"

        def image_for_domain(self, target_domain, target_z):
            # Same as cartopy's, but the tiles are merged in a fixed order. cartopy merges them in
            # the order the parallel downloads finish, which changes a few edge pixels per run.
            import concurrent.futures

            def fetch_tile(tile):
                img, extent, origin = self.get_image(tile)
                img = np.array(img)
                x = np.linspace(extent[0], extent[1], img.shape[1])
                y = np.linspace(extent[2], extent[3], img.shape[0])
                return img, x, y, origin

            tiles = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self._MAX_THREADS) as executor:
                futures = [executor.submit(fetch_tile, tile)
                           for tile in self.find_images(target_domain, target_z)]
                for future in futures:            # request order, not completion order
                    try:
                        tiles.append(list(future.result()))
                    except OSError:
                        pass

            img, extent, origin = cimgt._merge_tiles(tiles)
            return img, extent, origin

    _SHADED_RELIEF_CLASS = ShadedReliefESRI
    return ShadedReliefESRI


def plot_mapview_seismicity(
    catalog,
    stations,
    start,
    end,
    *,
    depth_vmin=0.0,
    depth_vmax=40.0,
    cmap_name="gist_rainbow",
    size_k=22.0,
    marker_min=6.0,
    plot_ellipse=False,
    ellipse_lon=None,
    ellipse_lat=None,
    a_km=12.0,
    b_km=6.0,
    theta_deg=0.0,
    relief_zoom=12,
    pad_lon=0.25,
    pad_lat=0.15,
    step_lon=0.5,
    step_lat=0.25,
    labelsize=20,
    figsize=(15, 8),
    savefig=False,
    out_png=None,
    dpi=600,
    region_name=None,
    catc: Optional[CatalogCols] = None,
    stc: Optional[StationCols] = None,
):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    # ESRI shaded-relief tiles with download timeout, retries and disk cache (see _shaded_relief_tile_class)
    ShadedReliefESRI = _shaded_relief_tile_class()

    # catc / stc map other column names (e.g. a USGS catalog) onto the default ones
    quakes_for_map = make_quakes_for_map(catalog, start, end, catc=catc)
    stations_wide = make_stations_wide_for_map(stations, stc=stc)

    if quakes_for_map.empty:
        raise ValueError("No events in the range [start, end].")

    minlon = float(quakes_for_map["lon"].min()) - float(pad_lon)
    maxlon = float(quakes_for_map["lon"].max()) + float(pad_lon)
    minlat = float(quakes_for_map["lat"].min()) - float(pad_lat)
    maxlat = float(quakes_for_map["lat"].max()) + float(pad_lat)
    extent = (minlon, maxlon, minlat, maxlat)

    cmap = plt.get_cmap(str(cmap_name), 256)
    norm = mpl.colors.Normalize(float(depth_vmin), float(depth_vmax))

    center_lon = (minlon + maxlon) / 2
    center_lat = (minlat + maxlat) / 2

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.LambertAzimuthalEqualArea(
                central_longitude=center_lon,
                central_latitude=center_lat
            )
        }
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(ShadedReliefESRI(), int(relief_zoom))

    mag = quakes_for_map["mag"].to_numpy(dtype=float)
    sizes = float(size_k) * mag
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    sc = ax.scatter(
        quakes_for_map["lon"], quakes_for_map["lat"],
        s=sizes,
        c=quakes_for_map["depth"],
        cmap=cmap, norm=norm,
        edgecolor="w", linewidth=0.3,
        transform=ccrs.PlateCarree(),
        alpha=0.9,
        label="Earthquakes"
    )

    stations_t = stations_wide.T
    scatter_stations_by_network(ax, stations_t["longitude"], stations_t["latitude"], stations_t.index,
                                size=125, marker="s", transform=ccrs.PlateCarree())

  

    if plot_ellipse:
        if ellipse_lon is None:
            ellipse_lon = float(np.nanmedian(quakes_for_map["lon"]))
        if ellipse_lat is None:
            ellipse_lat = float(np.nanmedian(quakes_for_map["lat"]))

        lon_ell, lat_ell = ellipse_lonlat_from_km(
            ellipse_lon, ellipse_lat, a_km, b_km, theta_deg=theta_deg, n=200
        )

        ax.plot(
            lon_ell, lat_ell,
            transform=ccrs.PlateCarree(),
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Reference ellipse"
        )

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": labelsize, "color": "black"}
    gl.ylabel_style = {"size": labelsize, "color": "black", "rotation": 90}

    gl.xlocator = mticker.FixedLocator(
        np.arange(np.floor(minlon), np.ceil(maxlon) + 1e-6, float(step_lon))
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(minlat), np.ceil(maxlat) + 1e-6, float(step_lat))
    )
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.015, aspect=40)
    cbar.set_label("Depth (km)", fontsize=20)
    ticks = np.linspace(float(depth_vmin), float(depth_vmax), 9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t)}" for t in ticks])
    cbar.ax.tick_params(labelsize=16)

    ax.legend(
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=0.8,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4
    )

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    if savefig:
        if out_png is None:
            out_png = f"enhanced_seismicity_{start_dt.date()}_{end_dt.date()}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    title = f"{region_name} seismicity" if region_name else "Seismicity"
    print(
        f"{title} map from {start_dt.date()} to {end_dt.date()} "
        "(Size = Magnitude | Color = Depth)"
    )

    
    return fig, ax


def plot_mapview_basemap_stations(
    catalog,
    stations,
    start,
    end,
    *,
    relief_zoom=12,
    pad_lon=0.25,
    pad_lat=0.15,
    step_lon=0.5,
    step_lat=0.25,
    labelsize=20,
    figsize=(15, 8),
    savefig=False,
    out_png=None,
    dpi=600,
    catc: Optional[CatalogCols] = None,
    stc: Optional[StationCols] = None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    # ESRI shaded-relief tiles with download timeout, retries and disk cache (see _shaded_relief_tile_class)
    ShadedReliefESRI = _shaded_relief_tile_class()

    # Use catalog ONLY to keep the exact same map bounds
    # catc / stc map other column names (e.g. a USGS catalog) onto the default ones
    quakes_for_map = make_quakes_for_map(catalog, start, end, catc=catc)
    stations_wide = make_stations_wide_for_map(stations, stc=stc)

    if quakes_for_map.empty:
        raise ValueError("No events to compute the map extent.")

    minlon = float(quakes_for_map["lon"].min()) - float(pad_lon)
    maxlon = float(quakes_for_map["lon"].max()) + float(pad_lon)
    minlat = float(quakes_for_map["lat"].min()) - float(pad_lat)
    maxlat = float(quakes_for_map["lat"].max()) + float(pad_lat)
    extent = (minlon, maxlon, minlat, maxlat)

    center_lon = (minlon + maxlon) / 2
    center_lat = (minlat + maxlat) / 2

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.LambertAzimuthalEqualArea(
                central_longitude=center_lon,
                central_latitude=center_lat,
            )
        },
    )

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(ShadedReliefESRI(), int(relief_zoom))

    stations_t = stations_wide.T
    scatter_stations_by_network(ax, stations_t["longitude"], stations_t["latitude"], stations_t.index,
                                size=125, marker="s", transform=ccrs.PlateCarree())

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": labelsize, "color": "black"}
    gl.ylabel_style = {"size": labelsize, "color": "black", "rotation": 90}

    gl.xlocator = mticker.FixedLocator(
        np.arange(np.floor(minlon), np.ceil(maxlon) + 1e-6, float(step_lon))
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(minlat), np.ceil(maxlat) + 1e-6, float(step_lat))
    )

    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    ax.legend(
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=0.8,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4,
    )

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    if savefig:
        if out_png is None:
            out_png = f"basemap_stations_{start_dt.date()}_{end_dt.date()}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    print(f"Basemap from {start_dt.date()} to {end_dt.date()} without seismicity")

    return fig, ax


def plot_mapview_seismicity_zoom(
    catalog,
    start,
    end,
    *,
    depth_vmin=0.0,
    depth_vmax=40.0,
    cmap_name="gist_rainbow",
    size_k=22.0,
    marker_min=6.0,
    plot_ellipse=False,
    ellipse_lon=None,
    ellipse_lat=None,
    a_km=12.0,
    b_km=6.0,
    theta_deg=0.0,
    relief_zoom=12,
    pad_lon=0.25,
    pad_lat=0.15,
    step_lon=0.5,
    step_lat=0.25,
    labelsize=20,
    figsize=(15, 8),
    savefig=False,
    out_png=None,
    dpi=600,
    # nuevo
    zoom_factor=None,
    xlim=None,
    ylim=None,
    focal_length=None,
    show=False,
    depth_filter_min=None,
    depth_filter_max=None,
    catc: Optional[CatalogCols] = None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    # ESRI shaded-relief tiles with download timeout, retries and disk cache (see _shaded_relief_tile_class)
    ShadedReliefESRI = _shaded_relief_tile_class()

    # catc maps other column names (e.g. a USGS catalog) onto the default ones
    quakes_for_map = make_quakes_for_map(catalog, start, end, catc=catc)

    if quakes_for_map.empty:
        raise ValueError("No events in the range [start, end].")

    # Optional depth window (km, positive down); the map extent follows the kept events
    if depth_filter_min is not None:
        quakes_for_map = quakes_for_map[
            quakes_for_map["depth"] >= float(depth_filter_min)
        ].copy()

    if depth_filter_max is not None:
        quakes_for_map = quakes_for_map[
            quakes_for_map["depth"] <= float(depth_filter_max)
        ].copy()

    if quakes_for_map.empty:
        raise ValueError(
            "No events remain after applying the depth filter. "
            "Check depth_filter_min and depth_filter_max."
        )

    # Extent base
    minlon_data = float(quakes_for_map["lon"].min())
    maxlon_data = float(quakes_for_map["lon"].max())
    minlat_data = float(quakes_for_map["lat"].min())
    maxlat_data = float(quakes_for_map["lat"].max())

    minlon = minlon_data - float(pad_lon)
    maxlon = maxlon_data + float(pad_lon)
    minlat = minlat_data - float(pad_lat)
    maxlat = maxlat_data + float(pad_lat)

    # Automatic zoom around the center
    if zoom_factor is not None:
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be > 0")

        clon = 0.5 * (minlon + maxlon)
        clat = 0.5 * (minlat + maxlat)

        half_width = 0.5 * (maxlon - minlon) / float(zoom_factor)
        half_height = 0.5 * (maxlat - minlat) / float(zoom_factor)

        minlon = clon - half_width
        maxlon = clon + half_width
        minlat = clat - half_height
        maxlat = clat + half_height

    # Zoom manual domina si se pasa
    if xlim is not None:
        minlon, maxlon = xlim
    if ylim is not None:
        minlat, maxlat = ylim

    extent = (minlon, maxlon, minlat, maxlat)

    cmap = plt.get_cmap(str(cmap_name), 256)
    norm = mpl.colors.Normalize(float(depth_vmin), float(depth_vmax))

    center_lon = (minlon + maxlon) / 2
    center_lat = (minlat + maxlat) / 2

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.LambertAzimuthalEqualArea(
                central_longitude=center_lon,
                central_latitude=center_lat
            )
        }
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(ShadedReliefESRI(), int(relief_zoom))

    mag = quakes_for_map["mag"].to_numpy(dtype=float)
    sizes = float(size_k) * mag
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    sc = ax.scatter(
        quakes_for_map["lon"], quakes_for_map["lat"],
        s=sizes,
        c=quakes_for_map["depth"],
        cmap=cmap, norm=norm,
        edgecolor="w", linewidth=0.3,
        transform=ccrs.PlateCarree(),
        alpha=0.9,
        label="Earthquakes"
    )

    if plot_ellipse:
        if ellipse_lon is None:
            ellipse_lon = float(np.nanmedian(quakes_for_map["lon"]))
        if ellipse_lat is None:
            ellipse_lat = float(np.nanmedian(quakes_for_map["lat"]))

        lon_ell, lat_ell = ellipse_lonlat_from_km(
            ellipse_lon, ellipse_lat, a_km, b_km, theta_deg=theta_deg, n=200
        )

        ax.plot(
            lon_ell, lat_ell,
            transform=ccrs.PlateCarree(),
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Ellipse"
        )

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": labelsize, "color": "black"}
    gl.ylabel_style = {"size": labelsize, "color": "black"}

    gl.xlocator = mticker.FixedLocator(
        np.arange(np.floor(minlon), np.ceil(maxlon) + 1e-6, float(step_lon))
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(minlat), np.ceil(maxlat) + 1e-6, float(step_lat))
    )
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.015, aspect=40)
    cbar.set_label("Depth (km)", fontsize=20)
    ticks = np.linspace(float(depth_vmin), float(depth_vmax), 9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t)}" for t in ticks])
    cbar.ax.tick_params(labelsize=16)

    ax.legend(
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=0.8,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4
    )


    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    if savefig:
        if out_png is None:
            out_png = f"enhanced_seismicity_zoom_{start_dt.date()}_{end_dt.date()}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    print(
        f"Seismicity map from {start_dt.date()} to {end_dt.date()} "
        "(Size = Magnitude | Color = Depth)"
    )

    if show:
        plt.show()

    meta = {
        "bounds": (minlon, maxlon, minlat, maxlat),
        "center": (center_lon, center_lat),
        "n_events": int(len(quakes_for_map)),
        "depth_filter_min": depth_filter_min,
        "depth_filter_max": depth_filter_max,
    }

    return fig, ax, meta


def plot_mapview_seismicity_zoom_2(
    catalog,
    start,
    end,
    *,
    depth_vmin=0.0,
    depth_vmax=40.0,
    cmap_name="gist_rainbow",
    size_k=22.0,
    marker_min=6.0,
    plot_ellipse=False,
    ellipse_lon=None,
    ellipse_lat=None,
    a_km=12.0,
    b_km=6.0,
    theta_deg=0.0,
    relief_zoom=12,
    pad_lon=0.25,
    pad_lat=0.15,
    step_lon=0.5,
    step_lat=0.25,
    labelsize=20,
    figsize=(15, 8),
    savefig=False,
    out_png=None,
    dpi=600,
    zoom_factor=None,
    xlim=None,
    ylim=None,
    focal_length=None,
    show=False,
    # nuevo
    depth_filter_min=None,
    depth_filter_max=None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    # ESRI shaded-relief tiles with download timeout, retries and disk cache (see _shaded_relief_tile_class)
    ShadedReliefESRI = _shaded_relief_tile_class()

    # filter by time first
    quakes_for_map = make_quakes_for_map(catalog, start, end)

    if quakes_for_map.empty:
        raise ValueError("No events in the range [start, end].")

    # -------------------------------
    # Depth filter
    # -------------------------------
    if depth_filter_min is not None:
        quakes_for_map = quakes_for_map[
            quakes_for_map["depth"] >= float(depth_filter_min)
        ].copy()

    if depth_filter_max is not None:
        quakes_for_map = quakes_for_map[
            quakes_for_map["depth"] <= float(depth_filter_max)
        ].copy()

    if quakes_for_map.empty:
        raise ValueError(
            "No events left after the depth filter. "
            "Check depth_filter_min/depth_filter_max."
        )

    # Extent base
    minlon_data = float(quakes_for_map["lon"].min())
    maxlon_data = float(quakes_for_map["lon"].max())
    minlat_data = float(quakes_for_map["lat"].min())
    maxlat_data = float(quakes_for_map["lat"].max())

    minlon = minlon_data - float(pad_lon)
    maxlon = maxlon_data + float(pad_lon)
    minlat = minlat_data - float(pad_lat)
    maxlat = maxlat_data + float(pad_lat)

    # Automatic zoom around the center
    if zoom_factor is not None:
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be > 0")

        clon = 0.5 * (minlon + maxlon)
        clat = 0.5 * (minlat + maxlat)

        half_width = 0.5 * (maxlon - minlon) / float(zoom_factor)
        half_height = 0.5 * (maxlat - minlat) / float(zoom_factor)

        minlon = clon - half_width
        maxlon = clon + half_width
        minlat = clat - half_height
        maxlat = clat + half_height

    # Zoom manual domina si se pasa
    if xlim is not None:
        minlon, maxlon = xlim
    if ylim is not None:
        minlat, maxlat = ylim

    extent = (minlon, maxlon, minlat, maxlat)

    cmap = plt.get_cmap(str(cmap_name), 256)
    norm = mpl.colors.Normalize(float(depth_vmin), float(depth_vmax))

    center_lon = (minlon + maxlon) / 2
    center_lat = (minlat + maxlat) / 2

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={
            "projection": ccrs.LambertAzimuthalEqualArea(
                central_longitude=center_lon,
                central_latitude=center_lat
            )
        }
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_image(ShadedReliefESRI(), int(relief_zoom))

    mag = quakes_for_map["mag"].to_numpy(dtype=float)
    sizes = float(size_k) * mag
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    sc = ax.scatter(
        quakes_for_map["lon"], quakes_for_map["lat"],
        s=sizes,
        c=quakes_for_map["depth"],
        cmap=cmap, norm=norm,
        edgecolor="w", linewidth=0.3,
        transform=ccrs.PlateCarree(),
        alpha=0.9,
        label="Earthquakes"
    )

    if plot_ellipse:
        if ellipse_lon is None:
            ellipse_lon = float(np.nanmedian(quakes_for_map["lon"]))
        if ellipse_lat is None:
            ellipse_lat = float(np.nanmedian(quakes_for_map["lat"]))

        lon_ell, lat_ell = ellipse_lonlat_from_km(
            ellipse_lon, ellipse_lat, a_km, b_km, theta_deg=theta_deg, n=200
        )

        ax.plot(
            lon_ell, lat_ell,
            transform=ccrs.PlateCarree(),
            color="black",
            linestyle="-",
            linewidth=1.5,
            label="Ellipse"
        )

    gl = ax.gridlines(draw_labels=True, linestyle="--", linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": labelsize, "color": "black"}
    gl.ylabel_style = {"size": labelsize, "color": "black"}

    gl.xlocator = mticker.FixedLocator(
        np.arange(np.floor(minlon), np.ceil(maxlon) + 1e-6, float(step_lon))
    )
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.floor(minlat), np.ceil(maxlat) + 1e-6, float(step_lat))
    )
    gl.xformatter = cticker.LongitudeFormatter()
    gl.yformatter = cticker.LatitudeFormatter()

    cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.015, aspect=40)
    cbar.set_label("Depth (km)", fontsize=20)
    ticks = np.linspace(float(depth_vmin), float(depth_vmax), 9)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(t)}" for t in ticks])
    cbar.ax.tick_params(labelsize=16)

    ax.legend(
        loc="upper left",
        fontsize=18,
        frameon=True,
        framealpha=0.8,
        borderpad=0.3,
        labelspacing=0.3,
        handlelength=1.0,
        handletextpad=0.4
    )

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    if savefig:
        if out_png is None:
            out_png = f"enhanced_seismicity_zoom_{start_dt.date()}_{end_dt.date()}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    print(
        f"Seismicity map from {start_dt.date()} to {end_dt.date()} "
        f"(Size = Magnitude | Color = Depth | "
        f"Depth filter = [{depth_filter_min}, {depth_filter_max}])"
    )

    if show:
        plt.show()

    meta = {
        "bounds": (minlon, maxlon, minlat, maxlat),
        "center": (center_lon, center_lat),
        "n_events": int(len(quakes_for_map)),
        "depth_filter_min": depth_filter_min,
        "depth_filter_max": depth_filter_max,
    }

    return fig, ax, meta


# ============================================================
# TIMELINE PLOTS
# ============================================================

def _to_naive_utc_dt(x):
    """Convierte timestamp (str/datetime) a datetime naive en UTC."""
    t = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(t):
        return None
    return t.tz_convert("UTC").tz_localize(None)


def _series_to_naive_utc(series):
    """Serie a datetime naive UTC."""
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return s.dt.tz_convert("UTC").dt.tz_localize(None)


def plot_timeline_histogram(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    spans: Sequence[Tuple[str, str, str]],
    time_col: Optional[str] = None,
    fi_col: str = "FI",
    fi_categories: Sequence[str] = ("VT", "LP", "NS"),
    fi_colors: Optional[Dict[str, str]] = None,
    original_folder_col: str = "original_folder",
    original_folder_fmt: str = "%Y%m%dT%H%M%S",
    n_bins: int = 90,
    figsize: Tuple[float, float] = (14, 4),
    base_fontsize: int = 15,
    xtick_fontsize: int = 18,
    xlabel_fontsize: int = 20,
    legend_fontsize: int = 15,
    hist_color: str = "black",
    band_cmap: str = "Set2",
    month_interval: int = 1,
    band_y0: float = 0.78,
    band_height: float = 0.06,
    band_gap: float = 0.01,
    show_legend: bool = False,
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 600,
):
    """
    Timeline con:
      - Histogram stacked by FI class (VT, LP, NS)
      - Stacked bands (axvspan) per type on top
      - Monthly ticks, rotated 45 degrees
      - Optional legend
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.patches as mpatches

    plt.rcParams.update({"font.size": int(base_fontsize)})

    if fi_colors is None:
        fi_colors = {
            "VT": "#1f77b4",
            "LP": "#d62728",
            "NS": "#7f7f7f",
        }

    df = catalog.copy()

    # -------------------------
    # tiempo
    # -------------------------
    if time_col is not None and time_col in df.columns:
        tseries = df[time_col]
    elif "datetime" in df.columns:
        tseries = df["datetime"]
    else:
        catc = CatalogCols()
        if catc.time in df.columns:
            tseries = df[catc.time]
        elif original_folder_col in df.columns:
            tseries = pd.to_datetime(
                df[original_folder_col].astype(str),
                format=original_folder_fmt,
                errors="coerce"
            )
        else:
            raise KeyError(
                "No time column found. Pass time_col=... or provide "
                f"'datetime' o '{original_folder_col}'."
            )

    df["__t__"] = _series_to_naive_utc(tseries)
    df = df.dropna(subset=["__t__"])

    start_dt = _to_naive_utc_dt(start)
    end_dt = _to_naive_utc_dt(end)
    if start_dt is None or end_dt is None:
        raise ValueError("start/end could not be converted to dates.")

    df = df[(df["__t__"] >= start_dt) & (df["__t__"] <= end_dt)].copy()

    # -------------------------
    # columna FI
    # -------------------------
    # If the catalog has no FI/event-type column, fall back to one generic
    # "EVENTS" category instead of failing. This is useful for QuickQuake/GAMMA
    # catalogs that only have time/lat/lon/depth/magnitude.
    if fi_col is None or fi_col not in df.columns:
        fi_col = "__event_type__"
        df[fi_col] = "EVENTS"
        fi_categories = ("EVENTS",)
        fi_colors = {"EVENTS": hist_color}
    else:
        df[fi_col] = df[fi_col].astype(str).str.strip().str.upper()

    # Ensure every requested FI category has a color.
    fi_colors = dict(fi_colors or {})
    for _fi in fi_categories:
        fi_colors.setdefault(_fi, hist_color)

    # -------------------------
    # colores spans superiores
    # -------------------------
    types = list(dict.fromkeys(desc for _, _, desc in spans))
    cmap = plt.get_cmap(band_cmap)
    palette = getattr(cmap, "colors", None)
    if palette is None:
        palette = [cmap(i) for i in np.linspace(0, 1, max(1, len(types)))]

    color_map = {t: palette[i % len(palette)] for i, t in enumerate(types)}

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # -------------------------
    # histogram stacked by FI class
    # -------------------------
    t_num = mdates.date2num(df["__t__"])
    bin_edges = np.linspace(mdates.date2num(start_dt), mdates.date2num(end_dt), int(n_bins) + 1)
    bin_widths = np.diff(bin_edges)

    bottom = np.zeros(int(n_bins), dtype=float)
    fi_handles = []
    counts_by_fi = {}

    for fi in fi_categories:
        vals = t_num[df[fi_col] == fi]
        counts, _ = np.histogram(vals, bins=bin_edges)
        counts_by_fi[fi] = counts

        ax.bar(
            bin_edges[:-1],
            counts,
            width=bin_widths,
            bottom=bottom,
            align="edge",
            color=fi_colors.get(fi, hist_color),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.95,
            label=fi,
        )

        bottom = bottom + counts
        fi_handles.append(mpatches.Patch(color=fi_colors.get(fi, hist_color), label=fi))

    total_counts = bottom.copy()

    ax.set_ylabel("Earthquakes")
    ax.grid(axis="y", alpha=0.3)

    ymax = float(np.max(total_counts)) if len(total_counts) else 1.0
    ax.set_ylim(0, ymax * 1.3)

    # -------------------------
    # bandas superiores
    # -------------------------
    band_pos = {}
    for i, t in enumerate(types):
        ymin = band_y0 + i * (band_height + band_gap)
        ymax_band = ymin + band_height
        band_pos[t] = (ymin, ymax_band)

    seen = set()
    span_handles_for_meta = []
    for t0, t1, desc in spans:
        t0_dt = _to_naive_utc_dt(t0)
        t1_dt = _to_naive_utc_dt(t1)
        if t0_dt is None or t1_dt is None:
            continue

        ymin_band, ymax_band = band_pos[desc]
        label = desc if desc not in seen else "_nolegend_"

        ax.axvspan(
            t0_dt, t1_dt,
            ymin=ymin_band, ymax=ymax_band,
            transform=ax.get_xaxis_transform(),
            color=color_map[desc],
            alpha=0.9,
            label=label,
        )

        if desc not in seen:
            span_handles_for_meta.append(mpatches.Patch(color=color_map[desc], label=desc))
        seen.add(desc)

    # -------------------------
    # ejes x
    # -------------------------
    ax.set_xlim(start_dt, end_dt)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=int(month_interval)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
        lab.set_fontsize(int(xtick_fontsize))

    fig.subplots_adjust(bottom=0.25)
    ax.set_xlabel("Date", fontsize=int(xlabel_fontsize))

    # -------------------------
    # optional legend
    # -------------------------
    legend_handles = fi_handles + span_handles_for_meta
    legend_labels = [h.get_label() for h in legend_handles]

    if show_legend:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            fontsize=int(legend_fontsize),
            frameon=False,
            ncol=1,
        )

    plt.tight_layout()

    if savefig:
        if out_png is None:
            out_png = "timeline_hist_combined.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    meta = {
        "n_events": int(len(df)),
        "types": types,
        "color_map": color_map,
        "fi_categories": list(fi_categories),
        "fi_colors": fi_colors,
        "counts_by_fi": counts_by_fi,
        "total_counts": total_counts,
        "legend_handles": legend_handles,
        "legend_labels": legend_labels,
    }
    return fig, ax, meta


def plot_timeline_legend_only(
    *,
    fi_categories: Sequence[str],
    fi_colors: Dict[str, str],
    span_types: Sequence[str],
    span_color_map: Dict[str, str],
    figsize: Tuple[float, float] = (4, 3),
    legend_fontsize: int = 16,
    ncol: int = 1,
    title: Optional[str] = None,
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 600,
):
    """
    Separate figure that contains only the legend
    for FI classes + spans.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    handles = []

    for fi in fi_categories:
        handles.append(mpatches.Patch(color=fi_colors[fi], label=fi))

    for desc in span_types:
        handles.append(mpatches.Patch(color=span_color_map[desc], label=desc))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=int(legend_fontsize),
        ncol=int(ncol),
    )

    if title is not None:
        ax.set_title(title, fontsize=int(legend_fontsize) + 1)

    plt.tight_layout()

    if savefig:
        if out_png is None:
            out_png = "timeline_legend_only.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", transparent=True)
        print(f"[ok] saved: {out_png}")

    return fig, ax


# ============================================================
# ELEVATION GRID (topography + bathymetry) WITHOUT GMT
# ============================================================
#
# The 3D views need an elevation grid. pygmt cannot be installed next to
# TensorFlow 2.14 (GMT needs newer system libraries), so load_earth_relief()
# below downloads the SAME files pygmt downloads (SRTM15+ v2.7, Tozer et al.,
# 2019, https://doi.org/10.1029/2019EA000658) from the GMT data server and
# reads them with requests + Pillow. On the Pavlof and East Greenland test
# regions it returns exactly the same nodes and values as
# pygmt.datasets.load_earth_relief.

# GMT data servers, tried in this order until one answers. Only the main server is
# listed: it is the default of pygmt/GMT too, and on 2026-10-17 the mirrors on
# https://www.generic-mapping-tools.org/mirrors/ answered "404 not found" for these tiles.
GMT_DATA_SERVERS = (
    "https://oceania.generic-mapping-tools.org",          # main server (U. of Hawaii)
)

# Resolutions stored on the server as square tiles (from the server table
# gmt_data_server.txt): resolution -> (tile size in degrees, registrations).
# "p" = pixel (one value per cell, at the cell centre); "g" = gridline (values on the grid lines).
EARTH_RELIEF_TILES = {
    "15s": (10, ("p",)),          # 15 arc-seconds (~460 m at the equator)
    "30s": (15, ("g", "p")),
    "01m": (30, ("g", "p")),      # 1 arc-minute (~1.85 km)
    "02m": (60, ("g", "p")),
    "03m": (90, ("g", "p")),
    "04m": (180, ("g", "p")),
    "05m": (180, ("g", "p")),
}
EARTH_RELIEF_SCALE_M = 0.5        # tiles store integers: metres = integer * 0.5
EARTH_RELIEF_CACHE = Path.home() / ".quickquake" / "earth_relief"


def _read_earth_relief_tile(url_path: str, cache_file: Path) -> np.ndarray:
    """
    One tile as elevation in metres, row 0 = SOUTH edge. Downloaded once, then read
    from cache_file. The tiles are JPEG2000 images where each pixel is an elevation.
    """
    import io
    import requests
    from PIL import Image

    if cache_file.exists():
        data = cache_file.read_bytes()
    else:
        data = None
        for server in GMT_DATA_SERVERS:
            try:
                response = requests.get(server + url_path, timeout=120)
            except requests.RequestException:
                continue                      # server down or unreachable: try the next one
            if response.status_code == 200:
                data = response.content
                print(f"[earth relief] downloaded {cache_file.name} ({len(data) / 1e6:.1f} MB) from {server}")
                break
        if data is None:
            raise RuntimeError(f"No GMT data server answered for {url_path}. Check the internet connection.")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)

    # Is the stored integer signed? Read it from the JPEG2000 header (SIZ segment, which
    # comes right after the start marker FF4F FF51): byte Ssiz, bit 7 = signed, bits 0-6 = bits - 1.
    siz = data.find(b"\xff\x4f\xff\x51")
    if siz < 0:
        raise ValueError(f"{cache_file} is not a JPEG2000 file; delete it and run again.")
    ssiz = data[siz + 2 + 40]
    signed, nbits = bool(ssiz & 0x80), (ssiz & 0x7F) + 1

    Image.MAX_IMAGE_PIXELS = None             # a 15s tile is 2400 x 2400 pixels: not an attack
    image = Image.open(io.BytesIO(data))
    if nbits != 16 or not image.mode.startswith("I;16"):
        raise ValueError(f"Unexpected tile format ({nbits} bits, Pillow mode {image.mode}) in {cache_file}.")
    raw = np.asarray(image).astype(np.int32)
    # Pillow returns signed 16-bit data shifted by +32768 (checked against GMT's own
    # conversion of a tile: identical after removing the shift).
    stored = raw - 2 ** (nbits - 1) if signed else raw
    return stored[::-1].astype(float) * EARTH_RELIEF_SCALE_M


def load_earth_relief(
    resolution: str = "15s",
    region: Sequence[float] = None,
    registration: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Elevation grid for a region, like pygmt.datasets.load_earth_relief but without GMT.

    resolution  : "15s", "30s", "01m", "02m", "03m", "04m" or "05m"
    region      : [lon_min, lon_max, lat_min, lat_max] in degrees
    registration: None (same rule as pygmt: "gridline" if it exists, else "pixel"),
                  "gridline" or "pixel"
    cache_dir   : folder for the downloaded tiles (default ~/.quickquake/earth_relief);
                  later runs read them from there, without internet

    Returns lon (1-D, degrees east), lat (1-D, degrees north, increasing) and
    z (2-D, metres above sea level, negative = below; shape len(lat) x len(lon)).
    """
    if resolution not in EARTH_RELIEF_TILES:
        raise ValueError(f"resolution must be one of {list(EARTH_RELIEF_TILES)}, not {resolution!r}.")
    tile_deg, available = EARTH_RELIEF_TILES[resolution]
    if registration is None:
        reg = "g" if "g" in available else "p"
    else:
        reg = {"gridline": "g", "pixel": "p"}.get(registration)
        if reg not in available:
            raise ValueError(f"registration for {resolution} must be one of "
                             f"{['gridline' if r == 'g' else 'pixel' for r in available]}.")
    west, east, south, north = [float(v) for v in region]
    if not (-180.0 <= west < east <= 180.0 and -90.0 <= south < north <= 90.0):
        raise ValueError("region must be [lon_min, lon_max, lat_min, lat_max] inside -180..180 / -90..90 "
                         "(regions crossing the 180-degree meridian are not supported).")

    step = int(resolution[:2]) / (3600.0 if resolution.endswith("s") else 60.0)   # degrees
    half = 0.5 if reg == "p" else 0.0         # pixel: values sit half a step inside the cell
    eps = 1e-9                                # absorbs rounding when an edge falls exactly on the grid

    # 1) Which nodes to return, as integer indices on the global grid of this resolution
    #    (lon = -180 + (i + half) * step, lat = -90 + (j + half) * step). Same rule as GMT:
    #    pixel keeps every cell that touches the region; gridline widens the region
    #    outward to the nearest node.
    def index_range(lo, hi, origin):
        first = int(np.floor((lo - origin) / step + eps))
        last = int(np.ceil((hi - origin) / step - eps)) - (1 if reg == "p" else 0)
        return first, last

    i0, i1 = index_range(west, east, -180.0)
    j0, j1 = index_range(south, north, -90.0)
    z = np.full((j1 - j0 + 1, i1 - i0 + 1), np.nan)

    # 2) Read every tile that holds some of those nodes and copy its part into z.
    #    Tiles are named after their south-west corner, e.g. N50W170 = lat 50..60, lon -170..-160.
    nodes_per_tile = int(round(tile_deg / step))
    folder = Path(cache_dir) if cache_dir is not None else EARTH_RELIEF_CACHE
    dataset = f"earth_relief_{resolution}_{reg}"

    def tile_corners(first, last, origin, top):
        lo = origin + (first + half) * step
        hi = origin + (last + half) * step
        start = origin + np.floor((lo - origin) / tile_deg) * tile_deg
        stop = min(origin + np.floor((hi - origin) / tile_deg) * tile_deg, top - tile_deg)
        return np.arange(start, stop + tile_deg / 2, tile_deg)

    for lat0 in tile_corners(j0, j1, -90.0, 90.0):
        for lon0 in tile_corners(i0, i1, -180.0, 180.0):
            name = (f"{'N' if lat0 >= 0 else 'S'}{abs(int(lat0)):02d}"
                    f"{'E' if lon0 >= 0 else 'W'}{abs(int(lon0)):03d}")
            tile = _read_earth_relief_tile(
                f"/server/earth/earth_relief/{dataset}/{name}.{dataset}.jp2",
                folder / dataset / f"{name}.{dataset}.jp2",
            )
            expected = nodes_per_tile + (1 if reg == "g" else 0)    # gridline tiles include both edges
            if tile.shape != (expected, expected):
                raise ValueError(f"Tile {name} has shape {tile.shape}, expected {(expected, expected)}.")
            ti = int(round((lon0 + 180.0) / step))  # global index of the tile's first column / row
            tj = int(round((lat0 + 90.0) / step))
            a, b = max(i0, ti), min(i1, ti + expected - 1)
            c, d = max(j0, tj), min(j1, tj + expected - 1)
            if a <= b and c <= d:
                z[c - j0:d - j0 + 1, a - i0:b - i0 + 1] = tile[c - tj:d - tj + 1, a - ti:b - ti + 1]

    lon = -180.0 + (np.arange(i0, i1 + 1) + half) * step
    lat = -90.0 + (np.arange(j0, j1 + 1) + half) * step
    return lon, lat, z


# ============================================================
# 3D MAP VIEWS
# ============================================================

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
    image_engine: str = "kaleido",   # unused: plotly 7 always uses kaleido (kept so old calls still work)
    region_name: Optional[str] = None,
):
    """
    3D catalog plot (region_name is used in the title, e.g. "Pavlof Volcano"):
      - mode="year_month": filter by year and color by month
      - mode="time": color by time (old -> new)
      - DEM (load_earth_relief: SRTM15+ from the GMT server) + Surface (plotly) + Scatter3d (plotly)
      - interactive if the renderer is set correctly
      - optional: fig.show() and fig.write_image() (kaleido, needs Google Chrome)
    """
    import plotly.graph_objects as go

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    mode = str(mode).lower().strip()
    if mode not in {"time", "year_month"}:
        raise ValueError("mode must be 'time' or 'year_month'.")

    if mode == "year_month":
        if year is None:
            raise ValueError("mode='year_month' needs year=YYYY.")
        df = df[df[catc.time].dt.year == int(year)].copy()
        df["month"] = df[catc.time].dt.month.astype(int)

    if df.empty:
        raise ValueError("No events left after filtering (check year or NaN values).")

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    lon, lat, elev_m = load_earth_relief(dem_res, region=region)   # same grid as pygmt, without GMT
    elev_km = elev_m / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(topo_exagg)

    surface = go.Surface(
        x=Lon,
        y=Lat,
        z=Z,
        colorscale="earth",
        showscale=False,
        opacity=float(topo_opacity),
        name="Topography",
    )

    if catc.mag in df.columns:
        mags = pd.to_numeric(df[catc.mag], errors="coerce")
        mmin, mmax = float(mags.min()), float(mags.max())
        if np.isfinite(mmin) and np.isfinite(mmax) and (mmax > mmin):
            sizes = 1 + (mags - mmin) / (mmax - mmin) * 15
        else:
            sizes = np.full(len(df), 6.0)
    else:
        sizes = np.full(len(df), 6.0)

    if mode == "year_month":
        color_vals = df["month"].to_numpy()
        cmin, cmax = 1, 12
        cbar = dict(
            title=f"Month ({int(year)})",
            tickvals=list(range(1, 13)),
            ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        )
        title = f"{region_name} – {int(year)}" if region_name else f"Seismicity – {int(year)}"
    else:
        tsec = (df[catc.time].astype("int64") / 1e9).to_numpy()
        color_vals = tsec
        cmin, cmax = float(np.min(tsec)), float(np.max(tsec))

        qs = np.linspace(0, 1, 6)
        tickvals = [float(np.quantile(tsec, q)) for q in qs]
        ticktext = [
            pd.to_datetime(v, unit="s", utc=True).strftime("%Y-%m-%d<br>%H:%M")
            for v in tickvals
        ]
        cbar = dict(title="Time (old → new)", tickvals=tickvals, ticktext=ticktext)
        title = f"{region_name} – time colored" if region_name else "Seismicity – time colored"

    if catc.depth_km in df.columns:
        z_depth = pd.to_numeric(df[catc.depth_km], errors="coerce").fillna(0.0).to_numpy()
    else:
        z_depth = np.zeros(len(df))

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

    if show:
        if renderer is None:
            fig.show()
        else:
            fig.show(renderer=renderer)

    if save_image is not None:
        fig.write_image(
            str(save_image),
            format=image_format,
            width=int(image_width),
            height=int(image_height),
        )

    return fig


def plot_mapview3d_months(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    depth_max: float = 40.0,
    dem_res: str = "15s",
    exagg: float = 4.0,
    topo_cmap: str = "gist_earth_r",
    quake_cmap: str = "magma",
    size_k: float = 15.0,
    topo_alpha: float = 0.95,
    topo_stride: int = 5,
    view_elev: float = 4.0,
    view_azim: float = 34.0,
    camera_dist: Optional[float] = None,   # <-- nuevo
    n_xticks: int = 5,
    tick_fontsize: int = 13,
    cbar_tick_fontsize: int = 18,
    figsize: Tuple[float, float] = (12, 9),
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 800,
    catc: Optional[CatalogCols] = None,
):
    """
    3D figure:
      - PyGMT topography (earth_relief) -> 3D surface
      - Seismicity: x=lon, y=lat, z=depth (inverted)
      - Color: month (1-12)
      - Size: size_k * mag
      - Filters: [start, end) and depth <= depth_max

    Parameter:
      camera_dist : float | None
          Camera distance in Matplotlib 3D.
          Smaller value = more visual zoom.
          Examples:
              6  -> moderate zoom
              3  -> strong zoom

    Note:
      - This does NOT change the data or the region.
      - It only changes the viewing perspective.
      - Depends on Matplotlib and may not behave perfectly.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if catc is None:
        catc = CatalogCols()

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    t = pd.to_datetime(df[catc.time], utc=True, errors="coerce")
    df = df[pd.notna(t)].copy()
    df["_t"] = t

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    df = df[(df["_t"] >= start_dt) & (df["_t"] < end_dt)].copy()

    depth_col = catc.depth_km if catc.depth_km in df.columns else None
    if depth_col is None:
        df["depth_km"] = 0.0
        depth_col = "depth_km"
    else:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0.0)

    df = df[df[depth_col] <= float(depth_max)].copy()

    mag_col = catc.mag if catc.mag in df.columns else None
    if mag_col is None:
        df["magnitude"] = 1.0
        mag_col = "magnitude"
    else:
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").fillna(1.0)

    df["month"] = df["_t"].dt.month.astype(int)

    print(f"Number of events after filters (time, depth<={depth_max} km): {len(df)}")

    if df.empty:
        raise ValueError("No events left after filtering (check start/end/depth_max).")

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    lon, lat, elev_m = load_earth_relief(dem_res, region=region)   # same grid as pygmt, without GMT
    elev_km = elev_m / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(exagg)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.10, right=0.90)

    ax.plot_surface(
        Lon, Lat, Z,
        cmap=topo_cmap,
        rstride=int(topo_stride),
        cstride=int(topo_stride),
        linewidth=0,
        antialiased=True,
        alpha=float(topo_alpha),
    )

    sc = ax.scatter(
        df[catc.lon].astype(float),
        df[catc.lat].astype(float),
        df[depth_col].astype(float),
        c=df["month"].to_numpy(),
        cmap=quake_cmap,
        norm=Normalize(vmin=1, vmax=12),
        s=float(size_k) * df[mag_col].to_numpy(),   # marker area (pt^2) = size_k * magnitude, as before v4.7
        edgecolor="k",
        linewidth=0.2,
        alpha=1.0,
        label="Earthquakes",
    )

    ax.tick_params(axis="x", labelsize=int(tick_fontsize), labelrotation=0)
    ax.tick_params(axis="y", labelsize=int(tick_fontsize) + 1, labelrotation=0)
    ax.tick_params(axis="z", labelsize=int(tick_fontsize) + 1, labelrotation=0)

    xticks = np.linspace(minlon, maxlon, int(n_xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])

    ax.set_zlim(float(np.nanmin(Z)), float(depth_max))
    ax.invert_zaxis()

    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=1e-14,
        fraction=0.02,
        shrink=0.4,
        aspect=30,
    )
    cbar.ax.tick_params(labelsize=int(cbar_tick_fontsize))
    months = np.arange(1, 13)
    cbar.set_ticks(months)
    cbar.set_ticklabels([
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ])

    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    # Visual zoom through camera distance
    if camera_dist is not None:
        try:
            ax.dist = float(camera_dist)
        except Exception as e:
            print(f"[warn] No se pudo aplicar camera_dist={camera_dist}: {e}")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.99)
        axis._axinfo["grid"]["linewidth"] = 0.20
        axis._axinfo["grid"]["linestyle"] = "--"

    if savefig:
        if out_png is None:
            out_png = "mapview3d_months.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    meta: Dict[str, Any] = {
        "n_events": int(len(df)),
        "region": region,
        "bounds": (minlon, maxlon, minlat, maxlat),
        "dem_res": dem_res,
        "depth_max": float(depth_max),
        "camera_dist": camera_dist,
    }

    return fig, ax, meta


def plot_mapview3d_timecolor(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    depth_max: float = 40.0,
    dem_res: str = "15s",
    exagg: float = 4.0,
    topo_cmap: str = "gist_earth_r",
    quake_cmap: str = "magma",
    size_k: float = 15.0,
    topo_alpha: float = 0.95,
    topo_stride: int = 5,
    view_elev: float = 4.0,
    view_azim: float = 34.0,
    n_xticks: int = 5,
    n_yticks: int = 5,
    tick_fontsize: int = 13,
    cbar_tick_fontsize: int = 18,
    figsize: Tuple[float, float] = (12, 9),
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 800,
    catc: Optional["CatalogCols"] = None,
):
    """
    3D figure:
      - PyGMT topography (earth_relief) -> 3D surface
      - Seismicity: x=lon, y=lat, z=depth (inverted)
      - Color: continuous time
      - Size: size_k * mag
      - Filters: [start, end) and depth <= depth_max
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    if catc is None:
        catc = CatalogCols()

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    t = pd.to_datetime(df[catc.time], utc=True, errors="coerce")
    df = df[pd.notna(t)].copy()
    df["_t"] = t

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    df = df[(df["_t"] >= start_dt) & (df["_t"] < end_dt)].copy()

    depth_col = catc.depth_km if catc.depth_km in df.columns else None
    if depth_col is None:
        df["depth_km"] = 0.0
        depth_col = "depth_km"
    else:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0.0)

    df = df[df[depth_col] <= float(depth_max)].copy()

    mag_col = catc.mag if catc.mag in df.columns else None
    if mag_col is None:
        df["magnitude"] = 1.0
        mag_col = "magnitude"
    else:
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").fillna(1.0)

    print(f"Number of events after filters (time, depth<={depth_max} km): {len(df)}")

    if df.empty:
        raise ValueError("No events left after filtering (check start/end/depth_max).")

    # Continuous time for coloring
    df["_t_num"] = mdates.date2num(df["_t"].dt.to_pydatetime())

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    lon, lat, elev_m = load_earth_relief(dem_res, region=region)   # same grid as pygmt, without GMT
    elev_km = elev_m / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(exagg)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.10, right=0.90)

    ax.plot_surface(
        Lon, Lat, Z,
        cmap=topo_cmap,
        rstride=int(topo_stride),
        cstride=int(topo_stride),
        linewidth=0,
        antialiased=True,
        alpha=float(topo_alpha),
    )

    sc = ax.scatter(
        df[catc.lon].astype(float),
        df[catc.lat].astype(float),
        df[depth_col].astype(float),
        c=df["_t_num"].to_numpy(),
        cmap=quake_cmap,
        s=float(size_k) * df[mag_col].to_numpy(),   # marker area (pt^2) = size_k * magnitude, as before v4.7
        edgecolor="k",
        linewidth=0.2,
        alpha=1.0,
        label="Earthquakes",
    )

    ax.tick_params(axis="x", labelsize=int(tick_fontsize), labelrotation=0)
    ax.tick_params(axis="y", labelsize=int(tick_fontsize) + 1, labelrotation=0)
    ax.tick_params(axis="z", labelsize=int(tick_fontsize) + 1, labelrotation=0)

    xticks = np.linspace(minlon, maxlon, int(n_xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])

    yticks = np.linspace(minlat, maxlat, int(n_yticks))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])

    ax.set_zlim(float(np.nanmin(Z)), float(depth_max))
    ax.invert_zaxis()

    cbar = fig.colorbar(
        sc,
        ax=ax,
        pad=1e-14,
        fraction=0.02,
        shrink=0.4,
        aspect=30,
    )
    cbar.ax.tick_params(labelsize=int(cbar_tick_fontsize))
    cbar.set_label("Time", fontsize=int(cbar_tick_fontsize))

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    cbar.ax.yaxis.set_major_locator(locator)
    cbar.ax.yaxis.set_major_formatter(formatter)

    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.99)
        axis._axinfo["grid"]["linewidth"] = 0.20
        axis._axinfo["grid"]["linestyle"] = "--"

    if savefig:
        if out_png is None:
            out_png = "mapview3d_timecolor.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    meta: Dict[str, Any] = {
        "n_events": int(len(df)),
        "region": region,
        "bounds": (minlon, maxlon, minlat, maxlat),
        "dem_res": dem_res,
        "depth_max": float(depth_max),
        "time_min": str(df["_t"].min()),
        "time_max": str(df["_t"].max()),
    }

    return fig, ax, meta


def plot_mapview3d_groups(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    depth_max: float = 40.0,
    dem_res: str = "15s",
    exagg: float = 4.0,
    topo_cmap: str = "gist_earth_r",
    size_k: float = 15.0,
    topo_alpha: float = 0.95,
    topo_stride: int = 5,
    view_elev: float = 4.0,
    view_azim: float = 34.0,
    n_xticks: int = 5,
    n_yticks: int = 5,
    tick_fontsize: int = 13,
    figsize: Tuple[float, float] = (12, 9),
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 800,
    catc: Optional["CatalogCols"] = None,
    hollow_quakes: bool = True,
    quake_edge_lw: float = 0.7,
    group1_xyz: Optional[np.ndarray] = None,
    group2_xyz: Optional[np.ndarray] = None,
    group1_edgecolor: str = "darkred",
    group2_edgecolor: str = "navy",
    other_edgecolor: str = "black",
    plane_polys: Optional[list] = None,
):
    """
    Plot 3D topography, seismicity, and optional planes.

    Earthquake edge colors:
      - group1_xyz uses group1_edgecolor
      - group2_xyz uses group2_edgecolor
      - all remaining events use other_edgecolor

    Earthquake markers can optionally be hollow.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if catc is None:
        catc = CatalogCols()

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    t = pd.to_datetime(df[catc.time], utc=True, errors="coerce")
    df = df[pd.notna(t)].copy()
    df["_t"] = t

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)

    df = df[(df["_t"] >= start_dt) & (df["_t"] < end_dt)].copy()

    depth_col = catc.depth_km if catc.depth_km in df.columns else None
    if depth_col is None:
        df["depth_km"] = 0.0
        depth_col = "depth_km"
    else:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0.0)

    df = df[df[depth_col] <= float(depth_max)].copy()

    mag_col = catc.mag if catc.mag in df.columns else None
    if mag_col is None:
        df["magnitude"] = 1.0
        mag_col = "magnitude"
    else:
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").fillna(1.0)

    print(f"Number of events after filters: {len(df)}")

    if df.empty:
        raise ValueError("No events left after filtering.")

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    lon, lat, elev_m = load_earth_relief(dem_res, region=region)   # same grid as pygmt, without GMT
    elev_km = elev_m / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(exagg)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.10, right=0.90)

    ax.plot_surface(
        Lon,
        Lat,
        Z,
        cmap=topo_cmap,
        rstride=int(topo_stride),
        cstride=int(topo_stride),
        linewidth=0,
        antialiased=True,
        alpha=float(topo_alpha),
    )

    # ---------------------------------------------------------
    # preparar grupos
    # ---------------------------------------------------------
    pts_all = df[[catc.lon, catc.lat, depth_col]].to_numpy(dtype=float)
    mags = df[mag_col].to_numpy(dtype=float)
    sizes = float(size_k) * mags
    sizes = np.where(np.isfinite(sizes), sizes, 1.0)
    sizes = np.clip(sizes, 1.0, None)

    mask_g1 = np.zeros(len(df), dtype=bool)
    mask_g2 = np.zeros(len(df), dtype=bool)

    def rows_to_set(arr):
        if arr is None or len(arr) == 0:
            return set()
        arr = np.asarray(arr, dtype=float)
        arr = np.round(arr, 6)
        return set(map(tuple, arr))

    set_g1 = rows_to_set(group1_xyz)
    set_g2 = rows_to_set(group2_xyz)

    pts_round = np.round(pts_all, 6)

    for i, row in enumerate(pts_round):
        tup = tuple(row)
        if tup in set_g1:
            mask_g1[i] = True
        elif tup in set_g2:
            mask_g2[i] = True

    mask_other = ~(mask_g1 | mask_g2)

    # Scatter styles
    filled_kwargs = dict(
        linewidths=float(quake_edge_lw),
        alpha=1.0,
        depthshade=False,
    )

    hollow_kwargs = dict(
        linewidths=float(quake_edge_lw),
        alpha=1.0,
        depthshade=False,
        facecolors="none",
    )

    # Plot group 1 with filled markers.
    # ---------------------------------------------------------
    if mask_g1.any():
        ax.scatter(
            df.loc[mask_g1, catc.lon].astype(float),
            df.loc[mask_g1, catc.lat].astype(float),
            df.loc[mask_g1, depth_col].astype(float),
            s=sizes[mask_g1],
            facecolors=group1_edgecolor,
            edgecolors=group1_edgecolor,
            **filled_kwargs,
        )

    # ---------------------------------------------------------
    # Plot group 2 with filled markers.
    # ---------------------------------------------------------
    if mask_g2.any():
        ax.scatter(
            df.loc[mask_g2, catc.lon].astype(float),
            df.loc[mask_g2, catc.lat].astype(float),
            df.loc[mask_g2, depth_col].astype(float),
            s=sizes[mask_g2],
            facecolors=group2_edgecolor,
            edgecolors=group2_edgecolor,
            **filled_kwargs,
        )

    # ---------------------------------------------------------
    # Plot remaining events with hollow markers.
    # ---------------------------------------------------------
    if mask_other.any():
        remaining_style = hollow_kwargs.copy() if hollow_quakes else filled_kwargs.copy()
        remaining_style["facecolors"] = "none" if hollow_quakes else other_edgecolor
        ax.scatter(
            df.loc[mask_other, catc.lon].astype(float),
            df.loc[mask_other, catc.lat].astype(float),
            df.loc[mask_other, depth_col].astype(float),
            s=sizes[mask_other],
            edgecolors=other_edgecolor,
            **remaining_style,
        )

    # Optional plane polygons
    # plane_polys = [
    #   {"xyz": list(zip(plane1_lon, plane1_lat, plane1_z)),
    #    "facecolor": "red", "edgecolor": "darkred", "alpha": 0.30, "lw": 1.0},
    #   {"xyz": list(zip(plane2_lon, plane2_lat, plane2_z)),
    #    "facecolor": "blue", "edgecolor": "navy", "alpha": 0.30, "lw": 1.0},
    # ]
    if plane_polys is not None:
        for poly in plane_polys:
            xyz = np.asarray(poly["xyz"], dtype=float)

            ax.add_collection3d(
                Poly3DCollection(
                    [xyz],
                    facecolors=poly.get("facecolor", "red"),
                    edgecolors=poly.get("edgecolor", "k"),
                    alpha=float(poly.get("alpha", 0.30)),
                    linewidths=float(poly.get("lw", 1.0)),
                )
            )

            ax.plot(
                np.r_[xyz[:, 0], xyz[0, 0]],
                np.r_[xyz[:, 1], xyz[0, 1]],
                np.r_[xyz[:, 2], xyz[0, 2]],
                color=poly.get("edgecolor", "k"),
                lw=float(poly.get("lw", 1.0)) + 0.5,
            )

    # Axis configuration and styling
    ax.tick_params(axis="x", labelsize=int(tick_fontsize), labelrotation=0)
    ax.tick_params(axis="y", labelsize=int(tick_fontsize) + 1, labelrotation=0)
    ax.tick_params(axis="z", labelsize=int(tick_fontsize) + 1, labelrotation=0)

    xticks = np.linspace(minlon, maxlon, int(n_xticks))
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{x:.1f}" for x in xticks])

    yticks = np.linspace(minlat, maxlat, int(n_yticks))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.1f}" for y in yticks])

    ax.set_zlim(float(np.nanmin(Z)), float(depth_max))
    ax.invert_zaxis()
    ax.view_init(elev=float(view_elev), azim=float(view_azim))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.5, 0.5, 0.5, 0.99)
        axis._axinfo["grid"]["linewidth"] = 0.20
        axis._axinfo["grid"]["linestyle"] = "--"

    if savefig:
        if out_png is None:
            out_png = "mapview3d_groups.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    meta = {
        "n_events": int(len(df)),
        "n_group1": int(mask_g1.sum()),
        "n_group2": int(mask_g2.sum()),
        "n_other": int(mask_other.sum()),
        "region": region,
        "bounds": (minlon, maxlon, minlat, maxlat),
        "dem_res": dem_res,
        "depth_max": float(depth_max),
        "time_min": str(df["_t"].min()),
        "time_max": str(df["_t"].max()),
    }

    return fig, ax, meta


# ============================================================
# PROFILE AND CROSS-SECTION TOOLS
# ============================================================

def project_profile_distances(
    lon,
    lat,
    *,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
):
    """
    Project (lon, lat) points onto a geodesic profile start -> end.

    Returns
    -------
    along_km : np.ndarray
        Projected distance along the profile [km], 0 at the start point.
    across_km : np.ndarray
        Signed perpendicular distance from the profile [km].
        The sign depends on the azimuth geometry.
    profile_length_km : float
        Total profile length [km].
    az_line_deg : float
        Azimut forward del perfil [deg].
    """
    from pyproj import Geod

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    geod = Geod(ellps="WGS84")
    az_line_deg, _, dist_line_m = geod.inv(
        float(start_lon), float(start_lat), float(end_lon), float(end_lat)
    )
    profile_length_km = float(dist_line_m) / 1000.0

    az_eq_deg, _, dist_m = geod.inv(
        np.full(lon.shape, float(start_lon)),
        np.full(lat.shape, float(start_lat)),
        lon,
        lat,
    )

    delta_rad = np.deg2rad(az_eq_deg - az_line_deg)
    dist_km = dist_m / 1000.0

    along_km = dist_km * np.cos(delta_rad)
    across_km = dist_km * np.sin(delta_rad)

    return along_km, across_km, profile_length_km, float(az_line_deg)


def add_profile_line(
    ax,
    *,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    start_label: str = "A",
    end_label: str = "A'",
    line_color: str = "cyan",
    line_width: float = 2.8,
    text_offset_deg: float = 0.01,
    fontsize: int = 13,
    marker_size: float = 42,
    label_offset_start: Optional[Tuple[float, float]] = None,
    label_offset_end: Optional[Tuple[float, float]] = None,
    zorder: float = 200,
):
    """
    Draw a visible profile line on a Cartopy/Matplotlib map.

    Labels: by default each label sits text_offset_deg (degrees of longitude)
    outside its end point. label_offset_start / label_offset_end = (dx, dy) in
    typographic points instead place that label at a fixed screen offset from
    the point (as in research paper2/qq_helpers.py), which stays the same at any
    map zoom. zorder sets the drawing layer: line = zorder, end points =
    zorder + 1, labels = zorder + 2.
    """
    import cartopy.crs as ccrs
    import matplotlib.patheffects as pe

    pe_line = [pe.Stroke(linewidth=float(line_width) + 2.2, foreground="black"), pe.Normal()]
    pe_text = [pe.Stroke(linewidth=2.5, foreground="black"), pe.Normal()]

    # line
    line, = ax.plot(
        [float(start_lon), float(end_lon)],
        [float(start_lat), float(end_lat)],
        color=line_color,
        linewidth=float(line_width),
        transform=ccrs.PlateCarree(),
        zorder=float(zorder),
        solid_capstyle="round",
    )
    line.set_path_effects(pe_line)

    # extremos
    sc1 = ax.scatter(
        [float(start_lon)],
        [float(start_lat)],
        s=float(marker_size),
        marker="o",
        facecolor=line_color,
        edgecolor="black",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=float(zorder) + 1,
    )
    sc2 = ax.scatter(
        [float(end_lon)],
        [float(end_lat)],
        s=float(marker_size),
        marker="o",
        facecolor=line_color,
        edgecolor="black",
        linewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=float(zorder) + 1,
    )

    # etiquetas
    text_style = dict(fontsize=int(fontsize), weight="bold", color="white", zorder=float(zorder) + 2)

    def place_label(lon, lat, label, offset_pts, side):
        # side = -1 puts the label west of the point (start), +1 east (end)
        if offset_pts is None:
            return ax.text(
                float(lon) + side * float(text_offset_deg),
                float(lat),
                str(label),
                transform=ccrs.PlateCarree(),
                ha="right" if side < 0 else "left",
                va="center",
                **text_style,
            )
        return ax.annotate(
            str(label),
            xy=(float(lon), float(lat)),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=tuple(offset_pts),
            textcoords="offset points",
            ha="left",
            va="bottom",
            **text_style,
        )

    t1 = place_label(start_lon, start_lat, start_label, label_offset_start, side=-1)
    t2 = place_label(end_lon, end_lat, end_label, label_offset_end, side=+1)
    t1.set_path_effects(pe_text)
    t2.set_path_effects(pe_text)

    return ax


def plot_cross_section_seismicity(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    start_lon: float,
    start_lat: float,
    end_lon: float,
    end_lat: float,
    catc: Optional[CatalogCols] = None,
    swath_km: Optional[float] = None,
    depth_vmin: float = 0.0,
    depth_vmax: float = 40.0,
    depth_max_plot: Optional[float] = None,
    cmap_name: str = "gist_rainbow",
    size_k: float = 22.0,
    marker_min: float = 6.0,
    figsize: Tuple[float, float] = (12, 6),
    title: Optional[str] = None,
    x_label_start: str = "A",
    x_label_end: str = "A'",
    show_end_line: bool = False,
    grid: bool = True,
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 600,
):
    """
    Cross-section of seismicity along a geographic profile.

    depth_max_plot:
        If given, removes events with depth > depth_max_plot and
        limits the Y axis to [0, depth_max_plot].
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    catc = catc or CatalogCols()
    quakes = make_quakes_for_map(catalog, start, end, catc=catc).copy()

    if quakes.empty:
        raise ValueError("No events in the range [start, end].")

    along_km, across_km, profile_length_km, az_line_deg = project_profile_distances(
        quakes["lon"].to_numpy(),
        quakes["lat"].to_numpy(),
        start_lon=float(start_lon),
        start_lat=float(start_lat),
        end_lon=float(end_lon),
        end_lat=float(end_lat),
    )

    quakes["distance_km"] = along_km
    quakes["across_km"] = across_km

    if swath_km is not None:
        swath_km = float(swath_km)
        quakes = quakes[np.abs(quakes["across_km"]) <= swath_km].copy()

    quakes = quakes[np.isfinite(quakes["distance_km"])].copy()
    quakes = quakes[np.isfinite(quakes["depth"])].copy()

    if depth_max_plot is not None:
        depth_max_plot = float(depth_max_plot)
        quakes = quakes[quakes["depth"] <= depth_max_plot].copy()

    # keep only projections that fall within the profile length
    quakes = quakes[
        (quakes["distance_km"] >= 0.0) &
        (quakes["distance_km"] <= float(profile_length_km))
    ].copy()

    if quakes.empty:
        raise ValueError("No events left after the profile/swath/depth filters.")

    sizes = float(size_k) * quakes["mag"].to_numpy(dtype=float)
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    cmap = plt.get_cmap(str(cmap_name), 256)
    norm = mpl.colors.Normalize(float(depth_vmin), float(depth_vmax))

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    sc = ax.scatter(
        quakes["distance_km"],
        quakes["depth"],
        s=sizes,
        c=quakes["depth"],
        cmap=cmap,
        norm=norm,
        edgecolor="k",
        linewidth=0.25,
        alpha=0.85,
        zorder=10,
    )

    ax.set_xlabel(f"Distance along profile {x_label_start}–{x_label_end} (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_xlim(0.0, float(profile_length_km))

    if depth_max_plot is not None:
        ax.set_ylim(float(depth_max_plot), 0.0)
    else:
        ax.invert_yaxis()

    if grid:
        ax.grid(alpha=0.3, linestyle="--", zorder=0)

    if show_end_line:
        ax.axvline(float(profile_length_km), color="red", linestyle="--", alpha=0.7, zorder=5)

    if title is None:
        start_dt = pd.to_datetime(start, utc=True)
        end_dt = pd.to_datetime(end, utc=True)
        title = (
            f"Seismic profile: {start_dt.date()} to {end_dt.date()}\n"
            f"{x_label_start} ({start_lon:.4f}, {start_lat:.4f}) to "
            f"{x_label_end} ({end_lon:.4f}, {end_lat:.4f}) | "
            f"L = {profile_length_km:.1f} km"
        )
        if swath_km is not None:
            title += f" | swath = ±{swath_km:.1f} km"
        if depth_max_plot is not None:
            title += f" | depth ≤ {depth_max_plot:.1f} km"

    ax.set_title(title, pad=14)

    cbar = plt.colorbar(sc, ax=ax, label="Depth (km)", pad=0.02)
    cbar.set_alpha(1)

    if savefig:
        if out_png is None:
            out_png = "seismic_cross_section.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight", facecolor="white")
        print(f"[ok] saved: {out_png}")

    meta = {
        "profile_length_km": float(profile_length_km),
        "azimuth_deg": float(az_line_deg),
        "n_events": int(len(quakes)),
        "swath_km": None if swath_km is None else float(swath_km),
        "depth_max_plot": depth_max_plot,
        "start": (float(start_lon), float(start_lat)),
        "end": (float(end_lon), float(end_lat)),
        "distance_col": "distance_km",
        "across_col": "across_km",
        "projected_catalog": quakes.copy(),
    }
    return fig, ax, meta


def plot_plane_orientation_rose(
    dips,
    dipdirs,
    labels=None,
    colors=None,
    rmax=90,
    title="Plane orientations: dip direction / dip",
    savepath=None,
    dpi=300,
):
    """
    Polar plot:
      - angle = dip direction (deg clockwise from North)
      - radius = dip (deg)
    """

    import numpy as np
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [f"P{i+1}" for i in range(len(dips))]
    if colors is None:
        colors = ["darkred", "navy", "darkgreen", "purple"]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw={"projection": "polar"})

    # North at top, clockwise angles
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # Radial axis = dip
    ax.set_rlim(0, rmax)
    ax.set_rticks([15, 30, 45, 60, 75, 90])
    ax.set_rlabel_position(225)

    # Angular ticks every 30 deg
    theta_deg = np.arange(0, 360, 30)
    ax.set_thetagrids(theta_deg, labels=[f"{t}°" for t in theta_deg])

    # Grid appearance
    ax.grid(True, linestyle="--", alpha=0.55)
    ax.set_title(title, va="bottom", fontsize=13)

    # Cardinal labels
    for ang_deg, txt in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        ax.text(
            np.deg2rad(ang_deg),
            rmax + 6,
            txt,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )

    # Plot each plane
    for i, (dip, dipdir) in enumerate(zip(dips, dipdirs)):
        theta = np.deg2rad(dipdir)
        color = colors[i % len(colors)]
        label = labels[i]

        # radial guide line all the way to outer circle
        ax.plot([theta, theta], [0, rmax], linestyle=":", linewidth=1.8, color=color, alpha=0.8)

        # arrow from center to dip radius
        ax.annotate(
            "",
            xy=(theta, dip),
            xytext=(theta, 0),
            arrowprops=dict(arrowstyle="->", lw=3, color=color),
        )

        # point at the tip
        ax.plot(theta, dip, "o", color=color, markersize=9)

        # label
        ax.text(
            theta,
            min(dip + 7, rmax - 2),
            f"{label}\n{dipdir:.1f}° / {dip:.1f}°",
            color=color,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
        )

    plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight", transparent=True)

    return fig, ax


# ============================================================
# MAGNITUDE OF COMPLETENESS
# ============================================================

def build_fmd(magnitudes, bin_width=0.1):
    """
    Frequency-magnitude distribution (FMD) of a set of magnitudes.

    Bins are bin_width wide (magnitude units) and centred on multiples of
    bin_width. Returns (centers, edges, counts, cumulative_counts): counts[i]
    is the number of events in bin i and cumulative_counts[i] is N(M >= centers[i]).
    NaN magnitudes are ignored.
    """
    mags = np.asarray(magnitudes, dtype=float)
    mags = mags[np.isfinite(mags)]
    if mags.size == 0:
        raise ValueError("No valid magnitudes.")

    m_min = np.floor(mags.min() / bin_width) * bin_width
    m_max = np.ceil(mags.max() / bin_width) * bin_width

    centers = np.arange(m_min, m_max + bin_width, bin_width)

    edges = np.concatenate((
        [centers[0] - bin_width / 2],
        centers[:-1] + bin_width / 2,
        [centers[-1] + bin_width / 2]
    ))

    counts, _ = np.histogram(mags, bins=edges)
    cumulative_counts = counts[::-1].cumsum()[::-1]   # N(M >= m)

    return centers, edges, counts, cumulative_counts


def estimate_mc_maxc(centers, counts):
    """
    Magnitude of completeness by maximum curvature (MAXC): the centre of the
    FMD bin with the most events. MAXC tends to underestimate Mc when the FMD
    is gradually curved; adding +0.2 is a common correction (Woessner & Wiemer 2005).
    """
    return centers[np.argmax(counts)]


def fit_gutenberg_richter_line(centers, cumulative_counts, mc, min_cum_count=10):
    """
    Least-squares line log10 N(>=M) = intercept + slope * M through the FMD bins
    with M >= mc and at least min_cum_count events (drops the noisy large-M tail).

    slope is about -b. A least-squares b-value is only indicative; the
    maximum-likelihood estimate (Aki 1965) is the usual one to report.
    Returns (slope, intercept), or (nan, nan) when fewer than 2 bins qualify.
    """
    centers = np.asarray(centers, dtype=float)
    cumulative_counts = np.asarray(cumulative_counts)

    fit_mask = (centers >= mc) & (cumulative_counts >= min_cum_count)
    if fit_mask.sum() < 2:
        return np.nan, np.nan

    slope, intercept = np.polyfit(centers[fit_mask], np.log10(cumulative_counts[fit_mask]), 1)
    return slope, intercept


def _as_catalog_dict(catalogs, default_label="Catalog"):
    """One catalog, or a dict {label: catalog} to overlay several."""
    if isinstance(catalogs, dict):
        return dict(catalogs)
    return {default_label: catalogs}


def _column_for(col, label):
    """A column name given once for all catalogs, or per catalog as {label: name}."""
    return col.get(label) if isinstance(col, dict) else col


def _magnitudes_of(catalog, mag_col):
    """Finite magnitudes from a DataFrame column, or from an array of magnitudes."""
    if isinstance(catalog, pd.DataFrame):
        if mag_col not in catalog.columns:
            raise KeyError(f"Magnitude column '{mag_col}' not found in catalog.")
        return pd.to_numeric(catalog[mag_col], errors="coerce").dropna().to_numpy()
    mags = np.asarray(catalog, dtype=float)
    return mags[np.isfinite(mags)]


_MC_COLORS = ["black", "tab:blue", "tab:orange", "tab:green", "tab:purple"]


def plot_fmd_cumulative(
    catalogs,
    mag_col="magnitude",
    bin_width=0.1,
    min_cum_count_for_fit=10,
    show_fit=True,
    inset_histogram=True,
    colors=None,
    figsize=(10, 6),
    savefig=False,
    out_png=None,
    dpi=300,
    show=False,
):
    """
    Cumulative frequency-magnitude curve N(M >= m) on a log scale, with the MAXC
    magnitude of completeness and the Gutenberg-Richter least-squares line.

    catalogs: one catalog (DataFrame with `mag_col`, or an array of magnitudes),
    or {label: catalog} to overlay several, e.g. {"QuickQuake": qq, "USGS": usgs}.
    mag_col can be a dict {label: column}. Each catalog gets its own Mc and fit,
    drawn in its colour. With one catalog the figure is the paper 2 one (black
    squares, red fit and Mc line, FMD histogram inset).
    Overlaid catalogs should cover the same area and time window, and their
    magnitudes may be on different scales (e.g. ML vs QuickQuake magnitude).

    Returns fig, ax, results; results[label] holds n_events, mc, slope,
    intercept, centers, counts and cumulative_counts.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    cats = _as_catalog_dict(catalogs)
    single = len(cats) == 1
    colors = colors or {}

    fig, ax = plt.subplots(figsize=figsize)
    results = {}

    for k, (label, cat) in enumerate(cats.items()):
        mags = _magnitudes_of(cat, _column_for(mag_col, label))
        centers, edges, counts, cumulative_counts = build_fmd(mags, bin_width=bin_width)
        mc = estimate_mc_maxc(centers, counts)
        slope, intercept = fit_gutenberg_richter_line(centers, cumulative_counts, mc, min_cum_count_for_fit)
        y_fit_line = 10**(intercept + slope * centers)
        results[label] = dict(n_events=int(mags.size), mc=float(mc), slope=float(slope),
                              intercept=float(intercept), centers=centers, counts=counts,
                              cumulative_counts=cumulative_counts)

        if single:
            ax.semilogy(centers, cumulative_counts, "ks", ms=5, label="Cumulative count")
            if show_fit and np.isfinite(slope):
                ax.semilogy(centers, y_fit_line, "r-", lw=2, label="Linear fit")
            ax.axvline(mc, color="red", linestyle="--", lw=2, label=f"Mc = {mc:.2f}")
        else:
            c = colors.get(label, _MC_COLORS[k % len(_MC_COLORS)])
            ax.semilogy(centers, cumulative_counts, "s", color=c, ms=5, label=f"{label} (N = {mags.size})")
            if show_fit and np.isfinite(slope):
                ax.semilogy(centers, y_fit_line, "-", color=c, lw=2, label=f"{label} fit (slope {slope:.2f})")
            ax.axvline(mc, color=c, linestyle="--", lw=2, label=f"{label} Mc = {mc:.2f}")

    ax.set_xlabel("Magnitude", fontsize=16)
    ax.set_ylabel("Cumulative count", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12)

    if inset_histogram:
        axins = inset_axes(ax, width="35%", height="35%", loc="lower left", borderpad=4)
        for k, (label, r) in enumerate(results.items()):
            if single:
                axins.bar(r["centers"], r["counts"], width=bin_width * 0.9)
                axins.axvline(r["mc"], color="red", linestyle="--", lw=1.5)
            else:
                c = colors.get(label, _MC_COLORS[k % len(_MC_COLORS)])
                axins.step(r["centers"], r["counts"], where="mid", color=c, lw=1.2)
                axins.axvline(r["mc"], color=c, linestyle="--", lw=1.5)
        axins.set_xlabel("M", fontsize=11)
        axins.set_ylabel("N", fontsize=11)
        axins.tick_params(axis="both", labelsize=9)

    plt.tight_layout()

    if savefig and out_png:
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")
    if show:
        plt.show()

    return fig, ax, results


def plot_magnitude_time(
    catalogs,
    mc=None,
    mag_col="magnitude",
    time_col="time",
    bin_width=0.1,
    colors=None,
    title="Magnitudes through time",
    figsize=(11, 5),
    savefig=False,
    out_png=None,
    dpi=300,
    show=False,
):
    """
    Magnitude of every event against its origin time, with a horizontal line at Mc.

    catalogs: one DataFrame or {label: DataFrame} to overlay several.
    mc: a number, {label: number}, or None to compute the MAXC Mc of each
    catalog (bins of bin_width). mag_col / time_col can be dicts {label: column}.
    With one catalog the figure is the paper 2 one (open black circles, red Mc line).

    Returns fig, ax, {label: mc}.
    """
    import matplotlib.pyplot as plt

    cats = _as_catalog_dict(catalogs)
    single = len(cats) == 1
    colors = colors or {}

    fig, ax = plt.subplots(figsize=figsize)
    mcs = {}

    for k, (label, df) in enumerate(cats.items()):
        mcol, tcol = _column_for(mag_col, label), _column_for(time_col, label)
        if tcol not in df.columns:
            raise KeyError(f"Time column '{tcol}' not found in catalog '{label}'.")
        # format="mixed" (as in normalize_catalog): catalogs mix "...:08.61+00:00" and "...:46+00:00";
        # without it pandas >= 2 silently turns the second kind into NaT and drops those events
        times = pd.to_datetime(df[tcol], utc=True, errors="coerce", format="mixed")
        mags = pd.to_numeric(df[mcol], errors="coerce")
        valid = times.notna() & mags.notna()

        if mc is None:
            centers, _, counts, _ = build_fmd(_magnitudes_of(df, mcol), bin_width=bin_width)
            mc_k = estimate_mc_maxc(centers, counts)
        else:
            mc_k = _column_for(mc, label)
        mcs[label] = float(mc_k)

        c = "black" if single else colors.get(label, _MC_COLORS[k % len(_MC_COLORS)])
        ax.scatter(times[valid], mags[valid], s=18, facecolors="none", edgecolors=c, linewidths=0.8,
                   label=None if single else f"{label} (N = {int(valid.sum())})")
        ax.axhline(mc_k, color="red" if single else c, linestyle="-", linewidth=2,
                   label=f"Mc = {mc_k:.2f}" if single else f"{label} Mc = {mc_k:.2f}")

    ax.set_xlabel("Time")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if savefig and out_png:
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")
    if show:
        plt.show()

    return fig, ax, mcs


# ============================================================
# LOCATION CHANGES BETWEEN PIPELINE STAGES (GaMMA -> HypoInverse -> HypoDD)
# ============================================================

def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km on a sphere of radius 6371 km (inputs in degrees)."""
    R = 6371.0
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * R * np.arcsin(np.sqrt(a))


def _prep_location_stage(df, stage):
    """
    Standard columns (event_id, time, latitude, longitude, depth_km, magnitude)
    for one pipeline stage, one row per event (the earliest if an id repeats).

    stage = "gamma"   -> merged/input/<namebase>_catalog_cleaned.csv (event_id_mapped, depth in m)
            "hypoinv" -> merged/output/<namebase>_hypoinv_good.csv (event_id_mapped)
            "hypodd"  -> merged/output/<namebase>_hypodd_catalog*.csv (event_id)
    """
    df = df.copy()

    if stage == "gamma":
        out = pd.DataFrame({
            "event_id": df["event_id_mapped"],
            "time": df["time"],
            "latitude": df["latitude"],
            "longitude": df["longitude"],
            "depth_km": pd.to_numeric(df["depth(m)"], errors="coerce") / 1000.0,
            "magnitude": df["magnitude"] if "magnitude" in df.columns else np.nan,
        })

    elif stage == "hypoinv":
        out = pd.DataFrame({
            "event_id": df["event_id_mapped"],
            "time": df["time"],
            "latitude": df["latitude"],
            "longitude": df["longitude"],
            "depth_km": df["depth_km"],
            "magnitude": df["magnitude"] if "magnitude" in df.columns else np.nan,
        })

    elif stage == "hypodd":
        out = pd.DataFrame({
            "event_id": df["event_id"],
            "time": df["time"],
            "latitude": df["latitude"],
            "longitude": df["longitude"],
            "depth_km": df["depth_km"],
            "magnitude": df["magnitude"] if "magnitude" in df.columns else np.nan,
        })

    else:
        raise ValueError(f"Unknown stage: {stage} (use 'gamma', 'hypoinv' or 'hypodd').")

    out["event_id"] = pd.to_numeric(out["event_id"], errors="coerce").astype("Int64")
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce", format="mixed")

    for col in ["latitude", "longitude", "depth_km", "magnitude"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["event_id", "time", "latitude", "longitude"]).copy()
    out = out.sort_values("time").drop_duplicates(subset="event_id", keep="first")
    out["event_id"] = out["event_id"].astype(int)

    return out


def build_location_evolution_df(gamma_csv, hypoinv_csv, hypodd_csv, start=None, end=None):
    """
    Join the GaMMA, HypoInverse and HypoDD locations of the same events.

    Each input is a CSV path or a DataFrame. Only events present in all three
    stages are kept (inner join on event_id), optionally within [start, end] (UTC).
    Columns are prefixed gamma_ / hypoinv_ / hypodd_, plus the horizontal shifts
    in km (great-circle): shift_gamma_hypoinv_km, shift_hypoinv_hypodd_km and
    shift_gamma_hypodd_km.
    """
    def read(x):
        return x.copy() if isinstance(x, pd.DataFrame) else pd.read_csv(x)

    gamma = _prep_location_stage(read(gamma_csv), "gamma")
    hypoinv = _prep_location_stage(read(hypoinv_csv), "hypoinv")
    hypodd = _prep_location_stage(read(hypodd_csv), "hypodd")

    if start is not None:
        start = pd.to_datetime(start, utc=True)
        gamma = gamma[gamma["time"] >= start]
        hypoinv = hypoinv[hypoinv["time"] >= start]
        hypodd = hypodd[hypodd["time"] >= start]

    if end is not None:
        end = pd.to_datetime(end, utc=True)
        gamma = gamma[gamma["time"] <= end]
        hypoinv = hypoinv[hypoinv["time"] <= end]
        hypodd = hypodd[hypodd["time"] <= end]

    gamma = gamma.rename(columns={c: f"gamma_{c}" for c in gamma.columns if c != "event_id"})
    hypoinv = hypoinv.rename(columns={c: f"hypoinv_{c}" for c in hypoinv.columns if c != "event_id"})
    hypodd = hypodd.rename(columns={c: f"hypodd_{c}" for c in hypodd.columns if c != "event_id"})

    df = (
        gamma
        .merge(hypoinv, on="event_id", how="inner")
        .merge(hypodd, on="event_id", how="inner")
    )

    df["shift_gamma_hypoinv_km"] = _haversine_km(
        df["gamma_latitude"], df["gamma_longitude"],
        df["hypoinv_latitude"], df["hypoinv_longitude"],
    )

    df["shift_hypoinv_hypodd_km"] = _haversine_km(
        df["hypoinv_latitude"], df["hypoinv_longitude"],
        df["hypodd_latitude"], df["hypodd_longitude"],
    )

    df["shift_gamma_hypodd_km"] = _haversine_km(
        df["gamma_latitude"], df["gamma_longitude"],
        df["hypodd_latitude"], df["hypodd_longitude"],
    )

    return df


def _add_segments_geo(ax, x0, y0, x1, y1, color="0.20", lw=1.1, alpha=0.80):
    """One straight segment per event from (x0, y0) to (x1, y1), lon/lat in degrees."""
    import cartopy.crs as ccrs

    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        ax.plot(
            [xa, xb],
            [ya, yb],
            color=color,
            linewidth=lw,
            alpha=alpha,
            transform=ccrs.PlateCarree(),
            zorder=1,
        )


def add_panel_label(ax, label, fontsize=18):
    """
    Manuscript-style panel label, e.g. "(a)": bold, just outside the upper-left
    corner of the axes.
    """
    ax.text(
        0.0, 1.035,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
        clip_on=False,
        zorder=50,
    )


def plot_location_evolution_map(
    df,
    max_events=1500,
    random_state=42,
    figsize=(14, 7),
    relief_zoom=12,
    pad_lon=0.01,
    pad_lat=0.01,
    step_lon=0.10,
    step_lat=0.05,
    x_tick_label_size=18,
    y_tick_label_size=22,
    panel_label_size=18,
    stats_fontsize=13,
    legend_fontsize=14,
):
    """
    Two map panels showing how each event moves between pipeline stages:
    (a) GaMMA -> HypoInverse and (b) HypoInverse -> HypoDD. A segment joins the
    two locations of each event; each panel reports the median horizontal shift (km).

    df: output of build_location_evolution_df. If it has more than max_events
    rows, a random sample (random_state) is drawn so the map stays readable.
    Returns fig, axes, plot_df (the events actually drawn).
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    # ESRI shaded-relief tiles with download timeout, retries and disk cache (see _shaded_relief_tile_class)
    ShadedReliefESRI = _shaded_relief_tile_class()

    plot_df = df.copy()

    if max_events is not None and len(plot_df) > max_events:
        plot_df = plot_df.sample(max_events, random_state=random_state).copy()

    # Manuscript-style panel labels: lowercase, parentheses, bold
    panels = [
        ("gamma", "hypoinv", "(a)", "shift_gamma_hypoinv_km"),
        ("hypoinv", "hypodd", "(b)", "shift_hypoinv_hypodd_km"),
    ]

    lon_all = np.r_[
        plot_df["gamma_longitude"].values,
        plot_df["hypoinv_longitude"].values,
        plot_df["hypodd_longitude"].values,
    ]
    lat_all = np.r_[
        plot_df["gamma_latitude"].values,
        plot_df["hypoinv_latitude"].values,
        plot_df["hypodd_latitude"].values,
    ]

    minlon = float(np.nanmin(lon_all)) - pad_lon
    maxlon = float(np.nanmax(lon_all)) + pad_lon
    minlat = float(np.nanmin(lat_all)) - pad_lat
    maxlat = float(np.nanmax(lat_all)) + pad_lat
    extent = (minlon, maxlon, minlat, maxlat)

    center_lon = 0.5 * (minlon + maxlon)
    center_lat = 0.5 * (minlat + maxlat)

    proj = ccrs.LambertAzimuthalEqualArea(
        central_longitude=center_lon,
        central_latitude=center_lat,
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        subplot_kw={"projection": proj},
        sharex=False,
        sharey=False,
    )

    gridliners = []

    for ax, (a, b, panel_label, shift_col) in zip(axes, panels):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_image(ShadedReliefESRI(), int(relief_zoom))

        _add_segments_geo(
            ax,
            plot_df[f"{a}_longitude"].values,
            plot_df[f"{a}_latitude"].values,
            plot_df[f"{b}_longitude"].values,
            plot_df[f"{b}_latitude"].values,
        )

        if a == "gamma" and b == "hypoinv":
            ax.scatter(
                plot_df["gamma_longitude"],
                plot_df["gamma_latitude"],
                s=100,
                facecolors="none",
                edgecolors="#d95f02",
                linewidths=1.8,
                label="GaMMA",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

            ax.scatter(
                plot_df["hypoinv_longitude"],
                plot_df["hypoinv_latitude"],
                s=100,
                c="#1b9e77",
                edgecolors="none",
                label="HypoInverse",
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

        else:
            ax.scatter(
                plot_df["hypoinv_longitude"],
                plot_df["hypoinv_latitude"],
                s=100,
                c="#1b9e77",
                edgecolors="none",
                label="HypoInverse",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

            ax.scatter(
                plot_df["hypodd_longitude"],
                plot_df["hypodd_latitude"],
                s=100,
                facecolors="none",
                edgecolors="black",
                linewidths=1.6,
                label="HypoDD",
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

        add_panel_label(ax, panel_label, fontsize=panel_label_size)

        median_shift = plot_df[shift_col].median()

        ax.text(
            0.02, 0.04,
            f"median shift = {median_shift:.2f} km",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=stats_fontsize,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.9),
            zorder=10,
        )

        gl = ax.gridlines(
            draw_labels=True,
            linestyle="--",
            linewidth=0.5,
            alpha=0.4,
            x_inline=False,
            y_inline=False,
        )

        gl.top_labels = False
        gl.right_labels = False

        gl.xlabel_style = {
            "size": x_tick_label_size,
            "color": "black",
        }

        gl.ylabel_style = {
            "size": y_tick_label_size,
            "color": "black",
            "rotation": 90,
            "ha": "center",
            "va": "center",
        }

        x_start = np.floor(minlon / step_lon) * step_lon
        x_end = np.ceil(maxlon / step_lon) * step_lon

        y_start = np.floor(minlat / step_lat) * step_lat
        y_end = np.ceil(maxlat / step_lat) * step_lat

        gl.xlocator = mticker.FixedLocator(
            np.arange(x_start, x_end + step_lon * 0.5, step_lon)
        )

        gl.ylocator = mticker.FixedLocator(
            np.arange(y_start, y_end + step_lat * 0.5, step_lat)
        )

        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()

        gridliners.append(gl)

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            frameon=True,
            facecolor="white",
            framealpha=0.90,
            borderaxespad=0.0,
            fontsize=legend_fontsize,
        )

    plt.tight_layout(pad=1.2)

    # Cartopy creates the gridline label texts only at draw time; restyle them afterwards
    fig.canvas.draw()

    for gl in gridliners:
        for txt in gl.ylabel_artists:
            txt.set_rotation(90)
            txt.set_fontsize(y_tick_label_size)
            txt.set_ha("center")
            txt.set_va("center")

        for txt in gl.xlabel_artists:
            txt.set_fontsize(x_tick_label_size)

    return fig, axes, plot_df


# ============================================================
# FREQUENCY INDEX (FI) COMPOSITION BY MONTH
# ============================================================

FI_COMPOSITION_COLORS = {
    "LP": "#66b96b",
    "NS": "#808080",
    "VT": "#56B4E9",
}


def fi_monthly_tables(
    catalog,
    fi_col="FI",
    time_col="time",
    fi_order=("LP", "NS", "VT"),
    include_empty_months=False,
):
    """
    Monthly number of events per FI class, and the same as a percentage of each month.

    FI labels are stripped and upper-cased; times are parsed as UTC (format="mixed").
    Months are "YYYY-MM" strings. By default only months with events appear
    (as in the paper 2 figure), so a month without events leaves no gap on the
    axis; include_empty_months=True adds those months with zero counts.
    Returns (counts, percent), DataFrames indexed by month with fi_order columns.
    """
    df = catalog.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=[time_col])

    if fi_col in df.columns:
        df[fi_col] = df[fi_col].astype(str).str.strip().str.upper()
    else:
        df[fi_col] = "UNKNOWN"

    month = df[time_col].dt.to_period("M").astype(str)
    counts = pd.crosstab(month, df[fi_col])
    counts = counts.reindex(columns=[c for c in fi_order if c in counts.columns], fill_value=0)

    if include_empty_months and len(counts):
        all_months = pd.period_range(counts.index.min(), counts.index.max(), freq="M").astype(str)
        counts = counts.reindex(all_months, fill_value=0)

    counts.index.name = "month"
    # empty months give 0/0 -> NaN; show them as 0 %
    percent = (counts.div(counts.sum(axis=1), axis=0) * 100).fillna(0.0)
    return counts, percent


def add_month_spans_inside_axis(
    ax,
    month_labels,
    spans,
    *,
    top_margin_fraction=0.03,
    band_height_fraction=0.035,
    band_gap_fraction=0.010,
    cmap_name="Set2",
    alpha=0.95,
    text=False,
    text_fontsize=10,
):
    """
    Draw horizontal bands for time spans INSIDE the top of a monthly bar chart.

    month_labels: the "YYYY-MM" labels of the bars, in order (bar i is at x = i).
    spans: list of (start, end, label); spans that share a label share a band row
    and colour. The rows are stacked downwards from the top so they all fit.
    Each band covers the bars of the months from start to end that are present.
    Returns (legend handles, {label: colour}).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    month_labels = pd.Index(month_labels).astype(str)
    month_to_idx = {m: i for i, m in enumerate(month_labels)}

    span_types = list(dict.fromkeys(label for _, _, label in spans))

    cmap = plt.get_cmap(cmap_name)
    palette = getattr(cmap, "colors", None)
    if palette is None:
        palette = [cmap(i) for i in np.linspace(0, 1, max(1, len(span_types)))]
    color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(span_types)}

    ymin, ymax = ax.get_ylim()
    yr = ymax - ymin

    band_height = band_height_fraction * yr
    band_gap = band_gap_fraction * yr
    top_margin = top_margin_fraction * yr

    n_bands = len(span_types)
    total_band_block = n_bands * band_height + max(0, n_bands - 1) * band_gap

    # Start just low enough for ALL band rows to fit under the top
    first_y0 = ymax - top_margin - total_band_block

    band_pos = {}
    current_y0 = first_y0
    for lab in span_types:
        band_pos[lab] = (current_y0, current_y0 + band_height)
        current_y0 += band_height + band_gap

    handles = []
    seen = set()

    for start, end, label in spans:
        start_ts = pd.to_datetime(start, errors="coerce")
        end_ts = pd.to_datetime(end, errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue

        start_month = start_ts.to_period("M").strftime("%Y-%m")
        end_month = end_ts.to_period("M").strftime("%Y-%m")

        valid_months = month_labels[(month_labels >= start_month) & (month_labels <= end_month)]
        if len(valid_months) == 0:
            continue

        left_month = valid_months.min()
        right_month = valid_months.max()

        i0 = month_to_idx[left_month]
        i1 = month_to_idx[right_month]

        x0 = i0 - 0.45
        width = (i1 - i0) + 0.90

        y0, y1 = band_pos[label]

        rect = plt.Rectangle(
            (x0, y0),
            width,
            y1 - y0,
            facecolor=color_map[label],
            edgecolor="none",
            alpha=alpha,
            zorder=6,
            clip_on=True,
        )
        ax.add_patch(rect)

        if text:
            ax.text(
                x0 + width / 2,
                y0 + (y1 - y0) / 2,
                label,
                ha="center",
                va="center",
                fontsize=text_fontsize,
                zorder=7,
            )

        if label not in seen:
            handles.append(mpatches.Patch(color=color_map[label], label=label))
            seen.add(label)

    return handles, color_map


def plot_fi_monthly_composition(
    catalog,
    spans=None,
    *,
    fi_col="FI",
    time_col="time",
    fi_order=("LP", "NS", "VT"),
    fi_colors=None,
    include_empty_months=False,
    title=None,
    figsize=(16, 6),
    headroom=1.38,
    span_cmap="Set2",
    span_alpha=0.9,
    savefig=False,
    out_file=None,
    show=False,
):
    """
    Two-panel monthly FI figure (as in the paper 2 statistics notebook):
    top = stacked number of events per FI class, with optional time-span bands;
    bottom = the same composition in percent.

    spans: list of (start, end, label), e.g. ("2023-03-01", "2023-08-01", "Uplift-AVO").
    headroom: the top panel's y-limit is the largest monthly total times this
    factor, leaving room for the bands. fi_colors defaults to FI_COMPOSITION_COLORS.
    Returns fig, (ax_counts, ax_percent), info with counts, percent,
    span_handles and span_color_map (use them for the separate legend figure).
    """
    import matplotlib.pyplot as plt

    fi_colors = dict(FI_COMPOSITION_COLORS if fi_colors is None else fi_colors)
    counts, percent = fi_monthly_tables(catalog, fi_col=fi_col, time_col=time_col,
                                        fi_order=fi_order, include_empty_months=include_empty_months)
    bar_colors = [fi_colors[col] for col in counts.columns]

    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.09}
    )

    # Top panel: counts
    counts.plot(
        kind="bar",
        stacked=True,
        ax=ax1,
        width=0.9,
        color=bar_colors,
        legend=False,
    )

    ax1.set_ylabel("Number of events", fontsize=17)
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", labelbottom=False)
    ax1.tick_params(axis="y", labelsize=15)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    # Extra room at the top INSIDE the axis for the span bands
    ax1.set_ylim(0, counts.sum(axis=1).max() * headroom)

    span_handles, span_color_map = [], {}
    if spans:
        span_handles, span_color_map = add_month_spans_inside_axis(
            ax=ax1,
            month_labels=counts.index,
            spans=spans,
            top_margin_fraction=0.03,
            band_height_fraction=0.035,
            band_gap_fraction=0.010,
            cmap_name=span_cmap,
            alpha=span_alpha,
            text=False,
            text_fontsize=10,
        )

    if title:
        ax1.set_title(title, fontsize=24, pad=18)

    # Bottom panel: percentage
    percent.plot(
        kind="bar",
        stacked=True,
        ax=ax2,
        width=0.7,
        legend=False,
        color=bar_colors
    )

    ax2.set_ylabel("Percentage (%)", fontsize=17)
    ax2.set_xlabel("Month", fontsize=17)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=90, labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if savefig and out_file:
        fig.savefig(out_file, bbox_inches="tight")
        print(f"[ok] saved: {out_file}")
    if show:
        plt.show()

    info = dict(counts=counts, percent=percent, span_handles=span_handles, span_color_map=span_color_map)
    return fig, (ax1, ax2), info


def plot_monthly_fi_span_legend_only(
    *,
    fi_colors,
    fi_order,
    span_color_map,
    span_order,
    label_map=None,
    figsize=(4.2, 3.2),
    legend_fontsize=15,
    ncol=1,
    title=None,
    show=False,
):
    """
    Separate figure holding only the legend of plot_fi_monthly_composition:
    one patch per FI class (NS shown as "Unclassified" by default) and per span.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if label_map is None:
        label_map = {
            "LP": "LP",
            "NS": "Unclassified",
            "VT": "VT",
        }

    handles = []
    for fi in fi_order:
        handles.append(mpatches.Patch(color=fi_colors[fi], label=label_map.get(fi, fi)))

    for s in span_order:
        handles.append(mpatches.Patch(color=span_color_map[s], label=s))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        fontsize=legend_fontsize,
        ncol=ncol,
    )

    if title is not None:
        ax.set_title(title, fontsize=legend_fontsize + 1)

    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax


# ============================================================
# CATALOG COMPARISON WITH REFERENCE CATALOGS (e.g. USGS, AVO)
# ============================================================

def load_and_normalize(filepath, column_mapping, depth_in_meters=False, **kwargs):
    """
    Read a catalog CSV and give it the standard comparison columns:
    time (UTC), latitude, longitude, depth_km, mag, event_id (as text).

    column_mapping renames the file's columns, e.g. {"depth(m)": "depth_km",
    "magnitude": "mag"}; depth_in_meters=True divides depth by 1000.
    Rows without time, coordinates or depth are dropped; sorted by time.
    """
    df = pd.read_csv(filepath, **kwargs)

    rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=rename_dict).copy()
    df = df.loc[:, ~df.columns.duplicated()].copy()

    required = ["time", "latitude", "longitude", "depth_km", "mag", "event_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns after renaming: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Column mapping: {column_mapping}"
        )

    # format="mixed": catalogs mix "...:08.61+00:00" and "...:46+00:00" time strings
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="mixed")

    for col in ["latitude", "longitude", "depth_km", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if depth_in_meters:
        df["depth_km"] = df["depth_km"] / 1000.0

    df["event_id"] = df["event_id"].astype(str)
    df = df.dropna(subset=["time", "latitude", "longitude", "depth_km"]).copy()
    df = df[required].copy()
    df = df.sort_values("time").reset_index(drop=True)

    return df


def filter_catalog_by_year(df, year):
    """Events of one calendar year (UTC), sorted by time."""
    out = df[df["time"].dt.year == year].copy()
    return out.sort_values("time").reset_index(drop=True)


def obspy_horizontal_distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km on the WGS84 ellipsoid (ObsPy gps2dist_azimuth)."""
    from obspy.geodetics.base import gps2dist_azimuth

    dist_m, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    return dist_m / 1000.0


def obspy_distance_3d_km(lat1, lon1, depth1_km, lat2, lon2, depth2_km):
    """
    (dh_km, dz_km, d3_km): horizontal distance, depth difference (point 2 - point 1,
    positive = point 2 deeper) and their combination sqrt(dh^2 + dz^2), all in km.
    """
    dh_km = obspy_horizontal_distance_km(lat1, lon1, lat2, lon2)
    dz_km = depth2_km - depth1_km
    d3_km = np.sqrt(dh_km**2 + dz_km**2)
    return dh_km, dz_km, d3_km


def match_catalogs_time_space(ref_df, qry_df, time_tol_s, dist_tol_km):
    """
    All candidate pairs between a reference catalog (e.g. USGS) and a query catalog
    (e.g. QuickQuake): |origin-time difference| <= time_tol_s (s) and 3-D distance
    <= dist_tol_km (km). Each pair gets score = |dt|/time_tol_s + d3/dist_tol_km
    (lower = better). dt_sec and dz_km are query - reference.
    One reference event can have several candidates; see enforce_one_to_one.
    """
    ref = ref_df.sort_values("time").reset_index(drop=True).copy()
    qry = qry_df.sort_values("time").reset_index(drop=True).copy()

    rows = []

    for _, ref_row in ref.iterrows():
        t0 = ref_row["time"] - pd.Timedelta(seconds=time_tol_s)
        t1 = ref_row["time"] + pd.Timedelta(seconds=time_tol_s)

        cand = qry[(qry["time"] >= t0) & (qry["time"] <= t1)].copy()
        if cand.empty:
            continue

        cand["dt_sec"] = (cand["time"] - ref_row["time"]).dt.total_seconds()
        cand["abs_dt_sec"] = cand["dt_sec"].abs()

        dh_list, dz_list, d3_list = [], [], []
        for _, qrow in cand.iterrows():
            dh_km, dz_km, d3_km = obspy_distance_3d_km(
                ref_row["latitude"], ref_row["longitude"], ref_row["depth_km"],
                qrow["latitude"], qrow["longitude"], qrow["depth_km"],
            )
            dh_list.append(dh_km)
            dz_list.append(dz_km)
            d3_list.append(d3_km)

        cand["dh_km"] = dh_list
        cand["dz_km"] = dz_list
        cand["d3_km"] = d3_list
        cand = cand[cand["d3_km"] <= dist_tol_km].copy()

        if cand.empty:
            continue

        cand["score"] = (
            cand["abs_dt_sec"] / time_tol_s +
            cand["d3_km"] / dist_tol_km
        )

        for _, qrow in cand.iterrows():
            row = {f"{col}_ref": ref_row[col] for col in ref.columns}
            row.update({f"{col}_qry": qrow[col] for col in qry.columns})
            row["dt_sec"] = qrow["dt_sec"]
            row["abs_dt_sec"] = qrow["abs_dt_sec"]
            row["dh_km"] = qrow["dh_km"]
            row["dz_km"] = qrow["dz_km"]
            row["d3_km"] = qrow["d3_km"]
            row["score"] = qrow["score"]
            rows.append(row)

    return pd.DataFrame(rows)


def enforce_one_to_one(matched_candidates):
    """
    Keep at most one partner per event, preferring the lowest score: first the
    best candidate of each reference event, then the best of each query event.
    This is a greedy choice: a reference event whose best partner was taken by
    another event is not re-paired with its second-best candidate.
    """
    if matched_candidates.empty:
        return matched_candidates.copy()

    out = matched_candidates.sort_values("score").drop_duplicates(
        subset=["event_id_ref"], keep="first"
    )
    out = out.sort_values("score").drop_duplicates(
        subset=["event_id_qry"], keep="first"
    )
    return out.sort_values("time_ref").reset_index(drop=True)


def check_one_to_one(matched_df, comparison_name="comparison"):
    """Print how many matches there are and whether any event appears twice."""
    n_total = len(matched_df)
    n_dup_ref = matched_df["event_id_ref"].duplicated().sum()
    n_dup_qry = matched_df["event_id_qry"].duplicated().sum()

    print(comparison_name)
    print(f"  total matches: {n_total}")
    print(f"  duplicate reference events: {n_dup_ref}")
    print(f"  duplicate query events: {n_dup_qry}")

    if n_dup_ref == 0 and n_dup_qry == 0:
        print("  one-to-one matching confirmed")
    else:
        print("  duplicates still present")


def compute_residuals(matched_df):
    """
    Recompute dh_km, dz_km, d3_km for each matched pair and add dmag = mag_qry - mag_ref.
    """
    out = matched_df.copy()

    dh_list, dz_list, d3_list = [], [], []
    for _, row in out.iterrows():
        dh_km, dz_km, d3_km = obspy_distance_3d_km(
            row["latitude_ref"], row["longitude_ref"], row["depth_km_ref"],
            row["latitude_qry"], row["longitude_qry"], row["depth_km_qry"],
        )
        dh_list.append(dh_km)
        dz_list.append(dz_km)
        d3_list.append(d3_km)

    out["dh_km"] = dh_list
    out["dz_km"] = dz_list
    out["d3_km"] = d3_list
    out["dmag"] = out["mag_qry"] - out["mag_ref"]
    return out


def calculate_pair_metrics(matched_df, n_ref_buffer, n_qry_buffer):
    """
    Summary of one comparison: number of matches, recall of the reference catalog
    (matches / reference events), fraction of the query catalog matched, and
    mean / std / median / 90th percentile of dt (s), d3 and dz (km) and dmag.
    """
    n_matches = len(matched_df)

    if n_matches == 0:
        return {
            "n_ref_buffer": n_ref_buffer,
            "n_qry_buffer": n_qry_buffer,
            "n_matches": 0,
            "recall_ref": np.nan,
            "matched_fraction_qry": np.nan,
            "dt_mean": np.nan,
            "dt_std": np.nan,
            "dt_median_abs": np.nan,
            "dt_p90_abs": np.nan,
            "d3_mean": np.nan,
            "d3_std": np.nan,
            "d3_median": np.nan,
            "d3_p90": np.nan,
            "dz_mean": np.nan,
            "dz_std": np.nan,
            "dmag_mean": np.nan,
            "dmag_std": np.nan,
        }

    return {
        "n_ref_buffer": n_ref_buffer,
        "n_qry_buffer": n_qry_buffer,
        "n_matches": n_matches,
        "recall_ref": n_matches / n_ref_buffer if n_ref_buffer > 0 else np.nan,
        "matched_fraction_qry": n_matches / n_qry_buffer if n_qry_buffer > 0 else np.nan,
        "dt_mean": matched_df["dt_sec"].mean(),
        "dt_std": matched_df["dt_sec"].std(),
        "dt_median_abs": matched_df["dt_sec"].abs().median(),
        "dt_p90_abs": matched_df["dt_sec"].abs().quantile(0.90),
        "d3_mean": matched_df["d3_km"].mean(),
        "d3_std": matched_df["d3_km"].std(),
        "d3_median": matched_df["d3_km"].median(),
        "d3_p90": matched_df["d3_km"].quantile(0.90),
        "dz_mean": matched_df["dz_km"].mean(),
        "dz_std": matched_df["dz_km"].std(),
        "dmag_mean": matched_df["dmag"].mean(),
        "dmag_std": matched_df["dmag"].std(),
    }


def run_all_comparisons(catalog_buf, comparisons, catalogs, params, verbose=True):
    """
    Match every (reference, query) pair listed in comparisons, e.g.
    {"ref": "usgs", "qry": "qq", "name": "QuickQuake vs USGS"}, using
    params["time_tol_s"] (s) and params["dist_tol_km"] (km).
    catalog_buf: {key: catalog} already cut to the same area and time window.
    Returns {(ref_key, qry_key): {"name", "ref_label", "qry_label", "matches", "metrics", ...}}.
    """
    pair_results = {}

    for cfg in comparisons:
        ref_key = cfg["ref"]
        qry_key = cfg["qry"]
        comparison_name = cfg["name"]

        ref_df = catalog_buf[ref_key].copy()
        qry_df = catalog_buf[qry_key].copy()

        matched_candidates = match_catalogs_time_space(
            ref_df,
            qry_df,
            time_tol_s=params["time_tol_s"],
            dist_tol_km=params["dist_tol_km"],
        )
        matched_df = enforce_one_to_one(matched_candidates)
        matched_df = compute_residuals(matched_df)

        metrics = calculate_pair_metrics(
            matched_df,
            n_ref_buffer=len(ref_df),
            n_qry_buffer=len(qry_df),
        )

        pair_results[(ref_key, qry_key)] = {
            "name": comparison_name,
            "ref_key": ref_key,
            "qry_key": qry_key,
            "ref_label": catalogs[ref_key]["label"],
            "qry_label": catalogs[qry_key]["label"],
            "matches": matched_df,
            "metrics": metrics,
        }

        if verbose:
            print(f"\n{comparison_name}")
            print(f"  total matches: {len(matched_df)}")
            print(f"  duplicate reference events: {matched_df['event_id_ref'].duplicated().sum()}")
            print(f"  duplicate query events: {matched_df['event_id_qry'].duplicated().sum()}")

    return pair_results


def build_metrics_table(pair_results):
    """One row per comparison with readable column names and units (rounded to 3 decimals)."""
    rows = []

    for result in pair_results.values():
        row = {
            "Comparison": result["name"],
            "Reference catalog": result["ref_label"],
            "Query catalog": result["qry_label"],
            **result["metrics"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.rename(columns={
        "n_ref_buffer": "N in reference buffer",
        "n_qry_buffer": "N in query buffer",
        "n_matches": "One-to-one matches",
        "recall_ref": "Recall of reference by query",
        "matched_fraction_qry": "Fraction of query matched",
        "dt_mean": "Mean Δt (s)",
        "dt_std": "Std Δt (s)",
        "dt_median_abs": "Median |Δt| (s)",
        "dt_p90_abs": "P90 |Δt| (s)",
        "d3_mean": "Mean d3 (km)",
        "d3_std": "Std d3 (km)",
        "d3_median": "Median d3 (km)",
        "d3_p90": "P90 d3 (km)",
        "dz_mean": "Mean Δz (km)",
        "dz_std": "Std Δz (km)",
        "dmag_mean": "Mean Δm",
        "dmag_std": "Std Δm",
    })

    cols_order = [
        "Comparison",
        "Reference catalog",
        "Query catalog",
        "N in reference buffer",
        "N in query buffer",
        "One-to-one matches",
        "Recall of reference by query",
        "Fraction of query matched",
        "Mean d3 (km)",
        "Std d3 (km)",
        "Median d3 (km)",
        "P90 d3 (km)",
        "Mean Δt (s)",
        "Std Δt (s)",
        "Median |Δt| (s)",
        "P90 |Δt| (s)",
        "Mean Δz (km)",
        "Std Δz (km)",
        "Mean Δm",
        "Std Δm",
    ]

    df = df[cols_order]

    for col in df.columns:
        if col not in ["Comparison", "Reference catalog", "Query catalog"]:
            df[col] = df[col].astype(float).round(3)

    return df


def add_component_residuals_km(matched_df):
    """
    Signed residuals, query - reference: dx_km (east-west), dy_km (north-south),
    dz_km (depth, positive = query deeper) and dt_sec. dx/dy use a local
    conversion of 111.32 km per degree (dx scaled by cos of the mean latitude).
    """
    out = matched_df.copy()

    # Signed geographic residuals: query - reference
    out["dlon_deg"] = out["longitude_qry"] - out["longitude_ref"]
    out["dlat_deg"] = out["latitude_qry"] - out["latitude_ref"]
    out["dz_km"] = out["depth_km_qry"] - out["depth_km_ref"]

    if "dt_sec" not in out.columns:
        out["dt_sec"] = (
            pd.to_datetime(out["time_qry"], utc=True) -
            pd.to_datetime(out["time_ref"], utc=True)
        ).dt.total_seconds()

    # Local conversion to signed km components
    mean_lat_rad = np.deg2rad(0.5 * (out["latitude_ref"] + out["latitude_qry"]))
    out["dy_km"] = 111.32 * out["dlat_deg"]
    out["dx_km"] = 111.32 * np.cos(mean_lat_rad) * out["dlon_deg"]

    return out


def build_km_component_metrics_table(matched_df):
    """Mean, std and median of dx, dy, dz (km) and dt (s) for one comparison."""
    df = add_component_residuals_km(matched_df)

    rows = [
        {
            "Metric": "Δx east-west (km)",
            "Mean": df["dx_km"].mean(),
            "Std": df["dx_km"].std(),
            "Median": df["dx_km"].median(),
        },
        {
            "Metric": "Δy north-south (km)",
            "Mean": df["dy_km"].mean(),
            "Std": df["dy_km"].std(),
            "Median": df["dy_km"].median(),
        },
        {
            "Metric": "Δz depth (km)",
            "Mean": df["dz_km"].mean(),
            "Std": df["dz_km"].std(),
            "Median": df["dz_km"].median(),
        },
        {
            "Metric": "Δt (s)",
            "Mean": df["dt_sec"].mean(),
            "Std": df["dt_sec"].std(),
            "Median": df["dt_sec"].median(),
        },
    ]

    table = pd.DataFrame(rows)
    for col in ["Mean", "Std", "Median"]:
        table[col] = table[col].round(5)

    return table


def make_residual_summary_tables(residual_data, decimals=3):
    """
    Creates manuscript-friendly summary tables for QuickQuake residuals
    relative to USGS and AVO.

    Uses residual_data returned by:
        fig, axes, residual_data = plot_combined_reference_residual_histograms(...)
    """

    component_info = {
        "dx_km": {
            "label": r"$\Delta x$",
            "meaning": "east-west",
            "unit": "km",
        },
        "dy_km": {
            "label": r"$\Delta y$",
            "meaning": "north-south",
            "unit": "km",
        },
        "dz_km": {
            "label": r"$\Delta z$",
            "meaning": "depth",
            "unit": "km",
        },
        "dt_sec": {
            "label": r"$\Delta t$",
            "meaning": "origin time",
            "unit": "s",
        },
    }

    rows = []

    for pair_key, item in residual_data.items():
        reference = item["cfg"]["label"]
        df = item["df"].copy()

        for col, info in component_info.items():
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]

            if len(vals) == 0:
                continue

            rows.append({
                "Reference catalog": reference,
                "Residual": info["label"],
                "Component": info["meaning"],
                "Unit": info["unit"],
                "N": len(vals),
                "Median": np.median(vals),
                "Mean": np.mean(vals),
                "Std. dev.": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
                "16th perc.": np.percentile(vals, 16),
                "84th perc.": np.percentile(vals, 84),
                "5th perc.": np.percentile(vals, 5),
                "95th perc.": np.percentile(vals, 95),
                "Median absolute residual": np.median(np.abs(vals)),
                "95th perc. absolute residual": np.percentile(np.abs(vals), 95),
            })

    component_table = pd.DataFrame(rows)

    # Round numerical columns
    num_cols = component_table.select_dtypes(include=[np.number]).columns
    component_table[num_cols] = component_table[num_cols].round(decimals)

    # ------------------------------------------------------------------
    # Compact paragraph-ready table
    # ------------------------------------------------------------------
    paragraph_rows = []

    for pair_key, item in residual_data.items():
        reference = item["cfg"]["label"]
        df = item["df"].copy()

        for col in ["dx_km", "dy_km", "dz_km", "dt_sec"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        clean = df[["dx_km", "dy_km", "dz_km", "dt_sec"]].dropna()

        horizontal_offset = np.sqrt(clean["dx_km"]**2 + clean["dy_km"]**2)

        paragraph_rows.append({
            "Reference catalog": reference,
            "N matched events": len(clean),
            "Median Δx (km)": np.median(clean["dx_km"]),
            "Median Δy (km)": np.median(clean["dy_km"]),
            "Median Δz (km)": np.median(clean["dz_km"]),
            "Median Δt (s)": np.median(clean["dt_sec"]),
            "Median horizontal offset (km)": np.median(horizontal_offset),
            "Median |Δz| (km)": np.median(np.abs(clean["dz_km"])),
            "Median |Δt| (s)": np.median(np.abs(clean["dt_sec"])),
            "95th perc. horizontal offset (km)": np.percentile(horizontal_offset, 95),
            "95th perc. |Δz| (km)": np.percentile(np.abs(clean["dz_km"]), 95),
            "95th perc. |Δt| (s)": np.percentile(np.abs(clean["dt_sec"]), 95),
        })

    paragraph_table = pd.DataFrame(paragraph_rows)

    num_cols = paragraph_table.select_dtypes(include=[np.number]).columns
    paragraph_table[num_cols] = paragraph_table[num_cols].round(decimals)

    return component_table, paragraph_table


def plot_match_histograms(matched_df, comparison_name, save_path=None, show=True):
    import matplotlib.pyplot as plt

    if matched_df is None or len(matched_df) == 0:
        print(f"No matches for {comparison_name}")
        return

    # Compact figure so the text looks larger
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    color_d3 = "#4C78A8"
    color_dt = "#F58518"
    line_med = "#222222"

    # Font sizes
    label_fs = 13
    tick_fs = 11
    title_fs = 14
    box_fs = 11
    suptitle_fs = 15

    # -------------------------
    # (a) 3-D distance
    # -------------------------
    ax = axes[0]
    ax.hist(
        matched_df["d3_km"],
        bins=30,
        color=color_d3,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8
    )

    d3_med = matched_df["d3_km"].median()

    ax.axvline(d3_med, color=line_med, linestyle="--", linewidth=1.8)

    ax.text(
        0.97, 0.95,
        f"n = {len(matched_df)}\nMedian = {d3_med:.2f} km",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=box_fs,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.85",
            alpha=0.95
        )
    )

    ax.set_xlabel("Distance (km)", fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.set_title("(a) Distance", loc="left", fontsize=title_fs, fontweight="bold")
    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_fs)

    # -------------------------
    # (b) Time residual
    # -------------------------
    ax = axes[1]
    ax.hist(
        matched_df["dt_sec"],
        bins=30,
        color=color_dt,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8
    )

    dt_med = matched_df["dt_sec"].median()

    ax.axvline(dt_med, color=line_med, linestyle="--", linewidth=1.8)

    ax.text(
        0.97, 0.95,
        f"n = {len(matched_df)}\nMedian = {dt_med:.2f} s",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=box_fs,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="0.85",
            alpha=0.95
        )
    )

    ax.set_xlabel("Time residual Δt (s)", fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.set_title("(b) Time", loc="left", fontsize=title_fs, fontweight="bold")
    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=tick_fs)

    fig.suptitle(comparison_name, fontsize=suptitle_fs, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    return fig, axes


def plot_match_km_histograms(
    matched_df,
    comparison_name,
    save_path=None,
    bins=30,
    show=True,
):
    import matplotlib.pyplot as plt

    if matched_df is None or len(matched_df) == 0:
        print(f"No matches for {comparison_name}")
        return

    df = add_component_residuals_km(matched_df)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    specs = [
        ("dx_km",  "Δx east-west (km)",   "#4C78A8", "(a)"),
        ("dy_km",  "Δy north-south (km)", "#F58518", "(b)"),
        ("dz_km",  "Δz depth (km)",       "#54A24B", "(c)"),
        ("dt_sec", "Δt (s)",              "#B279A2", "(d)"),
    ]

    for ax, (col, xlabel, color, panel) in zip(axes.flat, specs):
        vals = df[col].dropna()

        ax.hist(
            vals,
            bins=bins,
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.8,
        )

        mean = vals.mean()
        med = vals.median()

        ax.axvline(mean, color="#C44E52", linestyle="-", linewidth=1.8, label="Mean")
        ax.axvline(med, color="#222222", linestyle="--", linewidth=1.8, label="Median")

        units = "s" if col == "dt_sec" else "km"
        ax.text(
            0.97, 0.95,
            f"n = {len(vals)}\nMean = {mean:.3f} {units}\nMedian = {med:.3f} {units}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="0.85",
                alpha=0.95,
            ),
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.set_title(panel, loc="left", fontsize=14, fontweight="bold")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(axis="y", alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(comparison_name, y=0.98, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    return fig, axes


def plot_match_km_scatter(
    matched_df,
    comparison_name,
    save_path=None,
    show=True,
):
    import matplotlib.pyplot as plt

    if matched_df is None or len(matched_df) == 0:
        print(f"No matches for {comparison_name}")
        return

    df = add_component_residuals_km(matched_df)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].scatter(df["dx_km"], df["dy_km"], s=12, alpha=0.35, edgecolors="none")
    axes[0].axhline(0, color="0.3", lw=1)
    axes[0].axvline(0, color="0.3", lw=1)
    axes[0].set_xlabel("Δx east-west (km)")
    axes[0].set_ylabel("Δy north-south (km)")
    axes[0].set_title("(a)", loc="left", fontweight="bold")

    axes[1].scatter(df["dx_km"], df["dz_km"], s=12, alpha=0.35, edgecolors="none")
    axes[1].axhline(0, color="0.3", lw=1)
    axes[1].axvline(0, color="0.3", lw=1)
    axes[1].set_xlabel("Δx east-west (km)")
    axes[1].set_ylabel("Δz depth (km)")
    axes[1].set_title("(b)", loc="left", fontweight="bold")

    axes[2].scatter(df["dy_km"], df["dz_km"], s=12, alpha=0.35, edgecolors="none")
    axes[2].axhline(0, color="0.3", lw=1)
    axes[2].axvline(0, color="0.3", lw=1)
    axes[2].set_xlabel("Δy north-south (km)")
    axes[2].set_ylabel("Δz depth (km)")
    axes[2].set_title("(c)", loc="left", fontweight="bold")

    for ax in axes:
        ax.grid(alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(comparison_name, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    return fig, axes


def plot_combined_reference_residual_histograms(
    pair_results,
    save_path=None,
    bins=32,
    use_density=False,
    spatial_plot_limit_km=8.0,
    time_plot_limit_s=2.0,
    show=True,
):
    """
    Plot one combined manuscript-ready residual histogram figure comparing:

        QuickQuake vs USGS
        QuickQuake vs AVO

    The function expects:
        pair_results[("usgs", "qq")]["matches"]
        pair_results[("avo", "qq")]["matches"]

    It also expects that add_component_residuals_km(matched_df) already exists
    and creates:
        dx_km, dy_km, dz_km, dt_sec

    Sign convention:
        residual = query - reference

        dx_km > 0: QuickQuake event is farther east than reference
        dy_km > 0: QuickQuake event is farther north than reference
        dz_km > 0: QuickQuake event is deeper than reference
        dt_sec > 0: QuickQuake event is later than reference
    """
    import matplotlib.pyplot as plt


    required_pairs = [
        {
            "key": ("usgs", "qq"),
            "label": "USGS",
            "short_label": "USGS",
            "color": "#1f77b4",   # blue
            "linestyle": "-",
        },
        {
            "key": ("avo", "qq"),
            "label": "AVO",
            "short_label": "AVO",
            "color": "#ff7f0e",   # orange
            "linestyle": "-",
        },
    ]

    # ------------------------------------------------------------------
    # Extract and compute signed residual components
    # ------------------------------------------------------------------
    residual_data = {}

    for cfg in required_pairs:
        pair_key = cfg["key"]

        if pair_key not in pair_results:
            raise KeyError(
                f"Missing pair_results[{pair_key}]. "
                "Expected pair_results[(ref_key, qry_key)] with "
                "('usgs', 'qq') and ('avo', 'qq')."
            )

        matched_df = pair_results[pair_key]["matches"]

        if matched_df is None or len(matched_df) == 0:
            raise ValueError(f"No matches found for pair_results[{pair_key}].")

        df = add_component_residuals_km(matched_df).copy()

        needed_cols = ["dx_km", "dy_km", "dz_km", "dt_sec"]
        missing_cols = [col for col in needed_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(
                f"Missing residual columns after add_component_residuals_km(): {missing_cols}"
            )

        for col in needed_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        residual_data[pair_key] = {
            "cfg": cfg,
            "df": df,
        }

    # ------------------------------------------------------------------
    # Figure style
    # ------------------------------------------------------------------
    old_rc = plt.rcParams.copy()

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    variables = [
        {
            "col": "dx_km",
            "xlabel": r"$\Delta x$ east-west residual (km)",
            "title": r"(a)",
        },
        {
            "col": "dy_km",
            "xlabel": r"$\Delta y$ north-south residual (km)",
            "title": r"(b)",
        },
        {
            "col": "dz_km",
            "xlabel": r"$\Delta z$ depth residual (km)",
            "title": r"(c)",
        },
        {
            "col": "dt_sec",
            "xlabel": r"$\Delta t$ time residual (s)",
            "title": r"(d)",
        },
    ]

    try:
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(7.2, 5.6),
            constrained_layout=False,
        )

        axes = axes.ravel()

        for ax, spec in zip(axes, variables):
            col = spec["col"]

            vals_by_pair = {}
            combined_vals = []

            for pair_key, item in residual_data.items():
                vals = item["df"][col].dropna().to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                vals_by_pair[pair_key] = vals
                combined_vals.append(vals)

            combined_vals = np.concatenate(combined_vals)

            if len(combined_vals) == 0:
                ax.set_visible(False)
                continue

            # ----------------------------------------------------------
            # Fixed symmetric plotting limits:
            #   dx, dy, dz: -8 to 8 km
            #   dt:        -2 to 2 s
            # ----------------------------------------------------------
            if col == "dt_sec":
                max_abs = time_plot_limit_s
            else:
                max_abs = spatial_plot_limit_km

            bin_edges = np.linspace(-max_abs, max_abs, bins + 1)

            # ----------------------------------------------------------
            # Histograms
            # ----------------------------------------------------------
            for pair_key, item in residual_data.items():
                cfg = item["cfg"]
                vals = vals_by_pair[pair_key]

                ax.hist(
                    vals,
                    bins=bin_edges,
                    histtype="step",
                    linewidth=1.45,
                    color=cfg["color"],
                    linestyle=cfg["linestyle"],
                    label=cfg["label"],
                    density=use_density,
                )

            # Force x-axis limits
            if col == "dt_sec":
                ax.set_xlim(-time_plot_limit_s, time_plot_limit_s)
            else:
                ax.set_xlim(-spatial_plot_limit_km, spatial_plot_limit_km)

            # Zero residual line
            ax.axvline(
                0.0,
                color="0.20",
                linestyle="-",
                linewidth=0.9,
                alpha=0.85,
                zorder=0,
            )

            # Formatting
            ax.set_title(spec["title"], loc="left", fontweight="bold", pad=4)
            ax.set_xlabel(spec["xlabel"])
            ax.set_ylabel("Density" if use_density else "Count")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(0.8)
            ax.spines["bottom"].set_linewidth(0.8)

            ax.grid(
                axis="y",
                color="0.88",
                linewidth=0.6,
                linestyle="-",
                alpha=0.7,
            )
            ax.grid(axis="x", visible=False)

            ax.tick_params(axis="both", direction="out", width=0.8, length=3)

        # ------------------------------------------------------------------
        # One concise legend for the whole figure
        # ------------------------------------------------------------------
        legend_handles = []
        legend_labels = []

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_handles = handles
                legend_labels = labels
                break

        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=2,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.5,
        )

        fig.tight_layout(rect=[0.02, 0.04, 1.0, 0.93])

        # ------------------------------------------------------------------
        # Save both PNG and PDF
        # ------------------------------------------------------------------
        if save_path is not None:
            save_path = Path(save_path)

            if save_path.suffix.lower() in [".png", ".pdf"]:
                base = save_path.with_suffix("")
            else:
                base = save_path

            png_path = base.with_suffix(".png")
            pdf_path = base.with_suffix(".pdf")

            fig.savefig(png_path, dpi=600, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")

            print(f"Saved PNG: {png_path}")
            print(f"Saved PDF: {pdf_path}")

        if show:
            plt.show()

        return fig, axes, residual_data

    finally:
        plt.rcParams.update(old_rc)


# Overview figure of several catalogs, and common map limits for two of them

def add_scale_bar(ax, length_km, lon_min, lon_max, lat_min, lat_max):
    """
    Horizontal scale bar of length_km (km) near the lower-left corner of a lon/lat
    (degree) axis. Uses 111 km per degree times cos(central latitude) for longitude.
    """
    lat_center = 0.5 * (lat_min + lat_max)
    km_per_deg_lon = 111.0 * np.cos(np.deg2rad(lat_center))
    scale_deg_lon = length_km / km_per_deg_lon

    x0 = lon_min + 0.08 * (lon_max - lon_min)
    y0 = lat_min + 0.07 * (lat_max - lat_min)

    ax.plot([x0, x0 + scale_deg_lon], [y0, y0], color="black", lw=2.2, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.001, y0 + 0.001], color="black", lw=1.7)
    ax.plot([x0 + scale_deg_lon, x0 + scale_deg_lon], [y0 - 0.001, y0 + 0.001], color="black", lw=1.7)

    ax.text(
        x0 + 0.5 * scale_deg_lon,
        y0 + 0.012 * (lat_max - lat_min),
        f"{length_km:.0f} km",
        ha="center",
        va="bottom",
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5)
    )


def style_axis_full_box(ax, grid_y=False, grid_x=False):
    """Grey box on all four sides, outward ticks, optional light grid lines."""
    # full box
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.0)
        ax.spines[side].set_color("#444444")

    ax.tick_params(
        axis="both",
        direction="out",
        length=4.5,
        width=1.0,
        colors="#333333",
        labelsize=12
    )

    if grid_y:
        ax.grid(axis="y", color="0.85", linestyle="-", linewidth=0.6, alpha=0.7)
    if grid_x:
        ax.grid(axis="x", color="0.9", linestyle="-", linewidth=0.6, alpha=0.6)


def get_limits_multi(catalog_dict, pad_deg=0.015):
    """Lon/lat limits (degrees) covering every catalog in {key: DataFrame}, padded by pad_deg."""
    lon_all = []
    lat_all = []

    for _, df in catalog_dict.items():
        if len(df) == 0:
            continue
        lon_all.extend(df["longitude"].values.tolist())
        lat_all.extend(df["latitude"].values.tolist())

    return {
        "lon_min": min(lon_all) - pad_deg,
        "lon_max": max(lon_all) + pad_deg,
        "lat_min": min(lat_all) - pad_deg,
        "lat_max": max(lat_all) + pad_deg,
    }


def compute_counts_per_bin(df, time_col="time", freq="7D"):
    """Number of events per time bin (pandas frequency, e.g. "5D" = 5 days)."""
    if df.empty:
        return pd.Series(dtype=float)

    return (
        df.set_index(time_col)
          .resample(freq)
          .size()
    )


def plot_three_catalog_overview_pretty(
    catalog_plot_dict,
    configs,
    volcano,
    year,
    radius_km=20.0,
    time_bin="7D",
    max_depth_km=40.0,
    save_path=None,
    plot_order=("qq", "usgs", "avo"),
    show=True,
):
    """
    Publication-style multi-panel comparison for three catalogs:
    (a) event counts through time (bins of time_bin, e.g. "5D")
    (b) magnitude through time
    (c) map, with the volcano and a 10 km scale bar
    (d) longitude vs depth (km, down to max_depth_km)
    (e) latitude vs depth

    catalog_plot_dict: {key: DataFrame with time, latitude, longitude, depth_km, mag}.
    configs[key] needs "label", "color", "scatter_size" and "scatter_alpha".
    volcano: {"name", "lat", "lon"}. Only the keys in plot_order are drawn, in that
    order; any other key is ignored. year and radius_km are only used by the
    (commented-out) title and are kept for compatibility.
    Returns fig, {"a": ax, ..., "e": ax}.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt


    catalog_plot_dict = {k: catalog_plot_dict[k] for k in plot_order if k in catalog_plot_dict}

    lims = get_limits_multi(catalog_plot_dict, pad_deg=0.015)

    # reduced figsize so text appears larger
    fig = plt.figure(figsize=(12.0, 8.8))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[0.9, 1.2, 1.0],
        width_ratios=[1.05, 1.0],
        hspace=0.5,
        wspace=0.24
    )

    handles = []
    labels = []

    # -------------------------
    # (a) Event counts
    # -------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    for key, df in catalog_plot_dict.items():
        counts = compute_counts_per_bin(df, time_col="time", freq=time_bin)

        line, = ax1.plot(
            counts.index,
            counts.values,
            lw=2.4,
            color=configs[key]["color"],
            alpha=0.95,
            label=f"{configs[key]['label']} (n={len(df)})"
        )
        ax1.scatter(
            counts.index,
            counts.values,
            s=20,
            color=configs[key]["color"],
            alpha=0.95,
            edgecolors="none",
            zorder=3
        )

        handles.append(line)
        labels.append(f"{configs[key]['label']} (n={len(df)})")

    ax1.set_ylabel("Event count", fontsize=14)
    ax1.set_title("(a)", loc="left", fontweight="bold", fontsize=15)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.get_xticklabels(), rotation=20, ha="right")
    style_axis_full_box(ax1, grid_y=True, grid_x=False)

    # -------------------------
    # (b) Magnitude through time
    # -------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    for key, df in catalog_plot_dict.items():
        ax2.scatter(
            df["time"],
            df["mag"],
            s=(configs[key]["scatter_size"] ** 2) * 1.3,
            color=configs[key]["color"],
            alpha=min(configs[key]["scatter_alpha"], 0.60),
            edgecolors="none",
            rasterized=True
        )

    ax2.set_ylabel("Magnitude", fontsize=14)
    ax2.set_title("(b)", loc="left", fontweight="bold", fontsize=15)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.get_xticklabels(), rotation=20, ha="right")
    style_axis_full_box(ax2, grid_y=True, grid_x=False)

    # -------------------------
    # (c) Map view
    # -------------------------
    ax3 = fig.add_subplot(gs[1:, 0])

    for key, df in catalog_plot_dict.items():
        ax3.scatter(
            df["longitude"],
            df["latitude"],
            s=(configs[key]["scatter_size"] ** 2),
            color=configs[key]["color"],
            alpha=configs[key]["scatter_alpha"],
            edgecolors="none",
            rasterized=True
        )

    ax3.scatter(
        volcano["lon"],
        volcano["lat"],
        marker="*",
        s=260,
        color="gold",
        edgecolors="black",
        linewidth=1.0,
        zorder=10
    )

    ax3.text(
        volcano["lon"],
        volcano["lat"] + 0.015 * (lims["lat_max"] - lims["lat_min"]),
        volcano["name"],
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.5)
    )

    ax3.set_xlim(lims["lon_min"], lims["lon_max"])
    ax3.set_ylim(lims["lat_min"], lims["lat_max"])
    ax3.set_xlabel("Longitude", fontsize=14)
    ax3.set_ylabel("Latitude", fontsize=14)
    ax3.set_title("(c)", loc="left", fontweight="bold", fontsize=15)
    style_axis_full_box(ax3, grid_y=False, grid_x=False)

    add_scale_bar(
        ax3,
        length_km=10,
        lon_min=lims["lon_min"],
        lon_max=lims["lon_max"],
        lat_min=lims["lat_min"],
        lat_max=lims["lat_max"]
    )

    # -------------------------
    # (d) Longitude vs depth
    # -------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    for key, df in catalog_plot_dict.items():
        ax4.scatter(
            df["longitude"],
            df["depth_km"],
            s=(configs[key]["scatter_size"] ** 2),
            color=configs[key]["color"],
            alpha=configs[key]["scatter_alpha"],
            edgecolors="none",
            rasterized=True
        )

    ax4.set_xlim(lims["lon_min"], lims["lon_max"])
    ax4.set_ylim(max_depth_km, 0)
    ax4.set_xlabel("Longitude", fontsize=14)
    ax4.set_ylabel("Depth (km)", fontsize=14)
    ax4.set_title("(d)", loc="left", fontweight="bold", fontsize=15)
    style_axis_full_box(ax4, grid_y=True, grid_x=False)

    # -------------------------
    # (e) Latitude vs depth
    # -------------------------
    ax5 = fig.add_subplot(gs[2, 1])

    for key, df in catalog_plot_dict.items():
        ax5.scatter(
            df["latitude"],
            df["depth_km"],
            s=(configs[key]["scatter_size"] ** 2),
            color=configs[key]["color"],
            alpha=configs[key]["scatter_alpha"],
            edgecolors="none",
            rasterized=True
        )

    ax5.set_xlim(lims["lat_min"], lims["lat_max"])
    ax5.set_ylim(max_depth_km, 0)
    ax5.set_xlabel("Latitude", fontsize=14)
    ax5.set_ylabel("Depth (km)", fontsize=14)
    ax5.set_title("(e)", loc="left", fontweight="bold", fontsize=15)
    style_axis_full_box(ax5, grid_y=True, grid_x=False)

    # Global legend
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.6,
        handlelength=2.0,
        fontsize=13
    )

    # fig.suptitle(
    #     f"{volcano['name']} seismicity within {radius_km:.0f} km ({year})",
    #     y=0.995,
    #     fontsize=17,
    #     fontweight="bold"
    # )

    fig.subplots_adjust(top=0.89, left=0.08, right=0.985, bottom=0.12)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    return fig, {"a": ax1, "b": ax2, "c": ax3, "d": ax4, "e": ax5}


def get_common_bounds_for_two_catalogs(
    catalog1,
    catalog2,
    start,
    end,
    *,
    depth_filter_min=None,
    depth_filter_max=None,
    pad_lon=0.08,
    pad_lat=0.06,
    zoom_factor=1.8,
):
    """
    Common map limits for two catalogs (e.g. QuickQuake and USGS) so their zoomed maps
    can be drawn with the same extent: same padding (degrees), optional depth window
    (km) and zoom_factor logic as plot_mapview_seismicity_zoom.
    Returns xlim (lon_min, lon_max) and ylim (lat_min, lat_max) for its xlim / ylim.
    """

    q1 = make_quakes_for_map(catalog1, start, end).copy()
    q2 = make_quakes_for_map(catalog2, start, end).copy()

    q = pd.concat([q1, q2], ignore_index=True)

    if depth_filter_min is not None:
        q = q[q["depth"] >= float(depth_filter_min)].copy()

    if depth_filter_max is not None:
        q = q[q["depth"] <= float(depth_filter_max)].copy()

    if q.empty:
        raise ValueError("No events remain after applying the common filters.")

    minlon_data = float(q["lon"].min())
    maxlon_data = float(q["lon"].max())
    minlat_data = float(q["lat"].min())
    maxlat_data = float(q["lat"].max())

    minlon = minlon_data - float(pad_lon)
    maxlon = maxlon_data + float(pad_lon)
    minlat = minlat_data - float(pad_lat)
    maxlat = maxlat_data + float(pad_lat)

    if zoom_factor is not None:
        if zoom_factor <= 0:
            raise ValueError("zoom_factor must be > 0")

        clon = 0.5 * (minlon + maxlon)
        clat = 0.5 * (minlat + maxlat)

        half_width  = 0.5 * (maxlon - minlon) / float(zoom_factor)
        half_height = 0.5 * (maxlat - minlat) / float(zoom_factor)

        minlon = clon - half_width
        maxlon = clon + half_width
        minlat = clat - half_height
        maxlat = clat + half_height

    xlim = (minlon, maxlon)
    ylim = (minlat, maxlat)

    return xlim, ylim


# ============================================================
# DBSCAN CLUSTERS AND BEST-FITTING SURFACES (paper 2, planos0.2)
# ============================================================

def lonlat_to_local_km(lon, lat, lon0, lat0):
    """Approximate lon/lat -> local Cartesian x/y in km. x=east, y=north."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * 111.32
    return x, y


def local_km_to_lonlat(x, y, lon0, lat0):
    """Local Cartesian x/y km -> lon/lat."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lon = lon0 + x / (111.32 * np.cos(np.deg2rad(lat0)))
    lat = lat0 + y / 111.32
    return lon, lat


def dbscan_parameter_scan(points_xyz, eps_values, min_samples_values, min_cluster_size=100, verbose=True):
    """
    Run DBSCAN for every (eps, min_samples) pair on earthquake hypocentres.

    points_xyz: N x 3 array of longitude, latitude (degrees), depth (km).
    Coordinates are converted to local km around the mean lon/lat (lon_ref, lat_ref)
    and then standardised (each axis divided by its own standard deviation), so
    eps is in standard-deviation units, not km, and horizontal and vertical
    distances are weighted by the spread of the data along each axis.
    A cluster is "large" when it has >= min_cluster_size events.

    Returns {"results": [...], "lon_ref", "lat_ref", "X_km", "X_scaled"}; each
    result has eps, min_samples, n_clusters, n_large_clusters, n_noise, sizes
    (descending), silhouette (on non-noise events, NaN if undefined) and labels.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    points_xyz_all = np.asarray(points_xyz, dtype=float)

    lon_all = points_xyz_all[:, 0]
    lat_all = points_xyz_all[:, 1]
    z_all   = points_xyz_all[:, 2]

    lon_ref = float(lon_all.mean())
    lat_ref = float(lat_all.mean())

    x_km, y_km = lonlat_to_local_km(lon_all, lat_all, lon_ref, lat_ref)
    X_km = np.column_stack([x_km, y_km, z_all])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_km)

    results = []

    for min_samples in min_samples_values:
        for eps in eps_values:
            db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
            labels_tmp = db.fit_predict(X_scaled)

            unique_labels_tmp = [k for k in sorted(set(labels_tmp)) if k != -1]
            sizes_tmp = sorted([int(np.sum(labels_tmp == k)) for k in unique_labels_tmp], reverse=True)

            n_clusters = len(unique_labels_tmp)
            n_large_clusters = int(sum(s >= min_cluster_size for s in sizes_tmp))
            n_noise = int(np.sum(labels_tmp == -1))

            sil = np.nan
            valid_mask = labels_tmp != -1
            if n_clusters >= 2 and valid_mask.sum() > 10:
                try:
                    sil = float(silhouette_score(X_scaled[valid_mask], labels_tmp[valid_mask]))
                except Exception:
                    sil = np.nan

            results.append({
                "eps": float(eps),
                "min_samples": int(min_samples),
                "n_clusters": n_clusters,
                "n_large_clusters": n_large_clusters,
                "n_noise": n_noise,
                "sizes": sizes_tmp,
                "silhouette": sil,
                "labels": labels_tmp.copy(),
            })

    if verbose:
        print("\nDBSCAN scan results:")
        for r in results:
            if r["n_clusters"] > 0:
                print(
                    f"eps={r['eps']:.3f}, min_samples={r['min_samples']:2d} | "
                    f"clusters={r['n_clusters']:2d}, large={r['n_large_clusters']:2d}, "
                    f"noise={r['n_noise']:4d}, sizes={r['sizes'][:8]}, sil={r['silhouette']}"
                )

    return {"results": results, "lon_ref": lon_ref, "lat_ref": lat_ref, "X_km": X_km, "X_scaled": X_scaled}


def _dbscan_ranking_key(r):
    """
    Ranking rule:
      1. Prefer more large clusters.
      2. Prefer fewer total clusters, avoiding excessive fragmentation.
      3. Prefer higher silhouette.
      4. Prefer fewer noise points.
    """
    sil = r["silhouette"]
    if np.isnan(sil):
        sil = -999.0
    return (r["n_large_clusters"], -r["n_clusters"], sil, -r["n_noise"])


def select_best_dbscan(results, min_large_clusters=2, verbose=True):
    """
    Best DBSCAN solution among those with at least min_large_clusters large clusters,
    ranked by _dbscan_ranking_key (more large clusters, then fewer clusters, then
    higher silhouette, then less noise).
    """
    valid_results = [r for r in results if r["n_large_clusters"] >= min_large_clusters]

    if len(valid_results) == 0:
        raise ValueError(
            f"No DBSCAN solution found with at least {min_large_clusters} large clusters. "
            "Inspect the printed table and consider lowering min_cluster_size "
            "or changing eps_values / min_samples_values."
        )

    best = sorted(valid_results, key=_dbscan_ranking_key, reverse=True)[0]

    if verbose:
        print("\nSelected DBSCAN solution:")
        print(f"eps = {best['eps']}")
        print(f"min_samples = {best['min_samples']}")
        print(f"cluster sizes = {best['sizes']}")
        print(f"noise points = {best['n_noise']}")

    return best


def dbscan_cluster_table(catalog, labels, catc: Optional[CatalogCols] = None):
    """
    One row per DBSCAN cluster (noise, label -1, excluded): cluster_id, n_events,
    mean_lon, mean_lat, mean_depth_km; sorted by n_events (largest first).
    DBSCAN ids are not ordered by size.
    """
    catc = catc or CatalogCols()
    labels_db = np.asarray(labels)

    cluster_rows = []
    for cid in sorted([k for k in set(labels_db) if k != -1]):
        mask = labels_db == cid
        cluster_rows.append({
            "cluster_id": int(cid),
            "n_events": int(mask.sum()),
            "mean_lon": float(catalog.loc[mask, catc.lon].mean()),
            "mean_lat": float(catalog.loc[mask, catc.lat].mean()),
            "mean_depth_km": float(catalog.loc[mask, catc.depth_km].mean()),
        })

    return (
        pd.DataFrame(cluster_rows)
        .sort_values("n_events", ascending=False)
        .reset_index(drop=True)
    )


def select_cluster_ids(df_cluster_sizes, mode="top_n", manual_ids=None, top_n=5, min_events=100):
    """
    Cluster ids to fit and plot: mode "all", "top_n" (largest top_n), "manual"
    (manual_ids, checked to exist) or "min_events" (clusters with >= min_events).
    """
    available_ids = df_cluster_sizes["cluster_id"].astype(int).tolist()

    if mode == "all":
        selected_ids = available_ids

    elif mode == "top_n":
        selected_ids = (
            df_cluster_sizes
            .sort_values("n_events", ascending=False)
            .head(int(top_n))["cluster_id"]
            .astype(int)
            .tolist()
        )

    elif mode == "manual":
        if manual_ids is None:
            raise ValueError("manual_ids must be provided when mode='manual'.")

        selected_ids = [int(cid) for cid in manual_ids]
        missing = [cid for cid in selected_ids if cid not in available_ids]
        if missing:
            raise ValueError(
                f"These cluster IDs do not exist: {missing}. "
                f"Available cluster IDs are: {available_ids}"
            )

    elif mode == "min_events":
        selected_ids = (
            df_cluster_sizes
            .loc[df_cluster_sizes["n_events"] >= int(min_events), "cluster_id"]
            .astype(int)
            .tolist()
        )

    else:
        raise ValueError("Invalid mode. Use 'all', 'top_n', 'manual', or 'min_events'.")

    return selected_ids


def fit_best_surface_pca(
    points_xyz,
    lon_ref,
    lat_ref,
    degrees=(1, 2, 3),
    cv_splits=5,
    simplicity_tolerance=0.05,
    ridge_alpha=1.0e-3,
):
    """
    Fit and select a smooth surface for one DBSCAN cluster.

    Input coordinates:
        longitude, latitude, depth_km

    Procedure:
        1. Convert longitude/latitude to local Cartesian kilometers.
        2. Rotate the cluster into its principal coordinate system using PCA/SVD.
        3. Fit w = f(u, v) using polynomial surfaces of several degrees.
        4. Evaluate each candidate using cross-validation.
        5. Select the simplest model whose CV RMSE is within the specified
           tolerance of the best model.

    Degree interpretation:
        1 = plane
        2 = quadratic curved surface
        3 = cubic curved surface
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    points_xyz = np.asarray(points_xyz, dtype=float)
    points_xyz = points_xyz[np.isfinite(points_xyz).all(axis=1)]

    if points_xyz.shape[0] < 10:
        raise ValueError(
            f"At least 10 points are recommended for surface fitting; "
            f"received {points_xyz.shape[0]}."
        )

    lon = points_xyz[:, 0]
    lat = points_xyz[:, 1]
    depth = points_xyz[:, 2]

    x, y = lonlat_to_local_km(
        lon,
        lat,
        lon_ref,
        lat_ref,
    )

    P = np.column_stack([x, y, depth])

    centroid = P.mean(axis=0)
    Q = P - centroid

    # PCA coordinate system.
    _, singular_values, basis = np.linalg.svd(
        Q,
        full_matrices=False,
    )

    # Remove arbitrary sign ambiguity.
    if basis[2, 2] < 0:
        basis[2] *= -1

    # Rotated coordinates:
    # u and v span the dominant surface directions.
    # w measures displacement normal to the dominant orientation.
    UVW = Q @ basis.T

    UV = UVW[:, :2]
    w = UVW[:, 2]

    n_splits = min(int(cv_splits), len(UV))

    if n_splits < 3:
        raise ValueError("Not enough points for cross-validation.")

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    candidate_results = []

    for degree in degrees:
        model = make_pipeline(
            PolynomialFeatures(
                degree=int(degree),
                include_bias=False,
            ),
            StandardScaler(),
            Ridge(alpha=float(ridge_alpha)),
        )

        negative_mse = cross_val_score(
            model,
            UV,
            w,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )

        fold_rmse = np.sqrt(-negative_mse)
        cv_rmse = float(np.mean(fold_rmse))
        cv_rmse_std = float(np.std(fold_rmse))

        model.fit(UV, w)

        predicted_w = model.predict(UV)
        residuals = w - predicted_w

        train_rmse = float(
            np.sqrt(np.mean(residuals**2))
        )

        candidate_results.append({
            "degree": int(degree),
            "model": model,
            "cv_rmse_km": cv_rmse,
            "cv_rmse_std_km": cv_rmse_std,
            "train_rmse_km": train_rmse,
        })

    best_rmse = min(
        result["cv_rmse_km"]
        for result in candidate_results
    )

    # Prefer a simpler model if its error is close to the best result.
    acceptable_limit = best_rmse * (
        1.0 + float(simplicity_tolerance)
    )

    acceptable_models = [
        result
        for result in candidate_results
        if result["cv_rmse_km"] <= acceptable_limit
    ]

    selected = min(
        acceptable_models,
        key=lambda result: result["degree"],
    )

    return {
        "P": P,
        "centroid": centroid,
        "basis": basis,
        "UV": UV,
        "w": w,
        "singular_values": singular_values,
        "model": selected["model"],
        "selected_degree": selected["degree"],
        "cv_rmse_km": selected["cv_rmse_km"],
        "cv_rmse_std_km": selected["cv_rmse_std_km"],
        "train_rmse_km": selected["train_rmse_km"],
        "candidate_results": candidate_results,
        "lon_ref": float(lon_ref),
        "lat_ref": float(lat_ref),
    }


def make_fitted_surface_mesh(
    fit,
    grid_size=45,
):
    """
    Generate a triangular mesh of the selected fitted surface.

    The mesh is restricted to the convex hull of the cluster in local
    surface coordinates.
    """
    from scipy.spatial import Delaunay

    UV = fit["UV"]
    model = fit["model"]

    umin = float(UV[:, 0].min())
    umax = float(UV[:, 0].max())
    vmin = float(UV[:, 1].min())
    vmax = float(UV[:, 1].max())

    u_values = np.linspace(umin, umax, int(grid_size))
    v_values = np.linspace(vmin, vmax, int(grid_size))

    U_grid, V_grid = np.meshgrid(
        u_values,
        v_values,
    )

    UV_grid = np.column_stack([
        U_grid.ravel(),
        V_grid.ravel(),
    ])

    # Keep only grid points inside the cluster convex hull.
    cluster_hull = Delaunay(UV)
    inside = cluster_hull.find_simplex(UV_grid) >= 0

    UV_inside = UV_grid[inside]

    if UV_inside.shape[0] < 3:
        raise ValueError(
            "Not enough grid points inside the cluster hull."
        )

    predicted_w = model.predict(UV_inside)

    UVW_surface = np.column_stack([
        UV_inside,
        predicted_w,
    ])

    # Transform from PCA coordinates back to x/y/depth coordinates.
    P_surface = (
        fit["centroid"]
        + UVW_surface @ fit["basis"]
    )

    x_surface = P_surface[:, 0]
    y_surface = P_surface[:, 1]
    depth_surface = P_surface[:, 2]

    lon_surface, lat_surface = local_km_to_lonlat(
        x_surface,
        y_surface,
        fit["lon_ref"],
        fit["lat_ref"],
    )

    triangulation = Delaunay(UV_inside)

    return {
        "lon": np.asarray(lon_surface),
        "lat": np.asarray(lat_surface),
        "depth": np.asarray(depth_surface),
        "uv": UV_inside,
        "triangles": triangulation.simplices,
    }


def plane_dip_deg(normal):
    """Plane dip in degrees from horizontal: 0 horizontal, 90 vertical."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return float(np.degrees(np.arccos(np.abs(n[2]))))


def plane_dip_direction_deg(normal):
    """
    Dip direction as azimuth clockwise from North.

    x = East, y = North, z = depth positive downward.
    Down-dip direction = horizontal projection of -normal.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)

    east  = -n[0]
    north = -n[1]

    az = (np.degrees(np.arctan2(east, north)) + 360.0) % 360.0
    return float(az)


def surface_fit_orientation(fit):
    """
    Dip and dip direction (degrees) of the best-fitting PLANE through a cluster, from a
    fit_best_surface_pca result: the plane normal is the third PCA axis (smallest
    spread, made to point downward). This is the total-least-squares plane used in
    planos0.1, computed here on the same cluster as the selected surface; it does not
    describe the curvature of a degree 2 or 3 surface.
    Returns {"dip_deg", "dip_direction_deg", "normal_x", "normal_y", "normal_z",
    "planarity_ratio_S3_S2", "linearity_ratio_S2_S1"} (S = singular values).
    """
    n = np.asarray(fit["basis"][2], dtype=float)
    s = np.asarray(fit["singular_values"], dtype=float)
    return {
        "dip_deg": plane_dip_deg(n),
        "dip_direction_deg": plane_dip_direction_deg(n),
        "normal_x": float(n[0]),
        "normal_y": float(n[1]),
        "normal_z": float(n[2]),
        "planarity_ratio_S3_S2": float(s[2] / s[1]),
        "linearity_ratio_S2_S1": float(s[1] / s[0]),
    }


def fit_cluster_surfaces(
    df_clusters,
    cluster_ids,
    lon_ref,
    lat_ref,
    *,
    label_col="cluster_dbscan",
    catc: Optional[CatalogCols] = None,
    degrees=(1, 2, 3),
    cv_splits=5,
    simplicity_tolerance=0.05,
    ridge_alpha=1.0e-3,
    grid_size=45,
    verbose=True,
):
    """
    Fit the best surface (fit_best_surface_pca) and its mesh for each selected cluster.

    df_clusters: catalog with the DBSCAN label of each event in label_col.
    Clusters with fewer than 10 events are skipped.
    Returns (surface_fits, surface_meshes, cluster_points, df_surface_fits):
    dicts keyed by cluster id, and one table row per fitted cluster with
    n_events, selected_degree, cv_rmse_km, cv_rmse_std_km, train_rmse_km plus
    the dip and dip direction of its best-fitting plane (surface_fit_orientation).
    """
    catc = catc or CatalogCols()

    surface_fits = {}
    surface_meshes = {}
    cluster_points = {}
    surface_fit_rows = []

    for cid in cluster_ids:
        cid = int(cid)

        mask = (
            df_clusters[label_col].to_numpy()
            == cid
        )

        points = (
            df_clusters.loc[
                mask,
                [catc.lon, catc.lat, catc.depth_km],
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        cluster_points[cid] = points

        if points.shape[0] < 10:
            print(
                f"Skipping cluster {cid}: "
                f"not enough points ({points.shape[0]})."
            )
            continue

        fit = fit_best_surface_pca(
            points_xyz=points,
            lon_ref=lon_ref,
            lat_ref=lat_ref,
            degrees=degrees,
            cv_splits=cv_splits,
            simplicity_tolerance=simplicity_tolerance,
            ridge_alpha=ridge_alpha,
        )

        mesh = make_fitted_surface_mesh(
            fit,
            grid_size=grid_size,
        )

        surface_fits[cid] = fit
        surface_meshes[cid] = mesh

        surface_fit_rows.append({
            "cluster_id": cid,
            "n_events": int(points.shape[0]),
            "selected_degree": fit["selected_degree"],
            "cv_rmse_km": fit["cv_rmse_km"],
            "cv_rmse_std_km": fit["cv_rmse_std_km"],
            "train_rmse_km": fit["train_rmse_km"],
            **surface_fit_orientation(fit),
        })

        if verbose:
            print(f"\nCluster {cid}")
            for candidate in fit["candidate_results"]:
                print(
                    f"  Degree {candidate['degree']}: "
                    f"CV RMSE = {candidate['cv_rmse_km']:.3f} km, "
                    f"training RMSE = "
                    f"{candidate['train_rmse_km']:.3f} km"
                )
            print(
                f"  Selected degree: "
                f"{fit['selected_degree']}"
            )

    df_surface_fits = pd.DataFrame(surface_fit_rows)

    if df_surface_fits.empty:
        raise RuntimeError(
            "No surfaces were fitted. "
            "Check the selected DBSCAN clusters."
        )

    df_surface_fits = (
        df_surface_fits
        .sort_values("n_events", ascending=False)
        .reset_index(drop=True)
    )

    return surface_fits, surface_meshes, cluster_points, df_surface_fits


CLUSTER_STYLES = [
    {"color": "red",     "edgecolor": "darkred"},
    {"color": "blue",    "edgecolor": "navy"},
    {"color": "orange",  "edgecolor": "darkorange"},
    {"color": "magenta", "edgecolor": "purple"},
    {"color": "gold",    "edgecolor": "goldenrod"},
    {"color": "cyan",    "edgecolor": "teal"},
    {"color": "lime",    "edgecolor": "green"},
    {"color": "violet",  "edgecolor": "indigo"},
    {"color": "brown",   "edgecolor": "saddlebrown"},
    {"color": "pink",    "edgecolor": "deeppink"},
]


def cluster_style_map(cluster_ids, styles=None):
    """{cluster id: {"color", "edgecolor"}}, cycling through CLUSTER_STYLES in the given order."""
    styles = styles or CLUSTER_STYLES
    return {cid: styles[i % len(styles)] for i, cid in enumerate(cluster_ids)}


_SURFACE_MODEL_NAMES = {
    1: "Plane",
    2: "Quadratic",
    3: "Cubic",
}


def plot_cluster_surfaces_3d(
    df_clusters,
    cluster_ids,
    cluster_points,
    surface_meshes,
    start,
    end,
    *,
    catc: Optional[CatalogCols] = None,
    cluster_styles=None,
    df_surface_fits=None,
    depth_max=10.0,
    exagg=2.0,
    view_elev=5.0,
    view_azim=265.0,
    other_hollow_black=True,
    surface_labels=False,
    crop_deg=(0.10, 0.03, 0.05, 0.02),
    focal_length=0.35,
    dem_res="15s",
    figsize=(12, 9),
):
    """
    Static 3-D view (topography + seismicity) of the selected DBSCAN clusters and
    their fitted surfaces, as in paper 2 (planos0.2).

    All events are drawn hollow (black edges if other_hollow_black), each selected
    cluster in its colour, and each fitted surface as a translucent triangular mesh.
    crop_deg = (west, east, south, north) degrees trimmed from the plot_mapview3d_groups
    bounds. surface_labels=True writes "C<id> / model / CV RMSE" at each cluster
    centre (needs df_surface_fits).
    Returns fig, ax, meta (meta from plot_mapview3d_groups).
    """
    catc = catc or CatalogCols()
    cluster_styles = cluster_styles or cluster_style_map(cluster_ids)

    fig, ax, meta = plot_mapview3d_groups(
        catalog=df_clusters,
        start=start,
        end=end,
        depth_max=depth_max,
        dem_res=dem_res,
        exagg=exagg,
        topo_cmap="gist_earth_r",
        size_k=10.0,
        topo_alpha=0.35,
        topo_stride=5,
        view_elev=view_elev,
        view_azim=view_azim,
        n_xticks=5,
        tick_fontsize=13,
        figsize=figsize,
        savefig=False,
        out_png=None,
        dpi=800,
        catc=catc,
        hollow_quakes=True,
        quake_edge_lw=0.05,
        group1_xyz=None,
        group2_xyz=None,
        other_edgecolor=(
            "black"
            if other_hollow_black
            else "lightgray"
        ),
        plane_polys=None,
    )

    # DBSCAN cluster points
    for cid in cluster_ids:
        cid = int(cid)

        points = cluster_points.get(cid)

        if points is None or points.shape[0] == 0:
            continue

        style = cluster_styles[cid]

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=5,
            c=style["color"],
            edgecolors=style["edgecolor"],
            linewidths=0.10,
            depthshade=False,
            alpha=0.25,
            label=f"Cluster {cid}",
        )

    # Fitted surfaces
    for cid in cluster_ids:
        cid = int(cid)

        if cid not in surface_meshes:
            continue

        mesh = surface_meshes[cid]
        style = cluster_styles[cid]

        ax.plot_trisurf(
            mesh["lon"],
            mesh["lat"],
            mesh["depth"],
            triangles=mesh["triangles"],
            color=style["color"],
            edgecolor=style["edgecolor"],
            linewidth=0.15,
            alpha=0.45,
            shade=False,
        )

    # Optional surface labels
    if surface_labels and df_surface_fits is not None:
        for _, row in df_surface_fits.iterrows():
            cid = int(row["cluster_id"])

            points = cluster_points.get(cid)

            if points is None or points.shape[0] == 0:
                continue

            lon_c = float(np.mean(points[:, 0]))
            lat_c = float(np.mean(points[:, 1]))
            depth_c = float(np.mean(points[:, 2]))

            degree = int(row["selected_degree"])
            model_name = _SURFACE_MODEL_NAMES.get(
                degree,
                f"Degree {degree}",
            )

            label = (
                f"C{cid}\n"
                f"{model_name}\n"
                f"CV RMSE={row['cv_rmse_km']:.2f} km"
            )

            ax.text(
                lon_c,
                lat_c,
                depth_c,
                label,
                fontsize=8,
                ha="center",
                va="center",
                zorder=100,
            )

    # Figure format
    minlon, maxlon, minlat, maxlat = meta["bounds"]
    crop_w, crop_e, crop_s, crop_n = crop_deg

    ax.set_xlim(minlon + crop_w, maxlon - crop_e)
    ax.set_ylim(minlat + crop_s, maxlat - crop_n)

    ax.set_proj_type(
        "persp",
        focal_length=focal_length,
    )

    ax.set_xlabel(
        "Longitude",
        labelpad=12,
        fontsize=13,
    )

    ax.set_ylabel(
        "Latitude",
        labelpad=12,
        fontsize=13,
    )

    ax.zaxis.set_rotate_label(False)

    ax.set_zlabel(
        "Depth (km)",
        rotation=90,
        labelpad=18,
        fontsize=13,
    )

    return fig, ax, meta


def plot_cluster_surfaces_interactive(
    cluster_ids,
    cluster_points,
    surface_meshes,
    surface_fits,
    df_surface_fits,
    lon_ref,
    lat_ref,
    *,
    cluster_styles=None,
):
    """
    Interactive, rotatable Plotly 3-D view of the selected clusters and fitted
    surfaces, in local km (East, North, Depth) with the same physical scale on
    the three axes. Returns the plotly Figure (call .show() or .write_html()).
    """
    import plotly.graph_objects as go

    cluster_styles = cluster_styles or cluster_style_map(cluster_ids)

    model_names = {
        1: "Plane",
        2: "Quadratic surface",
        3: "Cubic surface",
    }

    fig_interactive = go.Figure()

    for cid in cluster_ids:
        cid = int(cid)

        points = cluster_points.get(cid)

        if points is None or points.shape[0] == 0:
            continue

        style = cluster_styles[cid]
        cluster_color = style["color"]

        # Local Cartesian km keep x, y and depth in consistent physical units.
        x_points, y_points = lonlat_to_local_km(
            points[:, 0],
            points[:, 1],
            lon_ref,
            lat_ref,
        )

        model_row = df_surface_fits.loc[
            df_surface_fits["cluster_id"] == cid
        ]

        if not model_row.empty:
            model_row = model_row.iloc[0]

            degree = int(model_row["selected_degree"])
            model_name = model_names.get(
                degree,
                f"Polynomial degree {degree}",
            )

            cv_rmse = float(model_row["cv_rmse_km"])
        else:
            degree = None
            model_name = "Unknown surface"
            cv_rmse = np.nan

        # Earthquake points
        fig_interactive.add_trace(
            go.Scatter3d(
                x=x_points,
                y=y_points,
                z=points[:, 2],
                mode="markers",
                name=f"Cluster {cid} events",
                legendgroup=f"cluster_{cid}",
                marker=dict(
                    size=2.5,
                    color=cluster_color,
                    opacity=0.40,
                ),
                customdata=np.column_stack([
                    points[:, 0],
                    points[:, 1],
                ]),
                hovertemplate=(
                    f"<b>Cluster {cid}</b><br>"
                    "East: %{x:.2f} km<br>"
                    "North: %{y:.2f} km<br>"
                    "Depth: %{z:.2f} km<br>"
                    "Longitude: %{customdata[0]:.5f}<br>"
                    "Latitude: %{customdata[1]:.5f}"
                    "<extra></extra>"
                ),
            )
        )

        # Fitted finite surface
        if cid not in surface_meshes:
            continue

        mesh = surface_meshes[cid]
        triangles = np.asarray(mesh["triangles"])

        x_surface, y_surface = lonlat_to_local_km(
            mesh["lon"],
            mesh["lat"],
            lon_ref,
            lat_ref,
        )

        fig_interactive.add_trace(
            go.Mesh3d(
                x=x_surface,
                y=y_surface,
                z=mesh["depth"],
                i=triangles[:, 0],
                j=triangles[:, 1],
                k=triangles[:, 2],
                name=f"Cluster {cid}: {model_name}",
                legendgroup=f"cluster_{cid}",
                color=cluster_color,
                opacity=0.50,
                flatshading=False,
                lighting=dict(
                    ambient=0.55,
                    diffuse=0.85,
                    specular=0.25,
                    roughness=0.60,
                    fresnel=0.10,
                ),
                lightposition=dict(
                    x=100,
                    y=100,
                    z=-100,
                ),
                hovertemplate=(
                    f"<b>Cluster {cid}</b><br>"
                    f"Model: {model_name}<br>"
                    f"CV RMSE: {cv_rmse:.3f} km<br>"
                    "East: %{x:.2f} km<br>"
                    "North: %{y:.2f} km<br>"
                    "Depth: %{z:.2f} km"
                    "<extra></extra>"
                ),
            )
        )

        # Surface label at its centroid
        if cid in surface_fits:
            centroid = surface_fits[cid]["centroid"]

            fig_interactive.add_trace(
                go.Scatter3d(
                    x=[centroid[0]],
                    y=[centroid[1]],
                    z=[centroid[2]],
                    mode="text",
                    text=[
                        (
                            f"C{cid}<br>"
                            f"{model_name}<br>"
                            f"RMSE = {cv_rmse:.2f} km"
                        )
                    ],
                    textposition="middle center",
                    textfont=dict(
                        size=12,
                        color="black",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig_interactive.update_layout(
        title=(
            "Interactive 3D View of DBSCAN Clusters "
            "and Best-Fitting Surfaces"
        ),
        width=1050,
        height=850,
        margin=dict(
            l=10,
            r=10,
            b=10,
            t=60,
        ),
        legend=dict(
            title="Clusters and models",
            x=0.01,
            y=0.99,
        ),
        scene=dict(
            xaxis=dict(
                title="East (km)",
                showbackground=True,
            ),
            yaxis=dict(
                title="North (km)",
                showbackground=True,
            ),
            zaxis=dict(
                title="Depth (km)",
                autorange="reversed",
                showbackground=True,
            ),

            # Same physical scale for East, North, and depth.
            aspectmode="data",

            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.6,
                    z=0.8,
                )
            ),
        ),
    )

    return fig_interactive


# ============================================================
# ANIMATION
# ============================================================

def animate_mapview3d_timecolor(
    catalog: pd.DataFrame,
    *,
    start,
    end,
    out_gif: str = "time_evolution.gif",
    duration_s: float = 15.0,
    fps: int = 20,
    depth_max: float = 40.0,
    dem_res: str = "15s",
    exagg: float = 4.0,
    topo_cmap: str = "gist_earth_r",
    quake_cmap: str = "magma",
    size_k: float = 15.0,
    marker_min: float = 6.0,
    topo_alpha: float = 0.95,
    topo_stride: int = 5,
    view_elev: float = 18.0,
    view_azim: float = 235.0,
    n_xticks: int = 5,
    n_yticks: int = 5,
    tick_fontsize: int = 13,
    cbar_tick_fontsize: int = 18,
    figsize: Tuple[float, float] = (12, 9),
    dpi: int = 200,
    label_fmt: str = "Elapsed: {days:.0f} days",
    box_aspect: Tuple[float, float, float] = (1.35, 1.0, 1.0),
    proj_type: str = "persp",
    catc: Optional["CatalogCols"] = None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    from matplotlib.colors import Normalize
    from matplotlib.animation import FuncAnimation, PillowWriter

    if catc is None:
        catc = CatalogCols()

    df = normalize_catalog(catalog, catc=catc)
    df = df.dropna(subset=[catc.time, catc.lat, catc.lon]).copy()

    df["_t"] = pd.to_datetime(df[catc.time], utc=True, errors="coerce")
    df = df.dropna(subset=["_t"]).copy()

    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    df = df[(df["_t"] >= start_dt) & (df["_t"] <= end_dt)].copy()

    depth_col = catc.depth_km if catc.depth_km in df.columns else None
    if depth_col is None:
        df["depth_km"] = 0.0
        depth_col = "depth_km"
    else:
        df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce").fillna(0.0)

    df = df[df[depth_col] <= float(depth_max)].copy()

    mag_col = catc.mag if catc.mag in df.columns else None
    if mag_col is None:
        df["magnitude"] = 1.0
        mag_col = "magnitude"
    else:
        df[mag_col] = pd.to_numeric(df[mag_col], errors="coerce").fillna(1.0)

    df = df.sort_values("_t").reset_index(drop=True)

    if df.empty:
        raise ValueError("No events after filtering.")

    t_num = mdates.date2num(df["_t"].to_numpy(dtype="datetime64[ns]"))

    minlon = float(df[catc.lon].min())
    maxlon = float(df[catc.lon].max())
    minlat = float(df[catc.lat].min())
    maxlat = float(df[catc.lat].max())
    region = [minlon, maxlon, minlat, maxlat]

    lon, lat, elev_m = load_earth_relief(dem_res, region=region)   # same grid as pygmt, without GMT
    elev_km = elev_m / 1000.0

    Lon, Lat = np.meshgrid(lon, lat)
    Z = -elev_km * float(exagg)

    x = df[catc.lon].astype(float).to_numpy()
    y = df[catc.lat].astype(float).to_numpy()
    z = df[depth_col].astype(float).to_numpy()

    mags = df[mag_col].to_numpy(dtype=float)
    sizes = float(size_k) * mags
    sizes = np.where(np.isfinite(sizes), sizes, float(marker_min))
    sizes = np.clip(sizes, float(marker_min), None)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.92)

    ax.plot_surface(
        Lon, Lat, Z,
        cmap=topo_cmap,
        rstride=int(topo_stride),
        cstride=int(topo_stride),
        linewidth=0,
        antialiased=True,
        alpha=float(topo_alpha),
    )

    norm = Normalize(vmin=float(t_num.min()), vmax=float(t_num.max()))

    sc = ax.scatter(
        [], [], [],
        c=[],
        cmap=quake_cmap,
        norm=norm,
        s=[],
        edgecolor="k",
        linewidth=0.2,
        alpha=1.0,
        depthshade=False,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=quake_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.03, shrink=0.6, aspect=25)
    cbar.ax.tick_params(labelsize=int(cbar_tick_fontsize))
    cbar.set_label("Time", fontsize=int(cbar_tick_fontsize))

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    cbar.ax.yaxis.set_major_locator(locator)
    cbar.ax.yaxis.set_major_formatter(formatter)

    xticks = np.linspace(minlon, maxlon, int(n_xticks))
    yticks = np.linspace(minlat, maxlat, int(n_yticks))
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{v:.1f}" for v in xticks])
    ax.set_yticklabels([f"{v:.1f}" for v in yticks])

    ax.tick_params(axis="x", labelsize=int(tick_fontsize))
    ax.tick_params(axis="y", labelsize=int(tick_fontsize))
    ax.tick_params(axis="z", labelsize=int(tick_fontsize))

    ax.set_xlabel("Longitude", labelpad=12)
    ax.set_ylabel("Latitude", labelpad=12)
    ax.set_zlabel("Depth (km)", labelpad=10)

    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)
    ax.set_zlim(float(np.nanmin(Z)), float(depth_max))
    ax.invert_zaxis()

    # Visual scale for presentation, not the true physical scale.
    # It keeps the figure from looking flattened or distorted.
    ax.set_box_aspect(box_aspect)

    ax.view_init(elev=float(view_elev), azim=float(view_azim))
    ax.set_proj_type(proj_type)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
        except Exception:
            pass
        axis._axinfo["grid"]["color"] = (0.45, 0.45, 0.45, 0.8)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["grid"]["linestyle"] = "--"

    label = ax.text2D(
        0.5, 0.98,
        label_fmt.format(days=0.0),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", alpha=0.9),
    )

    n_frames = max(2, int(round(float(duration_s) * int(fps))))
    frame_times = np.linspace(t_num[0], t_num[-1], n_frames)

    def update(i):
        k = np.searchsorted(t_num, frame_times[i], side="right")
        sc._offsets3d = (x[:k], y[:k], z[:k])
        sc.set_sizes(sizes[:k])
        sc.set_array(t_num[:k])
        label.set_text(label_fmt.format(days=frame_times[i] - t_num[0]))
        return sc, label

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )

    anim.save(out_gif, writer=PillowWriter(fps=fps), dpi=int(dpi))
    plt.close(fig)

    return out_gif


# ============================================================
# NOTEBOOK UTILITIES MOVED OUT OF THE NOTEBOOK
# ============================================================


def plot_event_count_time(
    catalog: pd.DataFrame,
    *,
    time_col: str = "time",
    start=None,
    end=None,
    freq: str = "D",
    figsize: Tuple[float, float] = (12, 5),
    color: str = "black",
    linewidth: float = 1.5,
    marker=None,
    xlabel: str = "Time",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    grid: bool = True,
    savefig: bool = False,
    out_png: Optional[str] = None,
    dpi: int = 600,
):
    """
    Plot event counts through time.

    freq examples:
    - "D": daily
    - "W": weekly
    - "ME" or "M": monthly, depending on pandas version
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = catalog.copy()

    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' does not exist in the catalog.")

    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).copy()

    if start is not None:
        start_dt = pd.to_datetime(start, utc=True, errors="coerce")
        df = df[df[time_col] >= start_dt]

    if end is not None:
        end_dt = pd.to_datetime(end, utc=True, errors="coerce")
        df = df[df[time_col] <= end_dt]

    if df.empty:
        raise ValueError("No events within the selected time range.")

    counts = df.set_index(time_col).sort_index().resample(freq).size()

    if ylabel is None:
        label_by_freq = {"D": "Events per day", "W": "Events per week", "M": "Events per month", "ME": "Events per month"}
        ylabel = label_by_freq.get(freq, f"Events per {freq}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        counts.index,
        counts.values,
        color=color,
        linewidth=float(linewidth),
        marker=marker,
        alpha=1.0,
    )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    if title is not None:
        ax.set_title(title)

    ax.tick_params(axis="both", labelsize=16)
    if grid:
        ax.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    plt.tight_layout()

    if savefig:
        if out_png is None:
            out_png = f"event_count_{freq}.png"
        fig.savefig(out_png, dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    return fig, ax, counts


# Backward-compatible name used in the old notebook.
plot_event_count_time_v2 = plot_event_count_time


def plot_regional_location_map(
    *,
    volcano_lon: float,
    volcano_lat: float,
    region_extent: Tuple[float, float, float, float] = (-179, -145, 51, 66),
    figsize: Tuple[float, float] = (10, 10),
    land_facecolor: str = "#6da86d",
    ocean_color: str = "lightblue",
    coast_scale: str = "l",
    marker_color: str = "red",
    marker_size: float = 110,
    marker_style: str = "^",
    label: str = "",
    label_fontsize: int = 15,
    label_dlon: float = 0.0,
    label_dlat: float = -1.2,
    gridline_color: str = "gray",
    gridline_alpha: float = 0.5,
    gridline_style: str = "",
    gridline_width: float = 0.5,
    savefig: bool = False,
    out_png: str = "regional_location_map.png",
    dpi: int = 300,
    show: bool = False,
):
    """
    Regional location map for a volcano/catalog target.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker

    minlon, maxlon, minlat, maxlat = region_extent

    proj = ccrs.LambertAzimuthalEqualArea(
        central_longitude=np.mean([minlon, maxlon]),
        central_latitude=np.mean([minlat, maxlat]),
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw=dict(projection=proj),
        facecolor="w",
    )

    ax.set_extent(region_extent, crs=ccrs.PlateCarree())

    coast_regional = cfeature.GSHHSFeature(
        scale=coast_scale,
        edgecolor="grey",
        facecolor=land_facecolor,
    )

    ax.add_feature(cfeature.OCEAN, color=ocean_color, zorder=0)
    ax.add_feature(coast_regional, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="black", zorder=2)

    ax.plot(
        volcano_lon,
        volcano_lat,
        marker_style,
        color=marker_color,
        markersize=np.sqrt(marker_size),
        markeredgecolor="black",
        markeredgewidth=0.8,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )

    if label:
        ax.text(
            volcano_lon + label_dlon,
            volcano_lat + label_dlat,
            label,
            ha="center",
            va="center",
            fontsize=int(label_fontsize),
            transform=ccrs.PlateCarree(),
            zorder=6,
        )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        x_inline=False,
        y_inline=False,
        color=gridline_color,
        alpha=float(gridline_alpha),
        linestyle=gridline_style,
        linewidth=float(gridline_width),
    )
    gl.xlocator = mticker.MaxNLocator(5)
    gl.ylocator = mticker.MaxNLocator(5)

    if savefig:
        fig.savefig(out_png, format=Path(out_png).suffix.lstrip(".") or "png", dpi=int(dpi), bbox_inches="tight")
        print(f"[ok] saved: {out_png}")

    if show:
        plt.show()

    return fig, ax


def plot_aniakchak_alaska_location(**kwargs):
    """
    Backward-compatible wrapper for the old notebook function name.
    """
    kwargs.setdefault("volcano_lon", -158.17)
    kwargs.setdefault("volcano_lat", 56.88)
    kwargs.setdefault("label", "Aniakchak")
    kwargs.setdefault("out_png", "aniakchak_location.png")
    return plot_regional_location_map(**kwargs)


def draw_3d_box_edges(ax, color: str = "k", lw: float = 1.0, alpha: float = 0.95):
    """
    Draw the 12 edges of a Matplotlib 3D axes box using the current axis limits.
    """
    x0, x1 = ax.get_xlim3d()
    y0, y1 = ax.get_ylim3d()
    z0, z1 = ax.get_zlim3d()

    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for i, j in edges:
        xi, yi, zi = corners[i]
        xj, yj, zj = corners[j]
        ax.plot([xi, xj], [yi, yj], [zi, zj], color=color, lw=lw, alpha=alpha, zorder=1000)

    return ax


def style_3d_axes(
    ax,
    *,
    xlabel: str = "Longitude",
    ylabel: str = "Latitude",
    zlabel: str = "Depth (km)",
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    box_edge_color: str = "k",
    box_edge_lw: float = 1.0,
    show_view_text: bool = True,
    view_elev: Optional[float] = None,
    view_azim: Optional[float] = None,
):
    """
    Apply a consistent clean style to Matplotlib 3D axes.
    """
    ax.set_xlabel(xlabel, fontsize=int(label_fontsize), labelpad=14)
    ax.set_ylabel(ylabel, fontsize=int(label_fontsize), labelpad=14)
    ax.set_zlabel(zlabel, fontsize=int(label_fontsize), labelpad=10)

    ax.tick_params(axis="x", labelsize=int(tick_fontsize))
    ax.tick_params(axis="y", labelsize=int(tick_fontsize))
    ax.tick_params(axis="z", labelsize=int(tick_fontsize))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((0, 0, 0, 0))
        except Exception:
            pass

        axis._axinfo["grid"]["color"] = (0.45, 0.45, 0.45, 0.8)
        axis._axinfo["grid"]["linewidth"] = 0.5
        axis._axinfo["grid"]["linestyle"] = "--"

    draw_3d_box_edges(ax, color=box_edge_color, lw=float(box_edge_lw), alpha=0.95)

    if show_view_text and (view_elev is not None) and (view_azim is not None):
        ax.text2D(
            0.02, 0.98,
            f"View: elev={float(view_elev):.1f}°, azim={float(view_azim):.1f}°",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=int(label_fontsize) - 1,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.4", alpha=0.85),
        )

    return ax



# ============================================================
# PICKS / WAVEFORM HELPERS
# ============================================================

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


def normalize_stations_for_waveforms(
    stations: pd.DataFrame,
    stc: StationCols = StationCols(),
) -> pd.DataFrame:
    """
    Normalize station metadata for waveform/event-pick plots.

    Returns columns compatible with StationCols:
    station, latitude, longitude, elev_km.

    This accepts both:
    - the cleaned helper format: station/latitude/longitude/elev_km
    - the older waveform helper format: id/lat/lon/elev_km
    - raw GAMMA_station_list.json columns such as elevation(m)
    """
    df = stations.copy()

    rename_map = {}
    if stc.station_id not in df.columns:
        src = _first_existing_column(df, [stc.station_id, "id", "station_id", "station", "sta", "sid"])
        if src is not None:
            rename_map[src] = stc.station_id

    if stc.lat not in df.columns:
        src = _first_existing_column(df, [stc.lat, "lat", "latitude", "Lat", "LAT", "y"])
        if src is not None:
            rename_map[src] = stc.lat

    if stc.lon not in df.columns:
        src = _first_existing_column(df, [stc.lon, "lon", "longitude", "Lon", "LON", "x"])
        if src is not None:
            rename_map[src] = stc.lon

    if stc.elev_km not in df.columns:
        src = _first_existing_column(df, [stc.elev_km, "elevation_km", "elevation(m)", "elevation_m", "elev", "elevation"])
        if src is not None:
            rename_map[src] = stc.elev_km

    if rename_map:
        df = df.rename(columns=rename_map)

    _require_cols(df, [stc.station_id, stc.lat, stc.lon], name="stations")

    df[stc.lat] = pd.to_numeric(df[stc.lat], errors="coerce")
    df[stc.lon] = pd.to_numeric(df[stc.lon], errors="coerce")

    if stc.elev_km in df.columns:
        df[stc.elev_km] = pd.to_numeric(df[stc.elev_km], errors="coerce").fillna(0.0)
        if "elevation(m)" in stations.columns or "elevation_m" in stations.columns:
            df[stc.elev_km] = df[stc.elev_km] / 1000.0
    else:
        df[stc.elev_km] = 0.0

    return df.dropna(subset=[stc.station_id, stc.lat, stc.lon]).copy()


def load_inputs(
    catalog_csv: Path | str,
    picks_csv: Path | str,
    stations_json: Path | str,
    catc: CatalogCols = CatalogCols(),
    pkc: PicksCols = PicksCols(),
    stc: StationCols = StationCols(),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three files needed by waveform visualizations:
    catalog CSV, picks CSV, and station JSON.
    """
    catalog = load_catalog_csv(catalog_csv, catc=catc)

    picks = pd.read_csv(picks_csv)
    picks = normalize_picks(picks, pkc=pkc)

    stations = load_stations_json(stations_json)
    stations = normalize_stations_for_waveforms(stations, stc=stc)

    return catalog, picks, stations


# ============================================================
# WAVEFORMS: INDEX + CACHE
# ============================================================

class WaveformIndex:
    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("t0").reset_index(drop=True)
        self._t0 = self.df["t0"].to_numpy(dtype=np.float64)
        self._t1 = self.df["t1"].to_numpy(dtype=np.float64)

    @staticmethod
    def build(roots: List[Path], glob_pattern: str = "**/waveforms/*.mseed", quiet: bool = True) -> "WaveformIndex":
        from obspy import read

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
                    if all(tr.stats.npts == 0 for tr in st):
                        # Header-only reading returns empty traces for files larger than 2 GB;
                        # use the first and last records of the file instead
                        from obspy.io.mseed.util import get_start_and_end_time
                        t0, t1 = get_start_and_end_time(str(fp))
                    rows.append({"path": str(fp), "t0": float(t0.timestamp), "t1": float(t1.timestamp)})
                except Exception as e:
                    if not quiet:
                        print(f"[index warn] {fp}: {e}")

        if not rows:
            raise RuntimeError("No .mseed files found (or their headers could not be read).")

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


def harmonize_sampling_rates(st, rtol: float = 1e-4):
    """
    Make traces with the same id share one sampling rate so Stream.merge() works.

    Some day files store the rate measured by the datalogger (e.g. 50.0000114 or
    49.9999848 Hz instead of 50 Hz); ObsPy refuses to merge consecutive days whose
    rates differ by that jitter. Rates within `rtol` of the nominal value are snapped
    to it; a trace whose nominal rate really differs from the rest of its id (station
    changed its rate) is resampled to the rate of the longest trace of that id.
    """
    for tr in st:
        sr = float(tr.stats.sampling_rate)
        nominal = float(round(sr)) if sr >= 1.0 else 1.0 / round(1.0 / sr)
        if sr != nominal and abs(sr - nominal) <= rtol * nominal:
            tr.stats.sampling_rate = nominal

    by_id: Dict[str, list] = {}
    for tr in st:
        by_id.setdefault(tr.id, []).append(tr)
    for trs in by_id.values():
        target = max(trs, key=lambda t: t.stats.npts).stats.sampling_rate
        for tr in trs:
            if tr.stats.sampling_rate != target:
                tr.resample(target)
    return st


class StreamCache:
    def __init__(self):
        self._key = None
        self._st = None

    def get_stream(self, t_start: UTCDateTime, t_end: UTCDateTime, wf_index: WaveformIndex, pad_neighbors: int = 1):
        from obspy import read

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

        harmonize_sampling_rates(st)
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


# ============================================================
# PICKS / EVENT FILTERING
# ============================================================

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


def networks_per_event(
    catalog: pd.DataFrame,
    picks: pd.DataFrame,
    catc: CatalogCols = CatalogCols(),
    pkc: PicksCols = PicksCols(),
) -> pd.DataFrame:
    """
    For every catalog event, which networks have picks of that event.

    The catalog event_id is the same number as event_id_mapped in the picks file
    (GAMMA_picks_cleaned.csv), so the picks of one event are the rows with that number.

    Returns a copy of the catalog with three new columns:
      networks     e.g. "3Z+6H+DK" (sorted, joined with "+"; "" if no picks found)
      n_networks   number of different networks (e.g. 3)
      n_stations   number of different stations with a pick
    """
    pk_col = pkc.event_id_mapped if pkc.event_id_mapped in picks.columns else pkc.event_id
    p = picks[[pk_col, pkc.station_id]].dropna().copy()
    p["_net"] = p[pkc.station_id].map(network_of)
    g = p.groupby(pk_col)
    nets = g["_net"].apply(lambda s: "+".join(sorted(set(s))))
    nsta = g[pkc.station_id].nunique()

    out = catalog.copy()
    ids = pd.to_numeric(out[catc.event_id], errors="coerce")
    out["networks"] = ids.map(nets).fillna("")
    out["n_networks"] = out["networks"].map(lambda s: len(s.split("+")) if s else 0)
    out["n_stations"] = ids.map(nsta).fillna(0).astype(int)
    return out


def select_events_by_networks(
    catalog: pd.DataFrame,
    picks: pd.DataFrame,
    require_networks: Optional[Sequence[str]] = None,
    min_networks: int = 1,
    catc: CatalogCols = CatalogCols(),
    pkc: PicksCols = PicksCols(),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Keep only the events picked by certain networks.

    require_networks : every network in this list must have at least one pick of the event
                       (e.g. ["3Z", "6H", "DK"] = events seen by the three networks).
                       None or [] = no requirement.
    min_networks     : the event must have picks from at least this many different networks
                       (1 = no requirement, 2 = seen by two networks or more, ...).

    Returns the selected events (with the columns of networks_per_event). With verbose=True
    it prints how many events each combination of networks has, before and after.
    """
    cat = networks_per_event(catalog, picks, catc=catc, pkc=pkc)
    keep = cat["n_networks"] >= int(min_networks)
    for net in (require_networks or []):
        keep &= cat["networks"].str.split("+").map(lambda nets: net in nets)
    sel = cat[keep]
    if verbose:
        print("Events per combination of networks (whole catalog):")
        print(cat["networks"].replace("", "(no picks)").value_counts().to_string())
        print(f"\nSelected: {len(sel)} of {len(cat)} events "
              f"(require_networks={list(require_networks or [])}, min_networks={min_networks})")
    return sel


def add_mpl_xnum(pw: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    import matplotlib.dates as mdates

    pw = pw.copy()
    pw["xnum"] = mdates.date2num(pw[time_col].dt.tz_convert("UTC").dt.tz_localize(None))
    return pw


# ============================================================
# DISTANCES AND TRACE SELECTION
# ============================================================

def stations_by_hypo_distance_km(stations: pd.DataFrame, ev_lat: float, ev_lon: float, ev_depth_km: float,
                                 stc: StationCols = StationCols()) -> Tuple[List[str], Dict[str, float], float]:
    from obspy.geodetics.base import gps2dist_azimuth

    stations = normalize_stations_for_waveforms(stations, stc=stc)
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


# ============================================================
# EVENT WAVEFORM VISUALIZATIONS
# ============================================================

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
    from obspy import UTCDateTime

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


def _select_traces_for_sid(st, sid: str, components: str = "Z"):
    """Traces of one station id 'NET.STA.LOC.PREFIX' (e.g. '3Z.SK01L..CH').
    components = "Z": only the vertical; "all": vertical first, then the horizontals
    (N/E or 1/2 or 2/3, whatever the station has). Returns a list of (trace_copy, channel)."""
    parts = sid.split(".")
    if len(parts) < 4:
        return []
    net, sta, loc, chprefix = parts[0], parts[1], parts[2], parts[3]
    sel = st.select(network=net, station=sta, location=loc, channel=f"{chprefix}*")
    if len(sel) == 0:
        sel = st.select(network=net, station=sta, location=loc)
    vert = [tr for tr in sel if tr.stats.channel.endswith("Z")]
    horiz = sorted([tr for tr in sel if not tr.stats.channel.endswith("Z")], key=lambda tr: tr.stats.channel)
    chosen = vert[:1] if components == "Z" else vert[:1] + horiz[:2]
    return [(tr.copy(), tr.stats.channel) for tr in chosen]


def plot_event_record_section(
    row_event: pd.Series,
    picks: pd.DataFrame,
    stations: pd.DataFrame,
    cache: "StreamCache",
    wf_index: "WaveformIndex",
    *,
    pre_s: float = 2.0,
    post_s: float = 25.0,
    freqmin: Optional[float] = 2.0,
    freqmax: Optional[float] = 20.0,
    corners: int = 4,
    components: str = "Z",
    group_by_network: bool = True,
    min_prob: Optional[float] = None,
    amplitude: float = 0.45,
    show_unpicked_km: Optional[float] = None,
    vp_kms: Optional[float] = None,
    vs_kms: Optional[float] = None,
    p_color: str = "black",
    s_color: str = "red",
    save_png: Optional[Path] = None,
    figsize: Optional[Tuple[float, float]] = None,
):
    """
    Record section of one event: one row per station (or per channel), EQUALLY SPACED,
    so stations at very different distances do not leave empty gaps and stations of an
    array do not overlap.

    - Rows are ordered by hypocentral distance (nearest at the top). With
      group_by_network=True they are first grouped by network, then ordered by distance
      inside each network, with a line between networks.
    - Each trace is normalised to its OWN maximum inside the window, so small signals are
      as visible as large ones (amplitudes are NOT comparable between rows).
    - Colour = network (fixed colours, see NETWORK_COLOURS).
    - Label of each row: NET.STA, channel and hypocentral distance (km).
    - P picks: black ticks; S picks: red ticks. Only stations with picks for this event.
    - X axis: seconds after the origin time (the title gives the origin time).

    components: "Z" (vertical only) or "all" (vertical + horizontals, one row each).
    amplitude:  half height of each wiggle as a fraction of the row spacing (0.45 = almost
                touching the next row).
    show_unpicked_km: also draw, in GREY, the stations WITHOUT a pick of this event that are
                closer than this hypocentral distance (km). Useful to check the association:
                a real event should also be visible on nearby stations that have no pick.
                None = only stations with picks.
    vp_kms, vs_kms: if given, draw short dotted marks at the arrival times predicted by a
                straight ray at constant speed (time = hypocentral distance / speed), P in
                black and S in red. Only a rough guide (no layers), to see whether the picks
                are where an event at this location would put them.
    Returns (fig, ax, meta) with meta = {"files", "stations_used", "rows"}.
    """
    import matplotlib.pyplot as plt
    from obspy import UTCDateTime

    catc, pkc = CatalogCols(), PicksCols()
    t0 = UTCDateTime(row_event[catc.time].to_pydatetime())
    t_start, t_end = t0 - float(pre_s), t0 + float(post_s)

    pw = picks_for_event(row_event=row_event, picks=picks, t_start=t_start, t_end=t_end,
                         catc=catc, pkc=pkc, prefer_event_id=True, min_prob=min_prob)
    if pw.empty:
        print("[record section] No picks for this event.")
        return None, None, {"files": [], "stations_used": [], "rows": 0}

    _, dist_map, _ = stations_by_hypo_distance_km(
        stations=stations, ev_lat=float(row_event[catc.lat]), ev_lon=float(row_event[catc.lon]),
        ev_depth_km=float(row_event.get(catc.depth_km, 0.0)), stc=StationCols())
    sids = [sid for sid in pw[pkc.station_id].dropna().unique() if sid in dist_map]
    if not sids:
        print("[record section] Picks exist, but their stations are not in the station list.")
        return None, None, {"files": [], "stations_used": [], "rows": 0}
    picked = set(sids)
    if show_unpicked_km is not None:
        sids += [sid for sid, d in dist_map.items() if sid not in picked and d <= float(show_unpicked_km)]
    if group_by_network:
        # networks ordered by their nearest station, then stations by distance inside each network
        nearest = {}
        for sid in sids:
            nearest[network_of(sid)] = min(nearest.get(network_of(sid), np.inf), dist_map[sid])
        sids.sort(key=lambda sid: (nearest[network_of(sid)], network_of(sid), dist_map[sid]))
    else:
        sids.sort(key=lambda sid: dist_map[sid])

    st, files = get_trimmed_stream(cache, t_start, t_end, wf_index, pad_neighbors=1)
    if st is None or len(st) == 0:
        print("[record section] No waveforms in window.")
        return None, None, {"files": [str(f) for f in files], "stations_used": [], "rows": 0}

    # one row per (station, channel)
    rows = []
    for sid in sids:
        for tr, ch in _select_traces_for_sid(st, sid, components=components):
            tf = tr.copy()
            tf.detrend("demean")
            tf.taper(max_percentage=0.05, type="cosine")
            if freqmin is not None and freqmax is not None:
                nyq = 0.5 * float(tf.stats.sampling_rate)
                fmin = max(0.001, min(float(freqmin), nyq * 0.99))
                fmax = max(fmin + 0.001, min(float(freqmax), nyq * 0.99))
                tf.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=int(corners), zerophase=True)
            d = tf.data.astype(float)
            m = float(np.max(np.abs(d))) if d.size else 0.0
            if not np.isfinite(m) or m == 0.0:
                continue
            rows.append(dict(sid=sid, ch=ch, x=tf.times(reftime=t0), d=d / m))
    if not rows:
        print("[record section] No usable traces for the stations with picks.")
        return None, None, {"files": [str(f) for f in files], "stations_used": [], "rows": 0}

    if figsize is None:
        figsize = (11, max(4.0, 0.32 * len(rows) + 1.5))
    fig, ax = plt.subplots(figsize=figsize)
    pw = pw.copy()
    pw["t_rel"] = (pw[pkc.time] - pd.Timestamp(t0.datetime, tz="UTC")).dt.total_seconds()
    labels, prev_net = [], None
    for k, r in enumerate(rows):
        y = -k                                    # first row at the top
        net = network_of(r["sid"])
        colour = network_colour(net) if r["sid"] in picked else "0.65"
        if group_by_network and prev_net is not None and net != prev_net:
            ax.axhline(y + 0.5, color="0.6", lw=0.8, ls="--")
        prev_net = net
        ax.plot(r["x"], y + amplitude * r["d"], color=colour, lw=0.6)
        sta = r["sid"].split(".")[1]
        labels.append(f"{net}.{sta} {r['ch']}  {dist_map[r['sid']]:.1f} km"
                      + ("" if r["sid"] in picked else "  (no pick)"))
        for v, colour_pred in ((vp_kms, p_color), (vs_kms, s_color)):
            if v:
                ax.vlines(dist_map[r["sid"]] / float(v), y - 0.45, y + 0.45, colors=colour_pred,
                          lw=1.0, ls=":", zorder=2)
        for phase, colour_pick in (("P", p_color), ("S", s_color)):
            m = (pw[pkc.station_id] == r["sid"]) & pw[pkc.phase].astype(str).str.upper().eq(phase)
            for t in pw.loc[m, "t_rel"]:
                ax.vlines(t, y - 0.45, y + 0.45, colors=colour_pick, lw=1.6, zorder=3)
    ax.set_yticks([-k for k in range(len(rows))])
    ax.set_yticklabels(labels, fontsize=7)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(network_colour(network_of(r["sid"])) if r["sid"] in picked else "0.5")
    ax.set_ylim(-len(rows) + 0.3, 0.7)
    ax.set_xlim(-float(pre_s), float(post_s))
    ax.axvline(0, color="0.4", lw=0.8, ls=":")
    ax.plot([], [], color=p_color, lw=1.6, label="P pick")
    ax.plot([], [], color=s_color, lw=1.6, label="S pick")
    if vp_kms:
        ax.plot([], [], color=p_color, lw=1.0, ls=":", label=f"P predicted ({float(vp_kms):.2f} km/s)")
    if vs_kms:
        ax.plot([], [], color=s_color, lw=1.0, ls=":", label=f"S predicted ({float(vs_kms):.2f} km/s)")
    if show_unpicked_km is not None:
        ax.plot([], [], color="0.65", lw=1.0, label=f"no pick (< {float(show_unpicked_km):g} km)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("seconds after origin time")
    band = f"{freqmin:g}-{freqmax:g} Hz" if freqmin is not None and freqmax is not None else "no filter"
    ax.set_title(f"Record section | event_id={row_event.get(catc.event_id, '')} | {row_event[catc.time]} | "
                 f"depth {float(row_event.get(catc.depth_km, float('nan'))):.1f} km | {band} | "
                 f"each trace normalised to its own maximum", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    meta = {"files": [str(f) for f in files], "stations_used": sorted({r["sid"] for r in rows if r["sid"] in picked}),
            "rows": len(rows)}
    if save_png is not None:
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return None, None, meta
    return fig, ax, meta


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
    from obspy import UTCDateTime

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


# ============================================================
# MOVEOUT FIGURE FOR PUBLICATION (larger text, custom ticks)
# ============================================================

def plot_event_moveout_pretty(
    row_event,
    picks,
    stations,
    cache,
    wf_index,
    *,
    # waveform window (seconds before / after origin time)
    pre_s=5,
    post_s=40,
    min_prob=None,

    # bandpass filter (Hz)
    freqmin=1.0,
    freqmax=5.0,
    corners=4,
    zerophase=True,

    # figure size (inches)
    figsize=(13, 7),

    # vertical spacing and wiggle amplitude
    wiggle_frac_of_spacing=1.5,
    y_pad_km=2.0,
    y_tick_step_km=None,

    # picks
    p_color="black",
    s_color="red",
    p_style="vline",
    s_style="vline",
    line_height_km=2.2,
    line_width=1.8,
    marker_size=5.0,

    # text sizes
    label_fontsize=20,
    tick_fontsize=17,
    title_fontsize=17,
    legend_fontsize=15,

    # x-axis ticks
    x_tick_seconds=10,
    x_tick_rotation=0,

    # axis and tick line widths
    tick_length=7,
    tick_width=1.4,
    spine_width=1.2,

    # margins (fractions of the figure)
    left=0.11,
    right=0.98,
    bottom=0.14,
    top=0.90,

    # optional saving
    save_path=None,
    save_format=None,
    dpi=400,
):
    """
    Wrapper around plot_event_moveout_picks_only with control of:
      - figure size
      - font sizes
      - vertical spacing
      - x/y tick spacing
      - pick and axis line widths
      - optional saving (format taken from the save_path extension)

    Returns:
      fig, ax, meta
    """

    import numpy as np
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker
    from pathlib import Path

    fig, ax, meta = plot_event_moveout_picks_only(
        row_event=row_event,
        picks=picks,
        stations=stations,
        cache=cache,
        wf_index=wf_index,

        pre_s=pre_s,
        post_s=post_s,
        min_prob=min_prob,

        freqmin=freqmin,
        freqmax=freqmax,
        corners=corners,
        zerophase=zerophase,

        p_style=p_style,
        s_style=s_style,
        p_color=p_color,
        s_color=s_color,
        marker_size=marker_size,
        line_width=line_width,
        line_height_km=line_height_km,

        wiggle_frac_of_spacing=wiggle_frac_of_spacing,
        y_pad_km=y_pad_km,

        figsize=figsize,
        save_png=None,
    )

    if fig is None or ax is None:
        return fig, ax, meta

    # -------------------------
    # Labels and title
    # -------------------------
    ax.set_xlabel("Time", fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel("Distance (km)", fontsize=label_fontsize, labelpad=10)
    catc = CatalogCols()

    event_time = pd.to_datetime(row_event[catc.time], utc=True, errors="coerce")

    if pd.notna(event_time):
        event_time_str = event_time.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        event_time_str = str(row_event[catc.time])

    eid = row_event.get(catc.event_id, "")

    ax.set_title(
        f"event_id={eid} | {event_time_str}",
        fontsize=title_fontsize,
        pad=12,
    )

    # -------------------------
    # Ticks
    # -------------------------
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=tick_fontsize,
        length=tick_length,
        width=tick_width,
    )

    ax.tick_params(axis="x", labelrotation=x_tick_rotation)

    # x-axis tick spacing in seconds
    if x_tick_seconds is not None:
        ax.xaxis.set_major_locator(
            mdates.SecondLocator(interval=int(x_tick_seconds))
        )
        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%H:%M:%S")
        )

    # y-axis tick spacing in km
    if y_tick_step_km is not None:
        ax.yaxis.set_major_locator(
            mticker.MultipleLocator(float(y_tick_step_km))
        )

    # -------------------------
    # Frame (spine) width
    # -------------------------
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    # -------------------------
    # Legend
    # -------------------------
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontsize(legend_fontsize)
        leg.get_frame().set_linewidth(0.8)

    # -------------------------
    # Margins
    # -------------------------
    fig.subplots_adjust(
        left=left,
        right=right,
        bottom=bottom,
        top=top,
    )

    # -------------------------
    # Optional saving
    # -------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_format is None:
            save_format = save_path.suffix.replace(".", "")
            if save_format == "":
                save_format = "png"

        fig.savefig(
            save_path,
            format=save_format,
            dpi=int(dpi),
            bbox_inches="tight",
            pad_inches=0.03,
        )

    return fig, ax, meta


# Backward-compatible private name from exploratory notebook.
_draw_3d_box_edges = draw_3d_box_edges
_style_3d_axes = style_3d_axes


__all__ = [
    "CatalogCols",
    "StationCols",
    "PicksCols",
    "load_catalog_csv",
    "load_stations_json",
    "load_inputs",
    "ensure_dir",
    "normalize_catalog",
    "normalize_picks",
    "normalize_stations_for_waveforms",
    "filter_catalog_time",
    "filter_catalog_by_radius",
    "make_quakes_for_map",
    "make_stations_wide_for_map",
    "WaveformIndex",
    "StreamCache",
    "get_trimmed_stream",
    "picks_for_event",
    "add_mpl_xnum",
    "stations_by_hypo_distance_km",
    "window_l2_energy",
    "plot_event_moveout_picks_only",
    "plot_event_moveout_pretty",
    "plot_event_spectrograms_vertical_picks_only",
    "ellipse_lonlat_from_km",
    "plot_catalog_3d_topo_seismicity",
    "plot_mapview_seismicity",
    "plot_mapview_basemap_stations",
    "plot_mapview_seismicity_zoom",
    "plot_mapview_seismicity_zoom_2",
    "plot_timeline_histogram",
    "plot_timeline_legend_only",
    "plot_event_count_time",
    "plot_event_count_time_v2",
    "plot_mapview3d_months",
    "plot_mapview3d_timecolor",
    "plot_mapview3d_groups",
    "project_profile_distances",
    "add_profile_line",
    "plot_cross_section_seismicity",
    "plot_plane_orientation_rose",
    "build_fmd",
    "estimate_mc_maxc",
    "fit_gutenberg_richter_line",
    "plot_fmd_cumulative",
    "plot_magnitude_time",
    "build_location_evolution_df",
    "add_panel_label",
    "plot_location_evolution_map",
    "FI_COMPOSITION_COLORS",
    "fi_monthly_tables",
    "add_month_spans_inside_axis",
    "plot_fi_monthly_composition",
    "plot_monthly_fi_span_legend_only",
    "load_and_normalize",
    "filter_catalog_by_year",
    "obspy_horizontal_distance_km",
    "obspy_distance_3d_km",
    "match_catalogs_time_space",
    "enforce_one_to_one",
    "check_one_to_one",
    "compute_residuals",
    "calculate_pair_metrics",
    "run_all_comparisons",
    "build_metrics_table",
    "add_component_residuals_km",
    "build_km_component_metrics_table",
    "make_residual_summary_tables",
    "plot_match_histograms",
    "plot_match_km_histograms",
    "plot_match_km_scatter",
    "plot_combined_reference_residual_histograms",
    "add_scale_bar",
    "style_axis_full_box",
    "get_limits_multi",
    "compute_counts_per_bin",
    "plot_three_catalog_overview_pretty",
    "get_common_bounds_for_two_catalogs",
    "lonlat_to_local_km",
    "local_km_to_lonlat",
    "dbscan_parameter_scan",
    "select_best_dbscan",
    "dbscan_cluster_table",
    "select_cluster_ids",
    "fit_best_surface_pca",
    "make_fitted_surface_mesh",
    "plane_dip_deg",
    "plane_dip_direction_deg",
    "surface_fit_orientation",
    "fit_cluster_surfaces",
    "cluster_style_map",
    "plot_cluster_surfaces_3d",
    "plot_cluster_surfaces_interactive",
    "CLUSTER_STYLES",
    "animate_mapview3d_timecolor",
    "plot_regional_location_map",
    "plot_aniakchak_alaska_location",
    "draw_3d_box_edges",
    "style_3d_axes",
]