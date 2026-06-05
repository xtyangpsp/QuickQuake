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

This version is intentionally conservative:
- preserves the same merge logic on healthy inputs
- excludes the merged_dir subtree from discovery
- adds traceability / sanity warnings
- does NOT deduplicate or alter the scientific output automatically
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

CHUNK_RE = re.compile(r"^\d{8}T\d{6}$")


# ============================================================
# Discovery helpers
# ============================================================

def _is_under(path: Path, maybe_parent: Optional[Path]) -> bool:
    if maybe_parent is None:
        return False
    try:
        path.resolve().relative_to(maybe_parent.resolve())
        return True
    except Exception:
        return False


def _discover_files(
    base_dir: Path,
    pattern: str,
    exclude_subtrees: Optional[Iterable[Path]] = None,
) -> List[Path]:
    exclude_subtrees = [p.resolve() for p in (exclude_subtrees or [])]
    files: List[Path] = []
    for f in sorted(base_dir.glob(pattern)):
        rf = f.resolve()
        if any(_is_under(rf, ex) for ex in exclude_subtrees):
            continue
        files.append(rf)
    return files


def _summarize_discovery(label: str, files: List[Path]) -> None:
    print(f"[{label}] Found {len(files)} files")
    if not files:
        return

    parent_names = [f.parent.name for f in files]
    counts = Counter(parent_names)
    dup_parent_names = {k: v for k, v in counts.items() if v > 1}

    invalid_chunk_names = [name for name in counts if not CHUNK_RE.match(name)]
    if invalid_chunk_names:
        print(f"[{label}][WARN] {len(invalid_chunk_names)} parent directories do not look like YYYYmmddTHHMMSS.")
        for name in invalid_chunk_names[:10]:
            print(f"  - {name}")

    if dup_parent_names:
        print(f"[{label}][WARN] {len(dup_parent_names)} repeated chunk names found in different paths.")
        for name, n in list(sorted(dup_parent_names.items()))[:10]:
            print(f"  - {name}: {n} files")


# ============================================================
# Merge: gamma_picks.csv -> gammapicks_id.csv
# ============================================================

def merge_gamma_picks_with_event_id(
    base_dir: Path,
    out_dir: Path,
    exclude_subtrees: Optional[Iterable[Path]] = None,
) -> Path:
    pattern = "**/gamma_picks.csv"
    out_path = out_dir / "gammapicks_id.csv"

    files = _discover_files(base_dir, pattern, exclude_subtrees=exclude_subtrees)
    _summarize_discovery("merge picks", files)

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            window_id = f.parent.name

            if "event_idx" not in df.columns:
                raise ValueError(f"Missing 'event_idx' in {f}. Columns: {list(df.columns)}")

            mask = df["event_idx"].notna()
            df["event_id"] = pd.NA
            df.loc[mask, "event_id"] = (
                window_id + "_" + df.loc[mask, "event_idx"].astype(float).astype(int).astype(str)
            )

            frames.append(df)
        except Exception as e:
            print(f"[merge picks][WARN] Error reading {f}: {e}")

    if not frames:
        raise RuntimeError("Could not read any gamma_picks.csv files. Check base_dir and the folder structure.")

    merged = pd.concat(frames, ignore_index=True)
    n_unq = merged["event_id"].nunique(dropna=True)
    print(f"[merge picks] Total picks: {len(merged)} | unique event_id values: {n_unq}")

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[merge picks] Saved: {out_path}")
    return out_path


# ============================================================
# Merge: gamma_catalog.csv -> gammacatalog_id.csv
# ============================================================

def merge_gamma_catalog_with_event_id(
    base_dir: Path,
    out_dir: Path,
    exclude_subtrees: Optional[Iterable[Path]] = None,
) -> Path:
    pattern = "**/gamma_catalog.csv"
    out_path = out_dir / "gammacatalog_id.csv"

    files = _discover_files(base_dir, pattern, exclude_subtrees=exclude_subtrees)
    _summarize_discovery("merge catalog", files)

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            window_id = f.parent.name

            if "event_index" not in df.columns:
                raise ValueError(f"Missing 'event_index' in {f}. Columns: {list(df.columns)}")

            mask = df["event_index"].notna()
            df["event_id"] = pd.NA
            df.loc[mask, "event_id"] = (
                window_id + "_" + df.loc[mask, "event_index"].astype(float).astype(int).astype(str)
            )

            frames.append(df)
        except Exception as e:
            print(f"[merge catalog][WARN] Error reading {f}: {e}")

    if not frames:
        raise RuntimeError("Could not read any gamma_catalog.csv files. Check base_dir and the folder structure.")

    merged = pd.concat(frames, ignore_index=True)
    n_unq = merged["event_id"].nunique(dropna=True)
    print(f"[merge catalog] Total events (rows):{len(merged)} | unique event_id values: {n_unq}")
    if n_unq > 0 and len(merged) > 3 * n_unq:
        print(
            "[merge catalog][WARN] The number of catalog rows is much larger than the number of unique event_id values. "
            "This does not change the result, but it suggests contamination or duplication in the input."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[merge catalog] Saved: {out_path}")
    return out_path


# ============================================================
# Merge: stations.json (unique by key, same dict format)
# ============================================================

def merge_stations_json_unique(
    base_dir: Path,
    out_json: Path,
    pattern: str = "**/stations.json",
    prefer: str = "first",          # "first" or "last"
    check_conflicts: bool = True,
    tol: float = 1e-6,
    exclude_subtrees: Optional[Iterable[Path]] = None,
) -> Tuple[Path, int, int, List[str]]:

    files = _discover_files(base_dir, pattern, exclude_subtrees=exclude_subtrees)
    if not files:
        raise FileNotFoundError(f"Could not find stations.json with pattern '{pattern}' inside {base_dir}")

    _summarize_discovery("merge stations", files)

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
            conflicts.append(f"[WARN] Could not read {f}: {e}")
            continue

        if not isinstance(d, dict):
            conflicts.append(f"[WARN] {f} is not a JSON dictionary. Skipping.")
            continue

        for sid, meta in d.items():
            if sid not in merged:
                merged[sid] = meta
                continue

            if check_conflicts and not same_station_meta(merged[sid], meta):
                conflicts.append(f"[CONFLICT] {sid} differs. Keeping '{prefer}'. File: {f}")
                if prefer == "last":
                    merged[sid] = meta
            else:
                if prefer == "last":
                    merged[sid] = meta

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(merged, indent=2, sort_keys=True))

    print(f"[merge stations] Unique stations: {len(merged)}")
    print(f"[merge stations] Saved: {out_json}")

    if conflicts:
        print(f"[merge stations][WARN] Conflicts/warnings: {len(conflicts)} (showing up to 10)")
        for msg in conflicts[:10]:
            print("  " + msg)

    return out_json, len(files), len(merged), conflicts


# ============================================================
# Merge orchestrator called by driver
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

    exclude_subtrees = [merged_dir]

    # 1) Merge CSVs with event_id
    picks_out = merge_gamma_picks_with_event_id(data_root, merged_dir, exclude_subtrees=exclude_subtrees)
    catalog_out = merge_gamma_catalog_with_event_id(data_root, merged_dir, exclude_subtrees=exclude_subtrees)

    # 2) Merge unique stations without duplication, preserving the same format
    station_out = indir / f"{namebase}_station_list.json"
    merge_stations_json_unique(
        base_dir=data_root,
        out_json=station_out,
        pattern="**/stations.json",
        prefer="first",
        check_conflicts=True,
        exclude_subtrees=exclude_subtrees,
    )

    print("\n Merge completed.")
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
                   help="Base directory containing the chunks (e.g., .../QuickQuake/data)")
    p.add_argument("--merged_dir", type=str, default=None,
                   help= "Destination merged directory (default: <data_root>/merged)")
    p.add_argument("--namebase", type=str, default="GAMMA",
                   help="Prefix for station_list.json (default: GAMMA)")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    merged_dir = Path(args.merged_dir) if args.merged_dir else None
    run_merge_block(data_root=data_root, merged_dir=merged_dir, namebase=args.namebase)


if __name__ == "__main__":
    main()