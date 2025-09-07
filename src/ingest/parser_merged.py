# src/ingest/parser_merged.py
import argparse
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from db.models import RawFile, Session, SessionLocal, Trackpoint

load_dotenv()

NS = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}


# ------------------------ TCX parsing ------------------------
def parse_tcx(path: str) -> List[Dict[str, Any]]:
    """
    Parse a single TCX file into a list of trackpoint dicts.
    Keys: time, altitude, distance, heart_rate, cadence, power
    """
    tree = ET.parse(path)
    root = tree.getroot()

    data: List[Dict[str, Any]] = []
    for tp in root.findall(".//ns:Trackpoint", NS):
        time_elem = tp.find("ns:Time", NS)
        time = None
        if time_elem is not None and time_elem.text:
            txt = time_elem.text
            if txt.endswith("Z"):
                txt = txt.replace("Z", "+00:00")
            time = datetime.fromisoformat(txt)
        altitude = tp.find("ns:AltitudeMeters", NS)
        distance = tp.find("ns:DistanceMeters", NS)
        hr_elem = tp.find(".//ns:HeartRateBpm/ns:Value", NS)
        cadence = tp.find("ns:Cadence", NS)

        # power might live in Extensions
        power = None
        ext = tp.find("ns:Extensions", NS)
        if ext is not None:
            for child in ext.iter():
                tag_lower = child.tag.lower()
                if "watts" in tag_lower or "power" in tag_lower:
                    try:
                        power = float(child.text)
                    except Exception:
                        pass

        row = {
            "time": time,
            "altitude": float(altitude.text) if (altitude is not None and altitude.text) else np.nan,
            "distance": float(distance.text) if (distance is not None and distance.text) else np.nan,
            "heart_rate": int(hr_elem.text) if (hr_elem is not None and hr_elem.text) else None,
            "cadence": int(cadence.text) if (cadence is not None and cadence.text) else None,
            "power": float(power) if power is not None else None,
        }
        data.append(row)
    return data


# ------------------------ Merge helpers ------------------------
def _is_nan(x: Optional[float]) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def _bounds(points: List[Dict[str, Any]]) -> Tuple[Optional[datetime], Optional[datetime]]:
    times = [p["time"] for p in points if p["time"] is not None]
    return (min(times), max(times)) if times else (None, None)


def merge_no_gap_with_boundaries(paths: List[str]) -> Tuple[List[Dict[str, Any]], List[datetime]]:
    """
    Merge multiple TCX files into a single continuous timeline (gap=0).
    Distances are re-based per file then chained so they stay cumulative.
    Returns (merged_points, boundary_times) where boundary_times are the
    desired start timestamps of each subsequent file on the merged timeline.
    """
    per_file: List[List[Dict[str, Any]]] = [parse_tcx(p) for p in paths]
    bnds = [_bounds(pts) for pts in per_file]

    # order files by original start
    order = sorted(range(len(bnds)), key=lambda i: (bnds[i][0] or datetime.min.replace(tzinfo=timezone.utc)))
    if not order:
        return [], []

    first_idx = order[0]
    anchor = bnds[first_idx][0]
    desired_starts: Dict[int, datetime] = {}
    last_end = anchor
    for idx in order:
        s, e = bnds[idx]
        desired_starts[idx] = last_end
        if s and e:
            last_end = last_end + (e - s)

    merged: List[Dict[str, Any]] = []
    seen_times = set()
    boundary_times: List[datetime] = []
    cumulative_offset = 0.0

    for idx, pts in enumerate(per_file):
        if not pts:
            continue
        s, e = bnds[idx]
        desired_start = desired_starts.get(idx, s)

        # keep marker for every file *after* the first
        if desired_start and desired_start != anchor:
            boundary_times.append(desired_start)

        # distance base for this file
        base = None
        for p in pts:
            d = p["distance"]
            if not _is_nan(d):
                base = d
                break

        last_non_nan_this_file = None
        for p in pts:
            # time shift: new_t = desired_start + (t - s)
            t = p["time"]
            if t is not None and s is not None and desired_start is not None:
                new_t = desired_start + (t - s)
            else:
                new_t = t

            # distance rebase + cumulative offset
            d = p["distance"]
            if _is_nan(d):
                new_d = np.nan
            else:
                new_d = d - (base if base is not None else 0.0) + cumulative_offset
                last_non_nan_this_file = new_d

            row = dict(p)
            row["time"] = new_t
            row["distance"] = new_d

            if new_t is not None and new_t not in seen_times:
                seen_times.add(new_t)
                merged.append(row)

        # bump the offset at the end of the file to keep cumulative distance
        if last_non_nan_this_file is not None:
            cumulative_offset = last_non_nan_this_file

    merged.sort(key=lambda r: r["time"] or datetime.min.replace(tzinfo=timezone.utc))
    boundary_times.sort()
    return merged, boundary_times


def fix_boundary_spikes_inplace(points: List[Dict[str, Any]],
                                boundary_times: List[datetime],
                                window_after_s: float = 3.0,
                                seek_next_valid_s: float = 6.0,
                                min_valid_w: float = 20.0) -> None:
    """
    Smooth power dips to ~0W that sometimes happen at file boundaries.
    For each boundary B:
      - find last point before B (t_prev, p_prev)
      - find first point in [B, B+seek_next_valid_s] with power >= min_valid_w (t_next, p_next)
      - for points in [B, B+window_after_s], linearly interpolate between p_prev and p_next
    """
    if not points or not boundary_times:
        return
    # Precompute elapsed seconds
    t0 = points[0]["time"]
    def to_s(t: datetime) -> float:
        return (t - t0).total_seconds()

    # index points by time for faster search
    times = [p["time"] for p in points]
    powers = [p["power"] for p in points]

    for B in boundary_times:
        b_s = to_s(B)
        # prev index
        idx_prev = None
        for i in range(len(times) - 1, -1, -1):
            if times[i] is not None and to_s(times[i]) < b_s:
                idx_prev = i
                break
        if idx_prev is None:
            continue
        p_prev = powers[idx_prev]
        if p_prev is None or (isinstance(p_prev, float) and np.isnan(p_prev)):
            continue

        # candidate indices to correct
        idxs_window = [i for i, t in enumerate(times) if t is not None and b_s <= to_s(t) <= b_s + window_after_s]
        if not idxs_window:
            continue

        # find next valid
        idx_next = None
        for i, t in enumerate(times):
            if t is None:
                continue
            ts = to_s(t)
            if b_s <= ts <= b_s + seek_next_valid_s:
                pw = powers[i]
                if pw is not None and not (isinstance(pw, float) and np.isnan(pw)) and float(pw) >= float(min_valid_w):
                    idx_next = i
                    break

        if idx_next is None:
            # fallback: flat fill with p_prev
            for i in idxs_window:
                points[i]["power"] = float(p_prev)
            continue

        t_prev_s = to_s(times[idx_prev])
        t_next_s = to_s(times[idx_next])
        p_next = float(powers[idx_next]) if powers[idx_next] is not None else float(p_prev)
        if t_next_s <= t_prev_s:
            for i in idxs_window:
                points[i]["power"] = float(p_prev)
            continue

        # linear interpolation
        for i in idxs_window:
            ti = to_s(times[i])
            w = (ti - t_prev_s) / (t_next_s - t_prev_s)
            points[i]["power"] = (1.0 - w) * float(p_prev) + w * float(p_next)


# ------------------------ Ingest ------------------------
def ingest_files(
    tcx_paths: List[str],
    session_name: Optional[str] = None,
    fix_boundary_spikes: bool = True,
    window_after_s: float = 3.0,
    seek_next_valid_s: float = 6.0,
    min_valid_w: float = 20.0,
):
    """
    Ingest multiple TCX files as a single session with a *continuous* timeline (no gaps).

    - Distances are re-based per file, then chained cumulatively.
    - Timestamps are shifted so files are back-to-back; the merged start is the
      first timestamp of the earliest file (compatible with your dashboard).
    - Optional power spike fix at boundaries to remove artificial 0W dips.
    """
    if not tcx_paths:
        print("No input files provided.")
        return

    # Merge + (optional) spike correction
    merged_points, boundary_times = merge_no_gap_with_boundaries(tcx_paths)
    if not merged_points:
        print("No trackpoints found across files.")
        return

    if fix_boundary_spikes:
        fix_boundary_spikes_inplace(
            merged_points,
            boundary_times,
            window_after_s=window_after_s,
            seek_next_valid_s=seek_next_valid_s,
            min_valid_w=min_valid_w,
        )

    # Derive session-level bounds on merged timeline
    times = [p["time"] for p in merged_points if p["time"] is not None]
    start_time = min(times) if times else None
    end_time = max(times) if times else None
    duration_s = (end_time - start_time).total_seconds() if (start_time and end_time) else 0.0

    # Distance: last non-nan
    distances = [p["distance"] for p in merged_points if p["distance"] is not None and not np.isnan(p["distance"])]
    total_distance_km = (distances[-1] / 1000.0) if distances else 0.0

    # Per-file summary for metadata
    per_file_meta = []
    for pth in tcx_paths:
        pts = parse_tcx(pth)
        tms = [p["time"] for p in pts if p["time"] is not None]
        s, e = (min(tms), max(tms)) if tms else (None, None)
        dists = [p["distance"] for p in pts if p["distance"] is not None and not np.isnan(p["distance"])]
        dist_km = (dists[-1] / 1000.0) if dists else 0.0
        per_file_meta.append({
            "filename": os.path.basename(pth),
            "start": s,
            "end": e,
            "points": len(pts),
            "distance_km_device": dist_km,
        })

    session_db = SessionLocal()
    try:
        raw = RawFile(
            filename=session_name or f"MERGED_CONTINUOUS:{','.join(os.path.basename(p) for p in tcx_paths)}",
            metadata={
                "merged_files": [os.path.basename(p) for p in tcx_paths],
                "per_file": per_file_meta,
                "mode": "continuous_no_gap",
                "fix_boundary_spikes": bool(fix_boundary_spikes),
                "spike_params": {
                    "window_after_s": window_after_s,
                    "seek_next_valid_s": seek_next_valid_s,
                    "min_valid_w": min_valid_w,
                },
                "duration_s": duration_s,
                "distance_km": total_distance_km,
            },
        )
        session_db.add(raw)
        session_db.flush()

        session = Session(
            raw_file_id=raw.id,
            start_time=start_time,
            end_time=end_time,
            duration_s=duration_s,
            distance_km=total_distance_km,
            elevation_gain_m=0,
            avg_heart_rate=np.nan,
            avg_speed_kmh=np.nan,
        )
        session_db.add(session)
        session_db.flush()

        # Insert merged trackpoints (power already corrected if option enabled)
        for p in merged_points:
            tp = Trackpoint(
                session_id=session.id,
                time=p["time"],
                distance_m=p["distance"],
                altitude_m=p["altitude"],
                heart_rate=p["heart_rate"],
                cadence=p["cadence"],
                power=p["power"],
                power_filtered=None,
                speed_calc_kmh=None,
                pace_min_per_km=None,
                elevation_diff=None,
            )
            session_db.add(tp)

        session_db.commit()
        print(
            f"Ingested MERGED (continuous) session {session.id} from {len(tcx_paths)} files. "
            f"Duration: {duration_s:.1f}s, Distance: {total_distance_km:.3f} km, "
            f"fix_boundary_spikes={fix_boundary_spikes}"
        )
    except Exception as e:
        session_db.rollback()
        print("Error ingesting merged files:", e)
    finally:
        session_db.close()


# ------------------------ CLI ------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest multiple TCX files as a single continuous session (no gaps)")
    parser.add_argument("--inputs", nargs="+", help="Paths to .tcx files (same session parts)", required=False)
    parser.add_argument("--input", help="Single .tcx path (backward compatible)", required=False)
    parser.add_argument("--name", help="Optional name for the merged session", default=None)

    # Spike correction options
    parser.add_argument("--no-fix-boundary-spikes", action="store_true", help="Disable 0W spike correction at boundaries")
    parser.add_argument("--spike-window-after-s", type=float, default=3.0, help="Seconds after boundary to correct (default: 3)")
    parser.add_argument("--spike-seek-next-valid-s", type=float, default=6.0, help="Lookahead seconds to find a valid next power (default: 6)")
    parser.add_argument("--spike-min-valid-w", type=float, default=20.0, help="Minimum W to consider next power as valid (default: 20)")

    args = parser.parse_args()

    paths: List[str] = []
    if args.inputs:
        paths.extend(args.inputs)
    if args.input:
        paths.append(args.input)

    # Deduplicate while preserving order
    seen = set()
    uniq_paths: List[str] = []
    for p in paths:
        if p not in seen:
            uniq_paths.append(p)
            seen.add(p)

    if not uniq_paths:
        print("Please provide one or more TCX paths with --inputs or --input")
        return

    ingest_files(
        uniq_paths,
        session_name=args.name,
        fix_boundary_spikes=not args.no_fix_boundary_spikes,
        window_after_s=args.spike_window_after_s,
        seek_next_valid_s=args.spike_seek_next_valid_s,
        min_valid_w=args.spike_min_valid_w,
    )


if __name__ == "__main__":
    main()
