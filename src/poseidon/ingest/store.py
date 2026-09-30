"""Enregistrement d'une séance parsée en base."""

import math
import os
from dataclasses import asdict, dataclass
from typing import List, Optional

from poseidon.db import RawFile, Session, Trackpoint, get_session
from poseidon.ingest.merge import merge_continuous, smooth_boundary_power
from poseidon.ingest.tcx import TrackPoint, last_distance, parse_tcx, time_bounds


@dataclass(frozen=True)
class BoundaryFix:
    enabled: bool = True
    window_after_s: float = 3.0
    seek_next_valid_s: float = 6.0
    min_valid_w: float = 20.0


def _nan_to_none(x: Optional[float]) -> Optional[float]:
    return None if x is None or math.isnan(x) else x


def _file_summary(path: str, points: List[TrackPoint]) -> dict:
    start, end = time_bounds(points)
    return {
        "filename": os.path.basename(path),
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "points": len(points),
        "distance_km_device": last_distance(points) / 1000.0,
    }


def ingest(
    paths: List[str],
    name: Optional[str] = None,
    boundary_fix: BoundaryFix = BoundaryFix(),
) -> str:
    """Ingère un ou plusieurs TCX comme une seule séance. Retourne son id.

    Un fichier seul est importé tel quel. Plusieurs fichiers sont fusionnés en
    timeline continue (voir merge_continuous), avec correction optionnelle des
    chutes de puissance aux jonctions.
    """
    if not paths:
        raise ValueError("Aucun fichier fourni")
    files = [parse_tcx(p) for p in paths]
    basenames = [os.path.basename(p) for p in paths]

    if len(paths) == 1:
        points = files[0]
        filename = name or basenames[0]
        metadata = {}
    else:
        points, boundaries = merge_continuous(files)
        if boundary_fix.enabled:
            smooth_boundary_power(
                points,
                boundaries,
                window_after_s=boundary_fix.window_after_s,
                seek_next_valid_s=boundary_fix.seek_next_valid_s,
                min_valid_w=boundary_fix.min_valid_w,
            )
        filename = name or f"MERGED_CONTINUOUS:{','.join(basenames)}"
        metadata = {
            "mode": "continuous_no_gap",
            "merged_files": basenames,
            "per_file": [_file_summary(p, f) for p, f in zip(paths, files)],
            "boundary_fix": asdict(boundary_fix),
        }

    if not points:
        raise ValueError("Aucun point trouvé dans les fichiers fournis")

    start, end = time_bounds(points)
    duration_s = (end - start).total_seconds() if start and end else 0.0
    distance_km = last_distance(points) / 1000.0
    metadata.update(duration_s=duration_s, distance_km=distance_km)

    with get_session() as db:
        raw = RawFile(filename=filename, file_metadata=metadata)
        db.add(raw)
        db.flush()
        session = Session(
            raw_file_id=raw.id,
            start_time=start,
            end_time=end,
            duration_s=duration_s,
            distance_km=distance_km,
        )
        db.add(session)
        db.flush()
        db.add_all(
            Trackpoint(
                session_id=session.id,
                time=p.time,
                distance_m=_nan_to_none(p.distance),
                altitude_m=_nan_to_none(p.altitude),
                heart_rate=p.heart_rate,
                cadence=p.cadence,
                power=p.power,
            )
            for p in points
        )
        return str(session.id)
