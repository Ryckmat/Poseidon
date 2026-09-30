"""Enregistrement d'une séance parsée en base."""

import hashlib
import logging
import math
import os
import uuid
from dataclasses import asdict, dataclass

from sqlalchemy import insert

from poseidon.db import RawFile, Session, Trackpoint, get_session
from poseidon.db.repository import find_session_by_filename
from poseidon.ingest.merge import merge_continuous, smooth_boundary_power
from poseidon.ingest.tcx import TrackPoint, last_distance, parse_tcx, time_bounds

log = logging.getLogger(__name__)


class DuplicateSessionError(RuntimeError):
    """Une séance portant ce nom existe déjà."""

    def __init__(self, filename: str, session_id: uuid.UUID):
        super().__init__(f"'{filename}' est déjà importé (séance {session_id})")
        self.filename = filename
        self.session_id = session_id


@dataclass(frozen=True)
class BoundaryFix:
    enabled: bool = True
    window_after_s: float = 3.0
    seek_next_valid_s: float = 6.0
    min_valid_w: float = 20.0


def _nan_to_none(x: float | None) -> float | None:
    return None if x is None or math.isnan(x) else x


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_summary(path: str, points: list[TrackPoint]) -> dict:
    start, end = time_bounds(points)
    return {
        "filename": os.path.basename(path),
        "sha256": _sha256(path),
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "points": len(points),
        "distance_km_device": last_distance(points) / 1000.0,
    }


def default_name(paths: list[str]) -> str:
    basenames = [os.path.basename(p) for p in paths]
    if len(basenames) == 1:
        return basenames[0]
    return f"MERGED_CONTINUOUS:{','.join(basenames)}"


def ingest(
    paths: list[str],
    name: str | None = None,
    boundary_fix: BoundaryFix = BoundaryFix(),
) -> uuid.UUID:
    """Ingère un ou plusieurs TCX comme une seule séance. Retourne son id.

    Un fichier seul est importé tel quel. Plusieurs fichiers sont fusionnés en
    timeline continue (voir merge_continuous), avec correction optionnelle des
    chutes de puissance aux jonctions. Lève DuplicateSessionError si une
    séance du même nom existe déjà.
    """
    if not paths:
        raise ValueError("Aucun fichier fourni")
    filename = name or default_name(paths)
    files = [parse_tcx(p) for p in paths]
    metadata = {
        "files": [_file_summary(p, f) for p, f in zip(paths, files, strict=True)]
    }

    if len(files) == 1:
        points = files[0]
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
        metadata.update(mode="continuous_no_gap", boundary_fix=asdict(boundary_fix))

    start, end = time_bounds(points)
    duration_s = (end - start).total_seconds()
    distance_km = last_distance(points) / 1000.0
    metadata.update(duration_s=duration_s, distance_km=distance_km)

    with get_session() as db:
        existing = find_session_by_filename(db, filename)
        if existing is not None:
            raise DuplicateSessionError(filename, existing)
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
        db.execute(
            insert(Trackpoint),
            [
                {
                    "session_id": session.id,
                    "time": p.time,
                    "distance_m": _nan_to_none(p.distance),
                    "altitude_m": _nan_to_none(p.altitude),
                    "heart_rate": p.heart_rate,
                    "cadence": p.cadence,
                    "power": p.power,
                }
                for p in points
            ],
        )
        log.info(
            "Séance %s importée : %s, %d points, %.0f s, %.2f km",
            session.id,
            filename,
            len(points),
            duration_s,
            distance_km,
        )
        return session.id
