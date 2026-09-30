"""Fusion de plusieurs fichiers TCX d'une même séance en une timeline continue."""

import math
from dataclasses import replace
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from poseidon.ingest.tcx import TrackPoint, time_bounds

_MIN_TIME = datetime.min.replace(tzinfo=timezone.utc)


def merge_continuous(
    files: List[List[TrackPoint]],
) -> Tuple[List[TrackPoint], List[datetime]]:
    """Enchaîne les fichiers sans trou.

    Les fichiers sont ordonnés par heure de début, puis décalés pour démarrer
    à la fin du précédent. Les distances sont rebasées par fichier puis
    cumulées. Retourne (points fusionnés, instants de jonction entre fichiers).
    """
    bounds = [time_bounds(pts) for pts in files]
    order = sorted(range(len(files)), key=lambda i: bounds[i][0] or _MIN_TIME)

    merged: List[TrackPoint] = []
    boundaries: List[datetime] = []
    seen_times = set()
    cursor: Optional[datetime] = None
    distance_offset = 0.0

    for idx in order:
        points = files[idx]
        start, end = bounds[idx]
        if not points:
            continue
        if cursor is None:
            cursor = start
        else:
            boundaries.append(cursor)
        shift = (cursor - start) if (cursor and start) else None

        base = next((p.distance for p in points if not math.isnan(p.distance)), 0.0)
        last_distance = None
        for p in points:
            new_time = p.time + shift if (p.time and shift is not None) else p.time
            new_distance = p.distance
            if not math.isnan(p.distance):
                new_distance = p.distance - base + distance_offset
                last_distance = new_distance
            if new_time is not None and new_time not in seen_times:
                seen_times.add(new_time)
                merged.append(replace(p, time=new_time, distance=new_distance))

        if start and end:
            cursor = cursor + (end - start)
        if last_distance is not None:
            distance_offset = last_distance

    merged.sort(key=lambda p: p.time or _MIN_TIME)
    return merged, sorted(boundaries)


def smooth_boundary_power(
    points: List[TrackPoint],
    boundaries: List[datetime],
    window_after_s: float = 3.0,
    seek_next_valid_s: float = 6.0,
    min_valid_w: float = 20.0,
) -> None:
    """Corrige en place les chutes de puissance vers 0 W aux jonctions.

    Pour chaque jonction B, les points de [B, B + window_after_s] sont
    interpolés linéairement entre le dernier point avant B et le premier point
    de [B, B + seek_next_valid_s] ayant une puissance >= min_valid_w. A défaut
    de point valide après B, la puissance d'avant B est recopiée.
    """

    def valid(power: Optional[float]) -> bool:
        return power is not None and not math.isnan(power)

    timed = [p for p in points if p.time is not None]
    for b in boundaries:
        before = [p for p in timed if p.time < b]
        if not before or not valid(before[-1].power):
            continue
        prev = before[-1]

        def offset(p: TrackPoint) -> float:
            return (p.time - b).total_seconds()

        window = [p for p in timed if 0 <= offset(p) <= window_after_s]
        if not window:
            continue
        nxt = next(
            (
                p
                for p in timed
                if 0 <= offset(p) <= seek_next_valid_s
                and valid(p.power)
                and p.power >= min_valid_w
            ),
            None,
        )
        span = (nxt.time - prev.time).total_seconds() if nxt else 0.0
        for p in window:
            if nxt is None or span <= 0:
                p.power = float(prev.power)
            else:
                w = (p.time - prev.time).total_seconds() / span
                p.power = (1.0 - w) * prev.power + w * nxt.power
