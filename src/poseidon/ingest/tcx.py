"""Lecture des fichiers TCX (Garmin Training Center)."""

import math
import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from defusedxml import DefusedXmlException
from defusedxml.ElementTree import parse as safe_parse

NS = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}


class TcxError(ValueError):
    """Fichier TCX illisible ou vide."""


@dataclass
class TrackPoint:
    time: datetime | None
    altitude: float = math.nan
    distance: float = math.nan
    heart_rate: int | None = None
    cadence: int | None = None
    power: float | None = None


def _text(elem: ET.Element | None) -> str | None:
    if elem is None or not elem.text or not elem.text.strip():
        return None
    return elem.text.strip()


def _float(elem: ET.Element | None) -> float:
    txt = _text(elem)
    try:
        return float(txt) if txt else math.nan
    except ValueError:
        return math.nan


def _int(elem: ET.Element | None) -> int | None:
    value = _float(elem)
    return None if math.isnan(value) else int(value)


def _parse_time(elem: ET.Element | None) -> datetime | None:
    txt = _text(elem)
    if not txt:
        return None
    try:
        parsed = datetime.fromisoformat(txt.replace("Z", "+00:00"))
    except ValueError:
        return None
    # Un horodatage sans fuseau est considéré en UTC (norme TCX).
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _parse_power(tp: ET.Element) -> float | None:
    """La puissance est dans les Extensions (balise Watts ou *Power*)."""
    ext = tp.find("ns:Extensions", NS)
    if ext is None:
        return None
    power = None
    for child in ext.iter():
        tag = child.tag.lower()
        if "watts" in tag or "power" in tag:
            value = _float(child)
            if not math.isnan(value):
                power = value
    return power


def parse_tcx(path: str) -> list[TrackPoint]:
    """Lit les points d'un TCX. Lève TcxError si le fichier est inexploitable."""
    try:
        root = safe_parse(path).getroot()
    except (ET.ParseError, DefusedXmlException) as exc:
        raise TcxError(f"{path} : XML invalide ({exc})") from exc
    except OSError as exc:
        raise TcxError(f"{path} : lecture impossible ({exc.strerror})") from exc

    points = [
        TrackPoint(
            time=_parse_time(tp.find("ns:Time", NS)),
            altitude=_float(tp.find("ns:AltitudeMeters", NS)),
            distance=_float(tp.find("ns:DistanceMeters", NS)),
            heart_rate=_int(tp.find(".//ns:HeartRateBpm/ns:Value", NS)),
            cadence=_int(tp.find("ns:Cadence", NS)),
            power=_parse_power(tp),
        )
        for tp in root.findall(".//ns:Trackpoint", NS)
    ]
    points = [p for p in points if p.time is not None]
    if not points:
        raise TcxError(f"{path} : aucun point horodaté (fichier TCX attendu)")
    return points


def expand_paths(paths: Iterable[str]) -> list[str]:
    """Remplace chaque dossier par ses fichiers .tcx triés, sans doublon."""
    result: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            found = sorted(str(p) for p in Path(path).glob("*.tcx"))
            if not found:
                raise TcxError(f"{path} : aucun fichier .tcx dans ce dossier")
            result.extend(found)
        else:
            result.append(path)
    return list(dict.fromkeys(result))


def time_bounds(points: list[TrackPoint]):
    times = [p.time for p in points if p.time is not None]
    return (min(times), max(times)) if times else (None, None)


def last_distance(points: list[TrackPoint]) -> float:
    """Distance cumulée finale en mètres (0 si absente)."""
    distances = [p.distance for p in points if not math.isnan(p.distance)]
    return max(distances) if distances else 0.0
