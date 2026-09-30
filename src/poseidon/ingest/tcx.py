"""Lecture des fichiers TCX (Garmin Training Center)."""

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

NS = {"ns": "http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"}


@dataclass
class TrackPoint:
    time: Optional[datetime]
    altitude: float = math.nan
    distance: float = math.nan
    heart_rate: Optional[int] = None
    cadence: Optional[int] = None
    power: Optional[float] = None


def _text(elem: Optional[ET.Element]) -> Optional[str]:
    if elem is None or not elem.text:
        return None
    return elem.text.strip()


def _float(elem: Optional[ET.Element]) -> float:
    txt = _text(elem)
    return float(txt) if txt else math.nan


def _int(elem: Optional[ET.Element]) -> Optional[int]:
    txt = _text(elem)
    return int(float(txt)) if txt else None


def _parse_time(elem: Optional[ET.Element]) -> Optional[datetime]:
    txt = _text(elem)
    if not txt:
        return None
    return datetime.fromisoformat(txt.replace("Z", "+00:00"))


def _parse_power(tp: ET.Element) -> Optional[float]:
    """La puissance est dans les Extensions (balise Watts ou *Power*)."""
    ext = tp.find("ns:Extensions", NS)
    if ext is None:
        return None
    power = None
    for child in ext.iter():
        tag = child.tag.lower()
        if ("watts" in tag or "power" in tag) and _text(child):
            try:
                power = float(child.text)
            except ValueError:
                continue
    return power


def parse_tcx(path: str) -> List[TrackPoint]:
    root = ET.parse(path).getroot()
    return [
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


def time_bounds(points: List[TrackPoint]):
    times = [p.time for p in points if p.time is not None]
    return (min(times), max(times)) if times else (None, None)


def last_distance(points: List[TrackPoint]) -> float:
    """Distance cumulée finale en mètres (0 si absente)."""
    distances = [p.distance for p in points if not math.isnan(p.distance)]
    return max(distances) if distances else 0.0
