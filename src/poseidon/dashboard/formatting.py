"""Mise en forme des valeurs affichées."""

import math
from typing import Optional


def hhmmss(seconds) -> str:
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "00:00:00"
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"


def human_duration(seconds) -> str:
    try:
        seconds = int(float(seconds))
    except (TypeError, ValueError):
        return "—"
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        value, seconds = divmod(seconds, size)
        if value:
            parts.append(f"{value}{unit}")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def fmt(value: Optional[float], pattern: str = "{:.2f}", fallback: str = "—") -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return fallback
    return fallback if math.isnan(num) or math.isinf(num) else pattern.format(num)
