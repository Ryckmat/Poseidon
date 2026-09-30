"""Mise en forme des valeurs affichées."""

import math

MISSING = "-"


def _number(value) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(num) or math.isinf(num) else num


def hhmmss(seconds) -> str:
    num = _number(seconds)
    total = int(num) if num is not None else 0
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"


def human_duration(seconds) -> str:
    num = _number(seconds)
    if num is None:
        return MISSING
    remaining = int(num)
    parts = []
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        value, remaining = divmod(remaining, size)
        if value:
            parts.append(f"{value}{unit}")
    parts.append(f"{remaining}s")
    return " ".join(parts)


def split(seconds) -> str:
    """Allure au 500 m, format m:ss.d (convention des ergomètres)."""
    num = _number(seconds)
    if num is None or num <= 0:
        return MISSING
    minutes, secs = divmod(num, 60)
    return f"{int(minutes)}:{secs:04.1f}"


def fmt(value, pattern: str = "{:.2f}") -> str:
    num = _number(value)
    return MISSING if num is None else pattern.format(num)
