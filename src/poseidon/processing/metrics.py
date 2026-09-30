"""Calculs de séance, sans dépendance à la base ni à l'UI.

Toutes les fonctions travaillent sur un DataFrame trié par `time` avec les
colonnes: time, power, cadence, distance_m, altitude_m (speed_kmh optionnelle).
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from poseidon.config import AnalysisParams

# Zones de puissance (W) affichées dans le dashboard : (code, nom en, nom fr, de, à)
POWER_ZONES = [
    ("Z1", "Active Recovery", "Récup. active", 0, 34),
    ("Z2", "Endurance", "Endurance", 34, 47),
    ("Z3", "Tempo", "Tempo", 47, 56),
    ("Z4", "Threshold", "Seuil", 56, 66),
    ("Z5", "VO2max", "VO2max", 66, 75),
    ("Z6", "Anaerobic", "Anaérobie", 75, 94),
    ("Z7", "Neuromuscular", "Neuromusculaire", 94, 250),
]

BEST_EFFORT_WINDOWS_S = {"5s": 5, "1min": 60, "5min": 300, "20min": 1200}


# ------------------------------------------------------------------ préparation
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute temps écoulé, vitesse, allure et dénivelé instantané."""
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    for col in ("distance_m", "altitude_m", "cadence", "power"):
        if col not in df:
            df[col] = np.nan

    df["elapsed_time_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    dt = df["time"].diff().dt.total_seconds()
    speed_m_s = (df["distance_m"].diff() / dt.where(dt > 0)).fillna(0.0)
    speed_kmh = speed_m_s * 3.6

    # Une vitesse déjà calculée en base est prioritaire, complétée si trous.
    stored = df.get("speed_kmh")
    if stored is None or stored.fillna(0).eq(0).all():
        df["speed_kmh"] = speed_kmh
    else:
        df["speed_kmh"] = stored.fillna(speed_kmh)

    df["pace_min_per_km"] = (1000.0 / 60.0) / speed_m_s.where(speed_m_s > 1e-6)
    df["elevation_diff"] = df["altitude_m"].diff().fillna(0.0)
    return df


def median_sample_s(df: pd.DataFrame) -> float:
    dt = df["elapsed_time_s"].diff()
    median = dt[dt > 0].median()
    return float(median) if pd.notna(median) and median > 0 else 1.0


def points_for(df: pd.DataFrame, seconds: float) -> int:
    """Nombre de points couvrant `seconds` à l'échantillonnage médian."""
    return max(1, int(round(seconds / median_sample_s(df))))


def apply_power_filter(df: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    """Ajoute power_filtered (> max_power retiré) et son écart-type roulant."""
    df["power_filtered"] = df["power"].where(df["power"] <= params.max_power)
    window = points_for(df, params.std_window_s)
    df["power_std"] = df["power_filtered"].rolling(window, min_periods=1).std()
    return df


def prepare(df_raw: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    return apply_power_filter(add_derived(df_raw), params)


# ------------------------------------------------------------ segments stables
def detect_stable_segments(df: pd.DataFrame, params: AnalysisParams) -> List[dict]:
    """Plages continues où la puissance filtrée est haute et stable."""
    stable = (df["power_filtered"] >= params.min_stable_power) & (
        df["power_std"] <= params.std_threshold
    )
    run_id = (stable != stable.shift()).cumsum()
    segments = []
    for _, run in df[stable].groupby(run_id[stable]):
        start, end = run.iloc[0], run.iloc[-1]
        duration = (end["time"] - start["time"]).total_seconds()
        if duration < params.min_stable_duration_s:
            continue
        segments.append(
            {
                "start_time": start["time"],
                "end_time": end["time"],
                "elapsed_time_s_start": start["elapsed_time_s"],
                "elapsed_time_s_end": end["elapsed_time_s"],
                "duration_s": duration,
                "avg_power": run["power_filtered"].mean(),
                "std_power": run["power_filtered"].std(ddof=0),
                "avg_cadence": run["cadence"].mean(),
                "avg_speed_kmh": run["speed_kmh"].mean(),
                "points": len(run),
            }
        )
    return segments


# ------------------------------------------------------------------ régression
@dataclass
class LinearFit:
    slope: float
    intercept: float
    r2: float
    x: np.ndarray
    y_pred: np.ndarray


def linear_fit(x, y) -> Optional[LinearFit]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    xs, ys = x[mask], y[mask]
    if len(xs) < 2 or np.ptp(xs) == 0:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    y_pred = slope * xs + intercept
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - np.sum((ys - y_pred) ** 2) / ss_tot if ss_tot else np.nan
    return LinearFit(float(slope), float(intercept), float(r2), xs, y_pred)


# ---------------------------------------------------------- indicateurs séance
def _time_indexed(power: pd.Series, times: pd.Series) -> pd.Series:
    return pd.Series(power.to_numpy(), index=pd.DatetimeIndex(times))


def estimate_ftp(power: pd.Series, times: pd.Series) -> Optional[float]:
    """95 % de la meilleure moyenne glissante sur 20 minutes."""
    if power.dropna().empty:
        return None
    return 0.95 * _time_indexed(power, times).rolling("20min").mean().max()


def normalized_power(power: pd.Series, times: pd.Series) -> Optional[float]:
    """Moyenne 30 s glissante, puissance 4, moyenne, racine 4."""
    if power.dropna().empty:
        return None
    roll30 = _time_indexed(power, times).rolling("30s").mean()
    fourth = (roll30**4).mean()
    return None if pd.isna(fourth) else float(fourth**0.25)


def training_stress_score(
    np_w: Optional[float], ftp_w: Optional[float], duration_s: Optional[float]
) -> Optional[float]:
    if not np_w or not ftp_w or not duration_s:
        return None
    intensity = np_w / ftp_w
    return duration_s * np_w * intensity / (ftp_w * 3600) * 100


def elevation_gain(df: pd.DataFrame) -> float:
    return float(df["altitude_m"].diff().clip(lower=0).sum())


def best_efforts(df: pd.DataFrame) -> dict:
    """Meilleure puissance moyenne sur chaque fenêtre de BEST_EFFORT_WINDOWS_S."""
    power = df["power_filtered"]
    return {
        label: power.rolling(points_for(df, seconds), min_periods=1).mean().max()
        for label, seconds in BEST_EFFORT_WINDOWS_S.items()
    }


def longest_streak_s(df: pd.DataFrame, low_w: float, high_w: float) -> float:
    """Plus longue durée continue avec low_w <= puissance < high_w."""
    inside = df["power_filtered"].between(low_w, high_w, inclusive="left")
    runs = inside.groupby((inside != inside.shift()).cumsum()).sum()
    return float(runs.max() if len(runs) else 0) * median_sample_s(df)


def time_in_power_zones(df: pd.DataFrame) -> pd.DataFrame:
    edges = [z[3] for z in POWER_ZONES] + [POWER_ZONES[-1][4]]
    zones = pd.cut(
        df["power_filtered"],
        bins=edges,
        labels=[z[0] for z in POWER_ZONES],
        right=False,
    )
    counts = zones.value_counts().reindex([z[0] for z in POWER_ZONES], fill_value=0)
    return pd.DataFrame(
        {
            "zone": [z[0] for z in POWER_ZONES],
            "name_en": [z[1] for z in POWER_ZONES],
            "name_fr": [z[2] for z in POWER_ZONES],
            "from_w": [z[3] for z in POWER_ZONES],
            "to_w": [z[4] for z in POWER_ZONES],
            "seconds": counts.to_numpy() * median_sample_s(df),
        }
    )


@dataclass
class SessionSummary:
    duration_s: float
    distance_km: float
    elevation_gain_m: float
    avg_speed_kmh: Optional[float]
    ftp: Optional[float]
    normalized_power: Optional[float]
    tss: Optional[float]


def summarize(df: pd.DataFrame) -> SessionSummary:
    duration_s = float(df["elapsed_time_s"].iloc[-1]) if len(df) else 0.0
    ftp = estimate_ftp(df["power_filtered"], df["time"])
    np_w = normalized_power(df["power_filtered"], df["time"])
    speed = df["speed_kmh"].dropna()
    return SessionSummary(
        duration_s=duration_s,
        distance_km=float(df["distance_m"].max() or 0) / 1000.0,
        elevation_gain_m=elevation_gain(df),
        avg_speed_kmh=float(speed.mean()) if not speed.empty else None,
        ftp=ftp,
        normalized_power=np_w,
        tss=training_stress_score(np_w, ftp, duration_s),
    )
