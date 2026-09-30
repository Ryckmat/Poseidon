"""Calculs de séance, sans dépendance à la base ni à l'UI.

Les fonctions travaillent sur un DataFrame trié par `time` avec les colonnes
time, power, cadence, distance_m, altitude_m, et optionnellement heart_rate et
speed_kmh. La cadence est celle du rameur, en coups par minute.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from poseidon.config import AnalysisParams

# Zones de puissance (W) : (code, nom en, nom fr, de, à)
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

_OPTIONAL_COLUMNS = ("distance_m", "altitude_m", "cadence", "power", "heart_rate")


# ------------------------------------------------------------------ préparation
def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute temps écoulé, vitesse, allures, distance par coup et dénivelé."""
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    for col in _OPTIONAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan

    df["elapsed_time_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    dt = df["time"].diff().dt.total_seconds()
    speed_m_s = df["distance_m"].diff() / dt.where(dt > 0)
    speed_m_s = speed_m_s.where(speed_m_s >= 0)

    # Une vitesse déjà calculée en base est prioritaire, complétée si trous.
    stored = df.get("speed_kmh")
    if stored is None or pd.to_numeric(stored).fillna(0).eq(0).all():
        df["speed_kmh"] = speed_m_s * 3.6
    else:
        df["speed_kmh"] = pd.to_numeric(stored).fillna(speed_m_s * 3.6)

    moving = df["speed_kmh"].where(df["speed_kmh"] > 0.01) / 3.6
    df["pace_min_per_km"] = (1000.0 / 60.0) / moving
    df["split_500m_s"] = 500.0 / moving
    df["distance_per_stroke_m"] = moving * 60.0 / df["cadence"].where(df["cadence"] > 0)
    df["elevation_diff"] = df["altitude_m"].diff().fillna(0.0)
    return df


def median_sample_s(df: pd.DataFrame) -> float:
    dt = df["elapsed_time_s"].diff()
    median = dt[dt > 0].median()
    return float(median) if pd.notna(median) and median > 0 else 1.0


def points_for(df: pd.DataFrame, seconds: float) -> int:
    """Nombre de points couvrant `seconds` à l'échantillonnage médian."""
    return max(1, round(seconds / median_sample_s(df)))


def apply_power_filter(df: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    """Ajoute power_filtered (> max_power retiré) et son écart-type glissant."""
    df["power_filtered"] = df["power"].where(df["power"] <= params.max_power)
    window = points_for(df, params.std_window_s)
    df["power_std"] = df["power_filtered"].rolling(window, min_periods=1).std()
    return df


def prepare(df_raw: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    if df_raw.empty:
        raise ValueError("Aucun point à analyser")
    return apply_power_filter(add_derived(df_raw), params)


# ------------------------------------------------------------ segments stables
def detect_stable_segments(df: pd.DataFrame, params: AnalysisParams) -> list[dict]:
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


def linear_fit(x, y) -> LinearFit | None:
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
def rolling_mean_full(power: pd.Series, times: pd.Series, seconds: float) -> pd.Series:
    """Moyenne glissante sur `seconds`, limitée aux fenêtres complètes.

    Les premiers points n'ont pas encore `seconds` d'historique : leur moyenne
    porterait sur une fenêtre partielle (un pic de quelques secondes au
    démarrage deviendrait la « meilleure moyenne 20 min »). Ils sont exclus.
    """
    index = pd.DatetimeIndex(times)
    series = pd.Series(power.to_numpy(dtype=float), index=index)
    rolled = series.rolling(pd.Timedelta(seconds=seconds)).mean()
    elapsed = (index - index[0]).total_seconds().to_numpy()
    steps = np.diff(elapsed)
    sample = float(np.median(steps[steps > 0])) if (steps > 0).any() else 1.0
    return rolled[elapsed >= seconds - sample]


def _max_or_none(series: pd.Series) -> float | None:
    value = series.max()
    return None if pd.isna(value) else float(value)


def estimate_ftp(power: pd.Series, times: pd.Series) -> float | None:
    """95 % de la meilleure moyenne sur 20 minutes (None si séance plus courte)."""
    best = _max_or_none(rolling_mean_full(power, times, 1200))
    return None if best is None else 0.95 * best


def normalized_power(power: pd.Series, times: pd.Series) -> float | None:
    """Moyenne glissante 30 s, puissance 4, moyenne, racine 4."""
    roll30 = rolling_mean_full(power, times, 30)
    fourth = (roll30**4).mean()
    return None if pd.isna(fourth) else float(fourth**0.25)


def training_stress_score(
    np_w: float | None, ftp_w: float | None, duration_s: float | None
) -> float | None:
    if not np_w or not ftp_w or not duration_s:
        return None
    intensity = np_w / ftp_w
    return duration_s * np_w * intensity / (ftp_w * 3600) * 100


def elevation_gain(df: pd.DataFrame) -> float:
    return float(df["altitude_m"].diff().clip(lower=0).sum())


def best_efforts(df: pd.DataFrame) -> dict:
    """Meilleure puissance moyenne sur chaque durée de BEST_EFFORT_WINDOWS_S
    (None si la séance est plus courte que la durée)."""
    return {
        label: _max_or_none(
            rolling_mean_full(df["power_filtered"], df["time"], seconds)
        )
        for label, seconds in BEST_EFFORT_WINDOWS_S.items()
    }


def longest_streak_s(df: pd.DataFrame, low_w: float, high_w: float) -> float:
    """Plus longue durée continue avec low_w <= puissance < high_w."""
    inside = df["power_filtered"].between(low_w, high_w, inclusive="left")
    runs = inside.groupby((inside != inside.shift()).cumsum()).sum()
    return float(runs.max() if len(runs) else 0) * median_sample_s(df)


def time_in_power_zones(df: pd.DataFrame) -> pd.DataFrame:
    edges = [z[3] for z in POWER_ZONES] + [POWER_ZONES[-1][4]]
    codes = [z[0] for z in POWER_ZONES]
    zones = pd.cut(df["power_filtered"], bins=edges, labels=codes, right=False)
    counts = zones.value_counts().reindex(codes, fill_value=0)
    return pd.DataFrame(
        {
            "zone": codes,
            "name_en": [z[1] for z in POWER_ZONES],
            "name_fr": [z[2] for z in POWER_ZONES],
            "from_w": [z[3] for z in POWER_ZONES],
            "to_w": [z[4] for z in POWER_ZONES],
            "seconds": counts.to_numpy() * median_sample_s(df),
        }
    )


def _mean(series: pd.Series) -> float | None:
    value = series.mean()
    return None if pd.isna(value) else float(value)


@dataclass
class SessionSummary:
    duration_s: float
    distance_km: float
    elevation_gain_m: float
    avg_speed_kmh: float | None
    avg_split_500m_s: float | None
    avg_power: float | None
    avg_cadence: float | None
    avg_distance_per_stroke_m: float | None
    avg_heart_rate: float | None
    ftp: float | None
    normalized_power: float | None
    tss: float | None


def summarize(df: pd.DataFrame, reference_ftp_w: float | None = None) -> SessionSummary:
    """Indicateurs de séance. Le TSS utilise la FTP de référence si fournie,
    sinon la FTP estimée sur la séance elle-même."""
    duration_s = float(df["elapsed_time_s"].iloc[-1])
    max_distance = df["distance_m"].max()
    distance_km = 0.0 if pd.isna(max_distance) else float(max_distance) / 1000.0
    avg_speed = _mean(df["speed_kmh"])
    ftp = estimate_ftp(df["power_filtered"], df["time"])
    np_w = normalized_power(df["power_filtered"], df["time"])
    return SessionSummary(
        duration_s=duration_s,
        distance_km=distance_km,
        elevation_gain_m=elevation_gain(df),
        avg_speed_kmh=avg_speed,
        avg_split_500m_s=500.0 / (avg_speed / 3.6) if avg_speed else None,
        avg_power=_mean(df["power_filtered"]),
        avg_cadence=_mean(df["cadence"]),
        avg_distance_per_stroke_m=_mean(df["distance_per_stroke_m"]),
        avg_heart_rate=_mean(df["heart_rate"]),
        ftp=ftp,
        normalized_power=np_w,
        tss=training_stress_score(np_w, reference_ftp_w or ftp, duration_s),
    )
