import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import select

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from db.models import Session, SessionLocal, Trackpoint


def format_seconds_to_hhmmss(seconds):
    try:
        seconds = int(seconds)
    except Exception:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02}:{m:02}:{s:02}"


DEFAULT_ZONES = [0, 100, 150, 200, 250, 300, 400, 500, 9999]


def compute_zones(power_series, zones=DEFAULT_ZONES):
    counts, _ = np.histogram(power_series.dropna(), bins=zones)
    times = counts
    total = times.sum()
    pct = times / total * 100 if total > 0 else np.zeros_like(times)
    labels = [
        f"{zones[i]}–{zones[i+1]-1}W" if zones[i + 1] < 9999 else f"{zones[i]}+ W"
        for i in range(len(zones) - 1)
    ]
    return pd.DataFrame(
        {"zone": labels, "time_in_zone_s": times, "percent_time_in_zone": pct}
    )


def compute_cv(series):
    vals = series.dropna()
    if len(vals) < 2:
        return np.nan
    return np.std(vals) / np.mean(vals) if np.mean(vals) != 0 else np.nan


def moving_average(series, window_size_pts=60):
    return series.rolling(window=window_size_pts, min_periods=1, center=True).mean()


def compare_stable_segments(primary, compare):
    if not primary or not compare:
        return pd.DataFrame()
    primary_sorted = sorted(primary, key=lambda x: -x["duration_s"])
    compare_sorted = sorted(compare, key=lambda x: -x["duration_s"])
    rows = []
    for i in range(min(len(primary_sorted), len(compare_sorted))):
        p = primary_sorted[i]
        c = compare_sorted[i]
        rows.append(
            {
                "Primary avg power": p["avg_power"],
                "Primary duration": p["duration_s"],
                "Compare avg power": c["avg_power"],
                "Compare duration": c["duration_s"],
                "Delta power": p["avg_power"] - c["avg_power"],
                "Delta duration": p["duration_s"] - c["duration_s"],
            }
        )
    return pd.DataFrame(rows)


# ----------- Internationalization (i18n) -----------
LANGUAGES = {
    "en": {
        "flag": "🇬🇧",
        "language": "Language",
        "title": "Poseidon — Session Overview",
        "primary_session": "Primary session",
        "compare_to_optional": "Compare to (optional)",
        "controls": "Controls",
        "stable_params": "Stable segment & filtering params",
        "power_threshold": "Power max threshold (filter)",
        "min_stable_power": "Min power for stable segment",
        "rolling_std_window": "Rolling std window (s)",
        "std_threshold": "Std threshold for stability",
        "min_segment_duration": "Min segment duration (s)",
        "duration": "Duration",
        "distance": "Distance (km)",
        "elevation_gain": "Elevation Gain (m)",
        "avg_speed": "Avg Speed (km/h)",
        "ftp_est": "FTP Est.",
        "normalized_power": "Normalized Power",
        "tss": "TSS",
        "power_filtering": "Power filtering",
        "time_series": "Time Series",
        "power_over_time": "Power over Time",
        "cadence_over_time": "Cadence over Time",
        "speed_over_time": "Speed over Time",
        "correlations": "Correlations & Regression",
        "power_vs_cadence": "Power vs Cadence",
        "power_vs_speed": "Power vs Speed",
        "stable_segments": "Stable segments (primary)",
        "export": "Export",
        "cleaned_trackpoints": "Download cleaned trackpoints CSV",
        "download_pdf": "Download PDF Report",
        "comparison_summary": "Comparison summary",
        "primary": "Primary",
        "compare": "Compare",
        "delta_distance": "Delta distance",
        "delta_avg_speed": "Delta avg speed",
        "no_sessions": "No sessions found in database.",
        "no_trackpoints": "Primary session has no trackpoints.",
        "no_stable": "No stable segments detected with current parameters.",
        "filter_info": "threshold = {threshold:.1f} W → removed {removed} point(s) ({percent:.1f}%)",
        "tooltip_ftp": "Estimated from best 20-minute average × 0.95",
        "tooltip_np": "Normalized Power: 30s rolling average to the 4th power",
        "tooltip_tss": "Training Stress Score approximate",
        "select_segment": "Zoom on segment",
        "self_compare_warning": "Comparison session is the same as primary; ignored.",
        "progression": "Progression",
        "weekly_trends": "Weekly Trends",
        "training_load": "Training Load (TSS)",
        "ftp_trend": "FTP Trend",
        "np_trend": "Normalized Power Trend",
        "preset_save": "Save current preset",
        "preset_load": "Load preset",
        "preset_name": "Preset name",
        "preset_select": "Saved presets",
        "reset_params": "Reset filters",
        "advanced_tab": "Advanced analysis",
        "zones": "Power zones",
        "cv": "Coefficient of Variation (CV)",
        "compare_stable_segments": "Compare stable segments (top 3 by duration)",
        "full_export_csv": "Export all session data (CSV)",
        "full_export_pdf": "Export full PDF summary",
        "advanced": "Advanced",
        "Descriptive Statistics": "Descriptive Statistics",
        "Mean": "Mean",
        "Median": "Median",
        "Min": "Min",
        "Max": "Max",
        "Std": "Std",
        "Dispersion (Boxplots)": "Dispersion (Boxplots)",
        "Time spent in Power & Cadence Zones": "Time spent in Power & Cadence Zones",
        "Power zones (distribution):": "Power zones (distribution):",
        "Cadence zones (distribution):": "Cadence zones (distribution):",
        "Best average power (5s, 1min, 5min, 20min)": "Best average power (5s, 1min, 5min, 20min)",
        "Cadence variability": "Cadence variability",
        "Distribution across cadence zones above": "Distribution across cadence zones above",
        "Compare Distributions": "Compare Distributions",
        "Select a comparison session to show comparison histograms.": "Select a comparison session to show comparison histograms.",
        "Longest streak above 200W": "Longest streak above 200W",
        "Max streak: {duration} above {threshold}W": "Max streak: {duration} above {threshold}W",
        "Power (W)": "Power (W)",
        "Cadence (rpm)": "Cadence (rpm)",
        "Speed (km/h)": "Speed (km/h)",
    },
    "fr": {
        "flag": "🇫🇷",
        "language": "Langue",
        "title": "Poseidon — Vue de séance",
        "primary_session": "Séance principale",
        "compare_to_optional": "Comparer avec (optionnel)",
        "controls": "Contrôles",
        "stable_params": "Paramètres de segment stable et filtrage",
        "power_threshold": "Seuil max puissance (filtre)",
        "min_stable_power": "Puissance min pour segment stable",
        "rolling_std_window": "Fenêtre écart-type roulante (s)",
        "std_threshold": "Seuil écart-type pour stabilité",
        "min_segment_duration": "Durée min du segment (s)",
        "duration": "Durée",
        "distance": "Distance (km)",
        "elevation_gain": "Dénivelé (m)",
        "avg_speed": "Vitesse moy. (km/h)",
        "ftp_est": "FTP estimé",
        "normalized_power": "Puissance normalisée",
        "tss": "TSS",
        "power_filtering": "Filtrage de puissance",
        "time_series": "Séries temporelles",
        "power_over_time": "Puissance dans le temps",
        "cadence_over_time": "Cadence dans le temps",
        "speed_over_time": "Vitesse dans le temps",
        "correlations": "Corrélations & Régressions",
        "power_vs_cadence": "Puissance vs Cadence",
        "power_vs_speed": "Puissance vs Vitesse",
        "stable_segments": "Segments stables (principal)",
        "export": "Export",
        "cleaned_trackpoints": "Télécharger CSV nettoyé",
        "download_pdf": "Télécharger rapport PDF",
        "comparison_summary": "Résumé comparaison",
        "primary": "Principal",
        "compare": "Comparer",
        "delta_distance": "Delta distance",
        "delta_avg_speed": "Delta vitesse moy.",
        "no_sessions": "Aucune séance trouvée dans la base.",
        "no_trackpoints": "La séance n'a pas de points.",
        "no_stable": "Aucun segment stable détecté avec les paramètres actuels.",
        "filter_info": "seuil = {threshold:.1f} W → {removed} point(s) supprimé(s) ({percent:.1f}%)",
        "tooltip_ftp": "Estimé depuis la meilleure moyenne sur 20 minutes × 0.95",
        "tooltip_np": "Puissance normalisée : moyenne roulante 30s à la puissance 4",
        "tooltip_tss": "Score de charge d'entraînement approximatif",
        "select_segment": "Zoom sur segment",
        "self_compare_warning": "La séance de comparaison est la même que la principale ; ignorée.",
        "progression": "Progression",
        "weekly_trends": "Tendances hebdo",
        "training_load": "Charge d'entraînement (TSS)",
        "ftp_trend": "Tendance FTP",
        "np_trend": "Tendance NP",
        "preset_save": "Sauvegarder preset",
        "preset_load": "Charger preset",
        "preset_name": "Nom du preset",
        "preset_select": "Presets enregistrés",
        "reset_params": "Réinitialiser filtres",
        "advanced_tab": "Analyse avancée",
        "zones": "Zones de puissance",
        "cv": "Coefficient de Variation (CV)",
        "compare_stable_segments": "Comparatif des segments stables (top 3 par durée)",
        "full_export_csv": "Exporter toutes les données de séance (CSV)",
        "full_export_pdf": "Exporter le résumé PDF complet",
        "advanced": "Avancé",
        "Descriptive Statistics": "Statistiques descriptives",
        "Mean": "Moyenne",
        "Median": "Médiane",
        "Min": "Min",
        "Max": "Max",
        "Std": "Écart-type",
        "Dispersion (Boxplots)": "Dispersion (Boxplots)",
        "Time spent in Power & Cadence Zones": "Temps passé par zone de Puissance & Cadence",
        "Power zones (distribution):": "Zones de puissance (distribution):",
        "Cadence zones (distribution):": "Zones de cadence (distribution):",
        "Best average power (5s, 1min, 5min, 20min)": "Meilleurs efforts moyens (5s, 1min, 5min, 20min)",
        "Cadence variability": "Variabilité cadence",
        "Distribution across cadence zones above": "Distribution entre les zones de cadence ci-dessus",
        "Compare Distributions": "Comparer les distributions",
        "Select a comparison session to show comparison histograms.": "Sélectionnez une séance de comparaison pour voir les histogrammes.",
        "Longest streak above 200W": "Plus longue séquence >200W",
        "Max streak: {duration} above {threshold}W": "Séquence max : {duration} au-dessus de {threshold}W",
        "Power (W)": "Puissance (W)",
        "Cadence (rpm)": "Cadence (rpm)",
        "Speed (km/h)": "Vitesse (km/h)",
    },
}

# ----------- Utility functions for data conversion -----------


def to_number(x):
    """Convert to float or return None if conversion fails."""
    if x is None:
        return None
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def metric_value(val, fmt="{:.2f}", fallback="—"):
    """Format a numeric value for display with fallback."""
    num = to_number(val)
    if num is None or (isinstance(num, float) and (np.isnan(num) or np.isinf(num))):
        return fallback
    return fmt.format(num)


def human_duration(sec):
    """Display seconds as human readable duration (Xd Xh Xm Xs)."""
    if sec is None:
        return "—"
    try:
        sec = int(float(sec))
    except (ValueError, TypeError):
        return "—"
    parts = []
    days, rem = divmod(sec, 86400)
    if days:
        parts.append(f"{days}d")
    hrs, rem = divmod(rem, 3600)
    if hrs:
        parts.append(f"{hrs}h")
    mins, secs = divmod(rem, 60)
    if mins:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def compute_derived_df(df_raw):
    """Compute extra columns needed for plotting and analysis."""
    df = df_raw.copy()
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    df["elapsed_time_s"] = (df["time"] - df["time"].iloc[0]).dt.total_seconds()
    df["time_diff_s"] = df["time"].diff().dt.total_seconds().fillna(0)
    df["delta_dist"] = df["distance_m"].diff().fillna(0)
    df["speed_m_s"] = np.where(
        df["time_diff_s"] > 0,
        df["delta_dist"] / df["time_diff_s"],
        np.nan,
    )
    df["speed_kmh"] = df["speed_m_s"] * 3.6
    df["pace_min_per_km"] = np.where(
        df["speed_m_s"] > 0,
        (1 / df["speed_m_s"]) / 60 * 1000,
        np.nan,
    )
    df["elevation_diff"] = df["altitude_m"].diff().fillna(0)
    return df


def rolling_std(arr, window_pts):
    """Rolling std for an array using pandas."""
    return pd.Series(arr).rolling(window=window_pts, min_periods=1).std().to_numpy()


def detect_stable_segments(
    df,
    power_col="power_filtered",
    std_col="power_std",
    min_power=50,
    std_threshold=5,
    min_duration_s=60,
):
    """
    Identify stable segments in the time series based on power and std.
    Returns a list of dicts for each stable segment.
    """
    mask = (
        (~df[power_col].isna())
        & (df[power_col] >= min_power)
        & (df[std_col] <= std_threshold)
    )
    segments = []
    current = None
    for idx, flag in enumerate(mask):
        if flag:
            if current is None:
                current = {"start": idx, "end": idx}
            else:
                current["end"] = idx
        else:
            if current:
                segments.append(current)
                current = None
    if current:
        segments.append(current)
    final = []
    for seg in segments:
        start = seg["start"]
        end = seg["end"]
        start_time = df.loc[start, "time"]
        end_time = df.loc[end, "time"]
        start_elapsed = df.loc[start, "elapsed_time_s"]
        end_elapsed = df.loc[end, "elapsed_time_s"]
        duration = (end_time - start_time).total_seconds()
        if duration >= min_duration_s:
            seg_df = df.loc[start : end + 1]
            final.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "start_time": start_time,
                    "end_time": end_time,
                    "elapsed_time_s_start": start_elapsed,
                    "elapsed_time_s_end": end_elapsed,
                    "duration_s": duration,
                    "avg_power": np.nanmean(seg_df[power_col]),
                    "std_power": np.nanstd(seg_df[power_col]),
                    "avg_cadence": np.nanmean(seg_df.get("cadence", np.nan)),
                    "avg_speed_kmh": np.nanmean(seg_df.get("speed_kmh", np.nan)),
                    "points": end - start + 1,
                }
            )
    return final


def regression_with_ci(x, y, n_boot=200, ci=0.95):
    """Linear regression with confidence intervals (bootstrap)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    xs = x[mask]
    ys = y[mask]
    if len(xs) < 2:
        return None
    coeffs = np.polyfit(xs, ys, 1)
    y_pred = np.polyval(coeffs, xs)
    slopes = []
    intercepts = []
    r2s = []
    rng = np.random.default_rng()
    for _ in range(n_boot):
        idxs = rng.integers(0, len(xs), len(xs))
        xs_sample = xs[idxs]
        ys_sample = ys[idxs]
        try:
            c = np.polyfit(xs_sample, ys_sample, 1)
            y_s_pred = np.polyval(c, xs_sample)
            ss_res = np.sum((ys_sample - y_s_pred) ** 2)
            ss_tot = np.sum((ys_sample - np.mean(ys_sample)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            slopes.append(c[0])
            intercepts.append(c[1])
            r2s.append(r2)
        except Exception:
            continue

    def percentile(arr, p):
        return (
            np.nanpercentile(arr, [(1 - p) / 2 * 100, (1 + p) / 2 * 100])
            if len(arr) > 0
            else (np.nan, np.nan)
        )

    slope_ci = percentile(slopes, ci)
    intercept_ci = percentile(intercepts, ci)
    r2_mean = np.nanmean(r2s) if r2s else np.nan
    return {
        "slope": coeffs[0],
        "intercept": coeffs[1],
        "r2": 1 - np.sum((ys - y_pred) ** 2) / np.sum((ys - np.mean(ys)) ** 2)
        if np.sum((ys - np.mean(ys)) ** 2) != 0
        else np.nan,
        "x": xs,
        "y_pred": y_pred,
        "slope_ci": slope_ci,
        "intercept_ci": intercept_ci,
        "r2_bootstrap_mean": r2_mean,
    }


# ---------- Streamlit cache for database ----------
@st.cache_data(ttl=300)
def load_sessions():
    db = SessionLocal()
    try:
        return (
            db.execute(select(Session).order_by(Session.start_time.desc()))
            .scalars()
            .all()
        )
    finally:
        db.close()


@st.cache_data(ttl=300)
def load_trackpoints(session_id):
    db = SessionLocal()
    try:
        return (
            db.execute(
                select(Trackpoint)
                .where(Trackpoint.session_id == session_id)
                .order_by(Trackpoint.time)
            )
            .scalars()
            .all()
        )
    finally:
        db.close()


def build_df_from_trackpoints(tps):
    """Turn a list of ORM trackpoints into a DataFrame."""
    return pd.DataFrame(
        [
            {
                "time": tp.time,
                "power": to_number(tp.power),
                "power_filtered": to_number(tp.power_filtered),
                "cadence": to_number(tp.cadence),
                "speed_kmh": to_number(tp.speed_calc_kmh),
                "pace_min_per_km": to_number(tp.pace_min_per_km),
                "altitude_m": to_number(tp.altitude_m),
                "distance_m": to_number(tp.distance_m),
            }
            for tp in tps
        ]
    )


# --------- Preset state (in-memory for now) ----------
if "presets" not in st.session_state:
    st.session_state.presets = {}


def save_preset(name, params):
    st.session_state.presets[name] = params


def load_preset(name):
    return st.session_state.presets.get(name, {})


def sanitize_text(s: str) -> str:
    """Sanitize unicode for PDF export (e.g., dash, quotes)."""
    if not isinstance(s, str):
        return s
    replacements = {
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "…": "...",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def main():
    global TRANSLATIONSz
    load_dotenv()
    st.set_page_config(
        page_title="Poseidon Dashboard", layout="wide", initial_sidebar_state="expanded"
    )

    # Language and translations
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    lang_choice = st.sidebar.selectbox(
        f"{LANGUAGES[st.session_state.lang]['flag']} {LANGUAGES[st.session_state.lang]['language']}",
        ["en", "fr"],
        index=0 if st.session_state.lang == "en" else 1,
    )
    st.session_state.lang = lang_choice
    TRANSLATIONS = LANGUAGES.get(lang_choice, LANGUAGES["en"])

    st.sidebar.title(f"{TRANSLATIONS['flag']} {TRANSLATIONS['controls']}")
    st.title(f"{TRANSLATIONS['title']}")

    sessions = load_sessions()
    if not sessions:
        st.warning(TRANSLATIONS["no_sessions"])
        return

    session_map = {f"{s.start_time} | {s.id}": s for s in sessions}
    keys = list(session_map.keys())
    primary_choice = st.sidebar.selectbox(
        TRANSLATIONS["primary_session"], keys, index=0
    )
    compare_choice = st.sidebar.selectbox(
        TRANSLATIONS["compare_to_optional"], ["None"] + keys, index=0
    )

    st.sidebar.markdown(f"### {TRANSLATIONS['stable_params']}")
    preset_names = list(st.session_state.presets.keys())
    selected_preset = None
    if preset_names:
        selected_preset = st.sidebar.selectbox(
            TRANSLATIONS["preset_select"], ["None"] + preset_names
        )
        if selected_preset and selected_preset != "None":
            preset = load_preset(selected_preset)
        else:
            preset = {}
    else:
        preset = {}

    max_power_threshold = st.sidebar.number_input(
        TRANSLATIONS["power_threshold"],
        value=preset.get("max_power_threshold", 250.0),
        step=10.0,
    )
    min_stable_power = st.sidebar.number_input(
        TRANSLATIONS["min_stable_power"],
        value=preset.get("min_stable_power", 50.0),
        step=5.0,
    )
    std_window_s = st.sidebar.number_input(
        TRANSLATIONS["rolling_std_window"], value=preset.get("std_window_s", 30), step=5
    )
    std_thresh = st.sidebar.number_input(
        TRANSLATIONS["std_threshold"], value=preset.get("std_thresh", 5.0), step=0.5
    )
    min_segment_duration = st.sidebar.number_input(
        TRANSLATIONS["min_segment_duration"],
        value=preset.get("min_segment_duration", 60),
        step=10,
    )

    colp1, colp2 = st.sidebar.columns([2, 1])
    with colp1:
        new_name = st.text_input(TRANSLATIONS["preset_name"], value="")
    with colp2:
        if st.button(TRANSLATIONS["preset_save"]) and new_name:
            save_preset(
                new_name,
                {
                    "max_power_threshold": max_power_threshold,
                    "min_stable_power": min_stable_power,
                    "std_window_s": std_window_s,
                    "std_thresh": std_thresh,
                    "min_segment_duration": min_segment_duration,
                },
            )
    if preset_names:
        if (
            st.button(TRANSLATIONS["preset_load"])
            and selected_preset
            and selected_preset != "None"
        ):
            st.experimental_rerun()
    if st.button(TRANSLATIONS["reset_params"]):
        st.experimental_rerun()

    primary_session = session_map[primary_choice]
    primary_tps = load_trackpoints(primary_session.id)
    df_primary_raw = build_df_from_trackpoints(primary_tps)
    if df_primary_raw.empty:
        st.error(TRANSLATIONS["no_trackpoints"])
        return
    df_primary = compute_derived_df(df_primary_raw)

    df_primary["power_filtered"] = np.where(
        (~df_primary["power"].isna()) & (df_primary["power"] <= max_power_threshold),
        df_primary["power"],
        np.nan,
    )
    median_dt = df_primary["time"].diff().dt.total_seconds().replace(0, np.nan).median()
    window_pts = (
        max(1, int(round(std_window_s / median_dt)))
        if pd.notna(median_dt) and median_dt > 0
        else 5
    )
    df_primary["power_std"] = rolling_std(df_primary["power_filtered"], window_pts)
    stable_segments = detect_stable_segments(
        df_primary,
        power_col="power_filtered",
        std_col="power_std",
        min_power=min_stable_power,
        std_threshold=std_thresh,
        min_duration_s=min_segment_duration,
    )
    reg_pc = regression_with_ci(
        df_primary["power_filtered"].to_numpy(), df_primary["cadence"].to_numpy()
    )
    reg_ps = regression_with_ci(
        df_primary["power_filtered"].to_numpy(), df_primary["speed_kmh"].to_numpy()
    )

    compare_session = None
    df_compare = None
    reg_pc_comp = None
    reg_ps_comp = None
    if compare_choice != "None" and compare_choice != primary_choice:
        compare_session = session_map.get(compare_choice)
        if compare_session:
            compare_tps = load_trackpoints(compare_session.id)
            df_compare_raw = build_df_from_trackpoints(compare_tps)
            df_compare = compute_derived_df(df_compare_raw)
            df_compare["power_filtered"] = np.where(
                (~df_compare["power"].isna())
                & (df_compare["power"] <= max_power_threshold),
                df_compare["power"],
                np.nan,
            )
            median_dt_c = (
                df_compare["time"].diff().dt.total_seconds().replace(0, np.nan).median()
            )
            window_pts_c = (
                max(1, int(round(std_window_s / median_dt_c)))
                if pd.notna(median_dt_c) and median_dt_c > 0
                else 5
            )
            df_compare["power_std"] = rolling_std(
                df_compare["power_filtered"], window_pts_c
            )
            reg_pc_comp = regression_with_ci(
                df_compare["power_filtered"].to_numpy(),
                df_compare["cadence"].to_numpy(),
            )
            reg_ps_comp = regression_with_ci(
                df_compare["power_filtered"].to_numpy(),
                df_compare["speed_kmh"].to_numpy(),
            )
    elif compare_choice == primary_choice and compare_choice != "None":
        st.warning(TRANSLATIONS["self_compare_warning"])

    def estimate_ftp(series_power, times):
        if series_power.dropna().empty:
            return None
        tmp = pd.Series(series_power.values, index=times)
        rolling_20m = tmp.rolling("20min", min_periods=1).mean()
        if rolling_20m.dropna().empty:
            return None
        peak_20m = rolling_20m.max()
        return 0.95 * peak_20m

    def normalized_power(series_power, times):
        if series_power.dropna().empty:
            return None
        tmp = pd.Series(series_power.values, index=times)
        roll30 = tmp.rolling("30s", min_periods=1).mean()
        fourth = (roll30**4).mean(skipna=True)
        if fourth is None or np.isnan(fourth):
            return None
        return fourth**0.25

    def compute_tss(np_val, ftp_val, duration_s):
        if not np_val or not ftp_val or not duration_s:
            return None
        if ftp_val == 0:
            return None
        intensity_factor = np_val / ftp_val
        return (duration_s * np_val * intensity_factor) / (ftp_val * 3600) * 100

    ftp_est = estimate_ftp(df_primary["power_filtered"], df_primary["time"])
    npower = normalized_power(df_primary["power_filtered"], df_primary["time"])
    tss_val = compute_tss(npower, ftp_est, to_number(primary_session.duration_s))

    avg_speed_kmh = to_number(primary_session.avg_speed_kmh)
    if avg_speed_kmh is None or np.isnan(avg_speed_kmh):
        if not df_primary["speed_kmh"].dropna().empty:
            avg_speed_kmh = df_primary["speed_kmh"].dropna().mean()

    # --- UI: summary metrics ---
    st.subheader(f"{TRANSLATIONS['primary_session']}: {primary_session.id}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(TRANSLATIONS["duration"], human_duration(primary_session.duration_s))
    col2.metric(
        TRANSLATIONS["distance"], metric_value(primary_session.distance_km, "{:.2f}")
    )
    col3.metric(
        TRANSLATIONS["elevation_gain"],
        metric_value(primary_session.elevation_gain_m, "{:.1f}"),
    )
    col4.metric(TRANSLATIONS["avg_speed"], metric_value(avg_speed_kmh, "{:.2f}"))

    extra1, extra2, extra3 = st.columns(3)
    extra1.metric(
        TRANSLATIONS["ftp_est"],
        metric_value(ftp_est, "{:.1f}"),
        help=TRANSLATIONS["tooltip_ftp"],
    )
    extra2.metric(
        TRANSLATIONS["normalized_power"],
        metric_value(npower, "{:.1f}"),
        help=TRANSLATIONS["tooltip_np"],
    )
    extra3.metric(
        TRANSLATIONS["tss"],
        metric_value(tss_val, "{:.1f}"),
        help=TRANSLATIONS["tooltip_tss"],
    )
    total_pts = len(df_primary)
    kept = df_primary["power_filtered"].count()
    removed = total_pts - kept
    st.markdown(
        f"**{TRANSLATIONS['power_filtering']}:** "
        + TRANSLATIONS["filter_info"].format(
            threshold=max_power_threshold,
            removed=removed,
            percent=(removed / total_pts * 100 if total_pts else 0),
        )
    )

    # -------------- TABS Streamlit --------------
    tab1, tab2, tab3 = st.tabs(
        [
            TRANSLATIONS["title"],
            TRANSLATIONS["progression"],
            TRANSLATIONS["advanced"],
        ]
    )

    # ----------- Classic tab: time series and stats -----------
    with tab1:
        st.markdown(f"## {TRANSLATIONS['time_series']}")
        max_elapsed = max(
            df_primary["elapsed_time_s"].max(),
            df_compare["elapsed_time_s"].max() if df_compare is not None else 0,
        )
        tick_every = 300 if max_elapsed > 3600 else 60
        tickvals = np.arange(0, max_elapsed + tick_every, tick_every)

        def format_seconds_to_hhmmss(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02}:{m:02}:{s:02}"

        ticktext = [format_seconds_to_hhmmss(v) for v in tickvals]

        fig_power = go.Figure()
        fig_power.add_trace(
            go.Scatter(
                x=df_primary["elapsed_time_s"],
                y=df_primary["power"],
                name="Power raw",
                mode="lines",
                line=dict(color="lightgray"),
                opacity=0.6,
            )
        )
        fig_power.add_trace(
            go.Scatter(
                x=df_primary["elapsed_time_s"],
                y=df_primary["power_filtered"],
                name="Power filtered",
                mode="lines",
                line=dict(color="blue"),
            )
        )
        for seg in stable_segments:
            fig_power.add_vrect(
                x0=seg["elapsed_time_s_start"],
                x1=seg["elapsed_time_s_end"],
                fillcolor="green",
                opacity=0.15,
                line_width=0,
                annotation_text="Stable",
                annotation_position="top left",
            )
        if df_compare is not None:
            fig_power.add_trace(
                go.Scatter(
                    x=df_compare["elapsed_time_s"],
                    y=df_compare["power_filtered"],
                    name="Compare power filtered",
                    mode="lines",
                    line=dict(color="orange", dash="dash"),
                )
            )
        fig_power.update_layout(
            title=TRANSLATIONS["power_over_time"],
            xaxis_title="Elapsed Time (hh:mm:ss)",
            yaxis_title="Power (W)",
        )
        fig_power.update_xaxes(
            tickvals=tickvals,
            ticktext=ticktext,
        )
        st.plotly_chart(fig_power, use_container_width=True)

        if stable_segments:
            seg_labels = [
                f"{i+1}: {human_duration(s['duration_s'])} @ {s['avg_power']:.1f}W"
                for i, s in enumerate(stable_segments)
            ]
            sel = st.selectbox(TRANSLATIONS["select_segment"], ["None"] + seg_labels)
            if sel != "None":
                idx = seg_labels.index(sel)
                seg = stable_segments[idx]
                import copy

                fig_zoom = copy.deepcopy(fig_power)
                fig_zoom.update_xaxes(
                    range=[
                        seg["elapsed_time_s_start"] - 10,
                        seg["elapsed_time_s_end"] + 10,
                    ]
                )
                zoom_min = max(seg["elapsed_time_s_start"] - 10, 0)
                zoom_max = seg["elapsed_time_s_end"] + 10
                tick_every = 60 if (zoom_max - zoom_min) < 3600 else 300
                tickvals_zoom = np.arange(zoom_min, zoom_max + tick_every, tick_every)
                ticktext_zoom = [format_seconds_to_hhmmss(v) for v in tickvals_zoom]
                fig_zoom.update_xaxes(tickvals=tickvals_zoom, ticktext=ticktext_zoom)
                st.plotly_chart(fig_zoom, use_container_width=True)

        cad_speed_col1, cad_speed_col2 = st.columns(2)
        with cad_speed_col1:
            fig_cad = go.Figure()
            fig_cad.add_trace(
                go.Line(
                    x=df_primary["elapsed_time_s"],
                    y=df_primary["cadence"],
                    name=TRANSLATIONS["cadence_over_time"],
                )
            )
            if df_compare is not None:
                fig_cad.add_trace(
                    go.Line(
                        x=df_compare["elapsed_time_s"],
                        y=df_compare["cadence"],
                        name=f"Compare {TRANSLATIONS['cadence_over_time']}",
                        line=dict(dash="dash", color="red"),
                    )
                )
            fig_cad.update_layout(
                title=TRANSLATIONS["cadence_over_time"],
                xaxis_title="Elapsed Time (hh:mm:ss)",
                yaxis_title="Cadence (rpm)",
            )
            fig_cad.update_xaxes(tickvals=tickvals, ticktext=ticktext)
            st.plotly_chart(fig_cad, use_container_width=True)
        with cad_speed_col2:
            fig_spd = go.Figure()
            fig_spd.add_trace(
                go.Line(
                    x=df_primary["elapsed_time_s"],
                    y=df_primary["speed_kmh"],
                    name=TRANSLATIONS["speed_over_time"],
                )
            )
            if df_compare is not None:
                fig_spd.add_trace(
                    go.Line(
                        x=df_compare["elapsed_time_s"],
                        y=df_compare["speed_kmh"],
                        name=f"Compare {TRANSLATIONS['speed_over_time']}",
                        line=dict(dash="dash", color="red"),
                    )
                )
            fig_spd.update_layout(
                title=TRANSLATIONS["speed_over_time"],
                xaxis_title="Elapsed Time (hh:mm:ss)",
                yaxis_title="Speed (km/h)",
            )
            fig_spd.update_xaxes(tickvals=tickvals, ticktext=ticktext)
            st.plotly_chart(fig_spd, use_container_width=True)

        st.markdown("### Distributions")
        hist_col1, hist_col2 = st.columns(2)
        with hist_col1:
            fig_hist_power = px.histogram(
                df_primary,
                x="power_filtered",
                nbins=40,
                title="Power Filtered Distribution",
            )
            st.plotly_chart(fig_hist_power, use_container_width=True)
        with hist_col2:
            fig_hist_cadence = px.histogram(
                df_primary, x="cadence", nbins=30, title="Cadence Distribution"
            )
            st.plotly_chart(fig_hist_cadence, use_container_width=True)

        st.markdown(f"## {TRANSLATIONS['correlations']}")
        corr1, corr2 = st.columns(2)
        with corr1:
            st.write(TRANSLATIONS["power_vs_cadence"])
            fig_pc = go.Figure()
            fig_pc.add_trace(
                go.Scatter(
                    x=df_primary["power_filtered"],
                    y=df_primary["cadence"],
                    mode="markers",
                    name="Primary",
                    marker=dict(size=6, opacity=0.6),
                )
            )
            if reg_pc:
                fig_pc.add_trace(
                    go.Line(
                        x=reg_pc["x"],
                        y=reg_pc["y_pred"],
                        name=f"Fit primary (R²={reg_pc['r2']:.3f})",
                        line=dict(color="black"),
                    )
                )
                fig_pc.add_annotation(
                    x=np.nanmean(reg_pc["x"]),
                    y=np.nanmean(reg_pc["y_pred"]),
                    text=f"slope={reg_pc['slope']:.3f}, intercept={reg_pc['intercept']:.1f}",
                    showarrow=False,
                    bgcolor="white",
                )
            if df_compare is not None and reg_pc_comp:
                fig_pc.add_trace(
                    go.Scatter(
                        x=df_compare["power_filtered"],
                        y=df_compare["cadence"],
                        mode="markers",
                        name="Compare",
                        marker=dict(size=6, opacity=0.4),
                    )
                )
                fig_pc.add_trace(
                    go.Line(
                        x=reg_pc_comp["x"],
                        y=reg_pc_comp["y_pred"],
                        name=f"Fit compare (R²={reg_pc_comp['r2']:.3f})",
                        line=dict(dash="dash"),
                    )
                )
            fig_pc.update_layout(
                xaxis_title="Power filtered (W)", yaxis_title="Cadence (rpm)"
            )
            st.plotly_chart(fig_pc, use_container_width=True)
        with corr2:
            st.write(TRANSLATIONS["power_vs_speed"])
            fig_ps = go.Figure()
            fig_ps.add_trace(
                go.Scatter(
                    x=df_primary["power_filtered"],
                    y=df_primary["speed_kmh"],
                    mode="markers",
                    name="Primary",
                    marker=dict(size=6, opacity=0.6),
                )
            )
            if reg_ps:
                fig_ps.add_trace(
                    go.Line(
                        x=reg_ps["x"],
                        y=reg_ps["y_pred"],
                        name=f"Fit primary (R²={reg_ps['r2']:.3f})",
                        line=dict(color="black"),
                    )
                )
                fig_ps.add_annotation(
                    x=np.nanmean(reg_ps["x"]),
                    y=np.nanmean(reg_ps["y_pred"]),
                    text=f"slope={reg_ps['slope']:.4f}, intercept={reg_ps['intercept']:.1f}",
                    showarrow=False,
                    bgcolor="white",
                )
            if df_compare is not None and reg_ps_comp:
                fig_ps.add_trace(
                    go.Scatter(
                        x=df_compare["power_filtered"],
                        y=df_compare["speed_kmh"],
                        mode="markers",
                        name="Compare",
                        marker=dict(size=6, opacity=0.4),
                    )
                )
                fig_ps.add_trace(
                    go.Line(
                        x=reg_ps_comp["x"],
                        y=reg_ps_comp["y_pred"],
                        name=f"Fit compare (R²={reg_ps_comp['r2']:.3f})",
                        line=dict(dash="dash"),
                    )
                )
            fig_ps.update_layout(
                xaxis_title="Power filtered (W)", yaxis_title="Speed (km/h)"
            )
            st.plotly_chart(fig_ps, use_container_width=True)

        # -- Export buttons --
        st.markdown(f"## {TRANSLATIONS['export']}")
        cleaned = df_primary[
            [
                "time",
                "elapsed_time_s",
                "power",
                "power_filtered",
                "cadence",
                "speed_kmh",
                "pace_min_per_km",
                "altitude_m",
                "distance_m",
            ]
        ]
        csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
        st.download_button(
            TRANSLATIONS["cleaned_trackpoints"],
            data=csv_bytes,
            file_name=f"session_{primary_session.id}_cleaned.csv",
            mime="text/csv",
        )

        # Export ALL (advanced)
        st.download_button(
            TRANSLATIONS.get("full_export_csv", "Export all session data (CSV)"),
            data=df_primary.to_csv(index=False).encode("utf-8"),
            file_name=f"session_{primary_session.id}_full.csv",
            mime="text/csv",
        )

        # PDF export
        if st.button(TRANSLATIONS["download_pdf"]):
            pdf_bytes = make_pdf(
                primary_session,
                df_primary,
                stable_segments,
                reg_pc,
                reg_ps,
                compare_session,
                df_compare,
                reg_pc_comp,
                reg_ps_comp,
                avg_speed_kmh,
                ftp_est,
                npower,
                tss_val,
                TRANSLATIONS,
            )
            st.download_button(
                "📄 PDF Report",
                data=pdf_bytes,
                file_name=f"session_{primary_session.id}_report.pdf",
                mime="application/pdf",
            )

        # Full PDF export (advanced)
        if st.button(TRANSLATIONS.get("full_export_pdf", "Export full PDF summary")):
            pdf_bytes = make_pdf(
                primary_session,
                df_primary,
                stable_segments,
                reg_pc,
                reg_ps,
                compare_session,
                df_compare,
                reg_pc_comp,
                reg_ps_comp,
                avg_speed_kmh,
                ftp_est,
                npower,
                tss_val,
                TRANSLATIONS,
                advanced=True,
            )
            st.download_button(
                "📄 " + TRANSLATIONS.get("full_export_pdf", "Export full PDF summary"),
                data=pdf_bytes,
                file_name=f"session_{primary_session.id}_full_report.pdf",
                mime="application/pdf",
            )

    # ---------- Progression tab ----------
    with tab2:
        st.markdown(f"## {TRANSLATIONS['weekly_trends']}")
        all_records = []
        for s in sessions:
            tps = load_trackpoints(s.id)
            if not tps:
                continue
            df = build_df_from_trackpoints(tps)
            df = compute_derived_df(df)
            df["power_filtered"] = np.where(
                (~df["power"].isna()) & (df["power"] <= max_power_threshold),
                df["power"],
                np.nan,
            )
            ftp_s = estimate_ftp(df["power_filtered"], df["time"])
            np_s = normalized_power(df["power_filtered"], df["time"])
            tss_s = compute_tss(np_s, ftp_s, to_number(s.duration_s))
            week = pd.to_datetime(s.start_time).to_period("W").start_time
            all_records.append(
                {
                    "session_id": s.id,
                    "week": week,
                    "ftp": ftp_s,
                    "np": np_s,
                    "tss": tss_s,
                    "date": s.start_time,
                }
            )
        if not all_records:
            st.info("No data for progression.")
        else:
            trend_df = pd.DataFrame(all_records)
            weekly = (
                trend_df.groupby("week")
                .agg({"ftp": "mean", "np": "mean", "tss": "sum"})
                .reset_index()
            )
            fig_ftp = px.line(
                weekly, x="week", y="ftp", title=TRANSLATIONS["ftp_trend"], markers=True
            )
            fig_np = px.line(
                weekly, x="week", y="np", title=TRANSLATIONS["np_trend"], markers=True
            )
            fig_tss = px.bar(
                weekly, x="week", y="tss", title=TRANSLATIONS["training_load"]
            )
            st.plotly_chart(fig_ftp, use_container_width=True)
            st.plotly_chart(fig_np, use_container_width=True)
            st.plotly_chart(fig_tss, use_container_width=True)

    # ---------- Advanced tab: power zones, CV, stable segments comparatif, etc. ----------
    with tab3:
        st.header(TRANSLATIONS["advanced"])

        # ---------- 1. Statistiques descriptives -------------
        st.subheader(TRANSLATIONS["Descriptive Statistics"])
        cols_stats = ["power_filtered", "cadence", "speed_kmh"]
        labels_stats = {
            "power_filtered": TRANSLATIONS["Power (W)"],
            "cadence": TRANSLATIONS["Cadence (rpm)"],
            "speed_kmh": TRANSLATIONS["Speed (km/h)"],
        }
        stats_table = pd.DataFrame(
            {
                TRANSLATIONS["Mean"]: [df_primary[c].mean() for c in cols_stats],
                TRANSLATIONS["Median"]: [df_primary[c].median() for c in cols_stats],
                TRANSLATIONS["Min"]: [df_primary[c].min() for c in cols_stats],
                TRANSLATIONS["Max"]: [df_primary[c].max() for c in cols_stats],
                TRANSLATIONS["Std"]: [df_primary[c].std() for c in cols_stats],
            },
            index=[labels_stats[c] for c in cols_stats],
        )
        st.dataframe(stats_table.style.format("{:.2f}"))

        # ---------- 2. Dispersion (boxplots) -------------------
        st.subheader(TRANSLATIONS["Dispersion (Boxplots)"])
        box_col1, box_col2, box_col3 = st.columns(3)
        with box_col1:
            st.plotly_chart(
                px.box(
                    df_primary,
                    y="power_filtered",
                    points="outliers",
                    title=TRANSLATIONS["Power (W)"],
                ),
                use_container_width=True,
            )
        with box_col2:
            st.plotly_chart(
                px.box(
                    df_primary,
                    y="cadence",
                    points="outliers",
                    title=TRANSLATIONS["Cadence (rpm)"],
                ),
                use_container_width=True,
            )
        with box_col3:
            st.plotly_chart(
                px.box(
                    df_primary,
                    y="speed_kmh",
                    points="outliers",
                    title=TRANSLATIONS["Speed (km/h)"],
                ),
                use_container_width=True,
            )

        # ---------- 3. Power zones (TABLEAU DETALILLE) ---------
        st.subheader("Power zones")

        power_zone_defs = [
            {
                "zone": "Z1",
                "name": {"en": "Active Recovery", "fr": "Récup. active"},
                "from": 0,
                "to": 34,
            },
            {
                "zone": "Z2",
                "name": {"en": "Endurance", "fr": "Endurance"},
                "from": 34,
                "to": 47,
            },
            {
                "zone": "Z3",
                "name": {"en": "Tempo", "fr": "Tempo"},
                "from": 47,
                "to": 56,
            },
            {
                "zone": "Z4",
                "name": {"en": "Threshold", "fr": "Seuil"},
                "from": 56,
                "to": 66,
            },
            {
                "zone": "Z5",
                "name": {"en": "VO2max", "fr": "VO2max"},
                "from": 66,
                "to": 75,
            },
            {
                "zone": "Z6",
                "name": {"en": "Anaerobic", "fr": "Anaérobie"},
                "from": 75,
                "to": 94,
            },
            {
                "zone": "Z7",
                "name": {"en": "Neuromuscular", "fr": "Neuromusculaire"},
                "from": 94,
                "to": 250,
            },
        ]
        bins = [z["from"] for z in power_zone_defs] + [power_zone_defs[-1]["to"]]
        labels = [z["zone"] for z in power_zone_defs]
        df_primary["custom_power_zone"] = pd.cut(
            df_primary["power_filtered"], bins=bins, labels=labels, right=False
        )

        dt_median = np.median(np.diff(df_primary["elapsed_time_s"].dropna()))
        zone_summary = []
        for z in power_zone_defs:
            count = (df_primary["custom_power_zone"] == z["zone"]).sum()
            seconds = int(count * dt_median)
            minutes = seconds // 60
            sec = seconds % 60
            t_human = f"{minutes}m {sec}s" if minutes else f"{sec}s"
            zone_summary.append(
                {
                    "Zone": z["zone"],
                    "Name": z["name"][st.session_state.lang],
                    "From": f"{z['from']} W",
                    "To": f"{z['to']} W",
                    "Time in zone": t_human,
                }
            )
        df_zone_table = pd.DataFrame(zone_summary)
        st.dataframe(df_zone_table)

        # ---------- 4. Meilleurs efforts rolling 5s, 1min, 5min, 20min ----------
        st.subheader(
            {
                "en": "Best average power (5s, 1min, 5min, 20min)",
                "fr": "Meilleurs efforts moyens (5s, 1min, 5min, 20min)",
            }[st.session_state.lang]
        )

        def best_effort(series, elapsed_s, win_sec):
            dt = np.median(np.diff(elapsed_s))
            win_pts = max(1, int(round(win_sec / dt))) if dt > 0 else 1
            return pd.Series(series).rolling(win_pts, min_periods=1).mean().max()

        best5s = best_effort(
            df_primary["power_filtered"].values, df_primary["elapsed_time_s"].values, 5
        )
        best1m = best_effort(
            df_primary["power_filtered"].values, df_primary["elapsed_time_s"].values, 60
        )
        best5m = best_effort(
            df_primary["power_filtered"].values,
            df_primary["elapsed_time_s"].values,
            300,
        )
        best20m = best_effort(
            df_primary["power_filtered"].values,
            df_primary["elapsed_time_s"].values,
            1200,
        )
        st.markdown(
            f"5s: {best5s:.0f}W, 1min: {best1m:.0f}W, 5min: {best5m:.0f}W, 20min: {best20m:.0f}W"
        )
        best_avg_df = pd.DataFrame(
            {
                "Interval": ["5s", "1min", "5min", "20min"],
                "Best Power (W)": [best5s, best1m, best5m, best20m],
            }
        )

        st.plotly_chart(
            px.bar(
                best_avg_df,
                x="Interval",
                y="Best Power (W)",
                text="Best Power (W)",
                title="Best average power over key intervals",
                labels={"Interval": "Interval", "Best Power (W)": "Best Power (W)"},
            ).update_traces(texttemplate="%{text:.0f}W", textposition="outside"),
            use_container_width=True,
        )

        # ---------- 5. Variabilité cadence -----------------
        st.subheader(
            {"en": "Cadence variability", "fr": "Variabilité cadence"}[
                st.session_state.lang
            ]
        )
        mean_cad = df_primary["cadence"].mean()
        std_cad = df_primary["cadence"].std()
        st.write(
            {
                "en": f"Mean: {mean_cad:.1f} rpm, Std: {std_cad:.1f} rpm",
                "fr": f"Moyenne: {mean_cad:.1f} rpm, Écart-type: {std_cad:.1f} rpm",
            }[st.session_state.lang]
        )

        # ---------- 6. Longest streak above power threshold ----------
        st.subheader(
            {
                "en": "Longest streak between 100W and 250W",
                "fr": "Plus longue séquence entre 100W et 250W",
            }[st.session_state.lang]
        )
        min_power_thr = 100
        max_power_thr = 250
        mask = (df_primary["power_filtered"] >= min_power_thr) & (
            df_primary["power_filtered"] < max_power_thr
        )
        max_len = 0
        current_len = 0
        for v in mask.values:
            if v:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        streak_s = max_len * np.median(np.diff(df_primary["elapsed_time_s"].values))

        def format_seconds_to_hhmmss(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02}:{m:02}:{s:02}"

        st.write(
            {
                "en": f"Max streak: {format_seconds_to_hhmmss(streak_s)} between {min_power_thr}W and {max_power_thr}W",
                "fr": f"Séquence max : {format_seconds_to_hhmmss(streak_s)} entre {min_power_thr}W et {max_power_thr}W",
            }[st.session_state.lang]
        )


def make_pdf(
    primary_session,
    df_primary,
    stable_segments,
    reg_pc,
    reg_ps,
    compare_session,
    df_compare,
    reg_pc_comp,
    reg_ps_comp,
    avg_speed_kmh,
    ftp_est,
    npower,
    tss_val,
    TRANSLATIONS,
    advanced=False,
):
    import io

    import numpy as np
    import plotly.io as pio
    from fpdf import FPDF

    def sanitize_text(s: str) -> str:
        if not isinstance(s, str):
            return s
        replacements = {
            "\u2014": "-",
            "\u2013": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
            "…": "...",
        }
        for k, v in replacements.items():
            s = s.replace(k, v)
        return s

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, sanitize_text(TRANSLATIONS["title"]), ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(
        0,
        8,
        sanitize_text(f"{TRANSLATIONS['primary_session']}: {primary_session.id}"),
        ln=True,
    )
    pdf.ln(2)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 6, "Key Metrics", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.cell(
        0,
        5,
        sanitize_text(
            f"{TRANSLATIONS['duration']}: {human_duration(primary_session.duration_s)}"
        ),
        ln=True,
    )
    pdf.cell(
        0,
        5,
        sanitize_text(
            f"{TRANSLATIONS['distance']}: {metric_value(primary_session.distance_km, '{:.2f}')} km"
        ),
        ln=True,
    )
    pdf.cell(
        0,
        5,
        sanitize_text(
            f"{TRANSLATIONS['avg_speed']}: {metric_value(avg_speed_kmh, '{:.2f}')} km/h"
        ),
        ln=True,
    )
    pdf.cell(
        0,
        5,
        sanitize_text(f"{TRANSLATIONS['ftp_est']}: {metric_value(ftp_est, '{:.1f}')}"),
        ln=True,
    )
    pdf.cell(
        0,
        5,
        sanitize_text(
            f"{TRANSLATIONS['normalized_power']}: {metric_value(npower, '{:.1f}')}"
        ),
        ln=True,
    )
    pdf.cell(
        0,
        5,
        sanitize_text(f"{TRANSLATIONS['tss']}: {metric_value(tss_val, '{:.1f}')}"),
        ln=True,
    )
    pdf.ln(5)
    try:
        fig_power = go.Figure()
        fig_power.add_trace(
            go.Scatter(
                x=df_primary["elapsed_time_s"],
                y=df_primary["power_filtered"],
                name="Power filtered",
                mode="lines",
            )
        )
        for seg in stable_segments:
            fig_power.add_vrect(
                x0=seg["elapsed_time_s_start"],
                x1=seg["elapsed_time_s_end"],
                fillcolor="green",
                opacity=0.15,
                line_width=0,
            )
        max_elapsed = df_primary["elapsed_time_s"].max()
        tick_every = 300 if max_elapsed > 3600 else 60

        def format_seconds_to_hhmmss(seconds):
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = int(seconds % 60)
            return f"{h:02}:{m:02}:{s:02}"

        tickvals = np.arange(0, max_elapsed + tick_every, tick_every)
        ticktext = [format_seconds_to_hhmmss(v) for v in tickvals]
        fig_power.update_layout(
            title=TRANSLATIONS["power_over_time"],
            xaxis_title="Elapsed Time (hh:mm:ss)",
            yaxis_title="Power (W)",
        )
        fig_power.update_xaxes(
            tickvals=tickvals,
            ticktext=ticktext,
        )
        img_power = pio.to_image(fig_power, format="png", width=700, height=300)
        pdf.image(io.BytesIO(img_power), x=10, w=190)
    except Exception as e:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(
            0,
            6,
            sanitize_text(f"{TRANSLATIONS['power_over_time']} (image unavailable)"),
            ln=True,
        )
        pdf.set_font("Arial", size=9)
        pdf.multi_cell(0, 5, sanitize_text(f"Could not render chart: {str(e)}"))

    pdf.ln(5)
    if stable_segments:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 6, sanitize_text(TRANSLATIONS["stable_segments"]), ln=True)
        pdf.set_font("Arial", size=9)
        for s in stable_segments[:5]:
            line = f"- {human_duration(s['duration_s'])} @ {s['avg_power']:.1f}W (std {s['std_power']:.1f})"
            pdf.cell(0, 5, sanitize_text(line), ln=True)

    if advanced:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Full Session Data (first 40 rows)", ln=True)
        pdf.set_font("Arial", size=8)
        for idx, row in df_primary.head(40).iterrows():
            vals = ", ".join(str(row[c]) for c in df_primary.columns)
            pdf.cell(0, 5, vals[:180], ln=True)

    return pdf.output(dest="S").encode("latin1", errors="replace")


if __name__ == "__main__":
    main()
