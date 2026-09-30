"""Dashboard Streamlit : streamlit run src/poseidon/dashboard/app.py"""

import os
import tempfile
from collections.abc import Callable
from dataclasses import replace
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from poseidon.config import AnalysisParams, ConfigError, upload_enabled
from poseidon.dashboard import charts, data, pdf
from poseidon.dashboard.formatting import MISSING, fmt, hhmmss, human_duration, split
from poseidon.dashboard.i18n import LANGUAGES, Translator
from poseidon.db.repository import SessionInfo
from poseidon.ingest.store import DuplicateSessionError, ingest
from poseidon.ingest.tcx import TcxError
from poseidon.processing import metrics
from poseidon.processing.analysis import run_analysis

STREAK_RANGE_W = (100, 250)
PARAM_KEYS = (
    "max_power",
    "min_stable_power",
    "std_window_s",
    "std_threshold",
    "min_stable_duration_s",
    "reference_ftp_w",
)


class SessionView:
    """Séance chargée et analysée avec les paramètres courants."""

    def __init__(self, info: SessionInfo, params: AnalysisParams, bucket_s: int):
        self.info = info
        raw = data.load_trackpoints(info.id, bucket_s)
        self.df = metrics.prepare(raw, params) if not raw.empty else None
        if self.df is None:
            return
        self.summary = metrics.summarize(self.df, params.reference_ftp_w)
        self.segments = metrics.detect_stable_segments(self.df, params)
        self.fit_cadence = metrics.linear_fit(
            self.df["power_filtered"], self.df["cadence"]
        )
        self.fit_speed = metrics.linear_fit(
            self.df["power_filtered"], self.df["speed_kmh"]
        )

    @property
    def has_heart_rate(self) -> bool:
        return self.df["heart_rate"].notna().any()


def session_label(info: SessionInfo | None, t: Translator) -> str:
    if info is None:
        return t("none")
    start = info.start_time.strftime("%Y-%m-%d %H:%M") if info.start_time else "?"
    return f"{start} | {info.name}"


# -------------------------------------------------------------------- sidebar
def sidebar_language() -> Translator:
    lang = st.sidebar.selectbox(
        "Language / Langue", list(LANGUAGES), format_func=LANGUAGES.get, key="lang"
    )
    return Translator(lang)


def sidebar_sessions(t: Translator, sessions: list[SessionInfo]) -> tuple:
    primary = st.sidebar.selectbox(
        t("primary_session"), sessions, format_func=lambda s: session_label(s, t)
    )
    compare = st.sidebar.selectbox(
        t("compare_to_optional"),
        [None, *sessions],
        format_func=lambda s: session_label(s, t),
    )
    if compare is not None and compare.id == primary.id:
        st.sidebar.warning(t("self_compare_warning"))
        compare = None
    return primary, compare


def sidebar_params(t: Translator) -> tuple:
    """Paramètres d'analyse (défauts issus de l'environnement) et presets."""
    defaults = AnalysisParams.from_env()
    presets = st.session_state.setdefault("presets", {})
    st.sidebar.markdown(f"### {t('stable_params')}")

    if presets:
        chosen = st.sidebar.selectbox(t("preset_select"), [None, *presets])
        if chosen and st.session_state.get("_applied_preset") != chosen:
            st.session_state.update(presets[chosen])
            st.session_state["_applied_preset"] = chosen

    if st.sidebar.button(t("reset_params")):
        for key in (*PARAM_KEYS, "_applied_preset"):
            st.session_state.pop(key, None)

    def number(label: str, key: str, step: float, value: float) -> float:
        return float(
            st.sidebar.number_input(
                t(label), value=float(value), min_value=0.0, step=step, key=key
            )
        )

    params = AnalysisParams(
        max_power=number("power_threshold", "max_power", 10.0, defaults.max_power),
        min_stable_power=number(
            "min_stable_power", "min_stable_power", 5.0, defaults.min_stable_power
        ),
        std_window_s=number(
            "rolling_std_window", "std_window_s", 5.0, defaults.std_window_s
        ),
        std_threshold=number(
            "std_threshold", "std_threshold", 0.5, defaults.std_threshold
        ),
        min_stable_duration_s=number(
            "min_segment_duration",
            "min_stable_duration_s",
            10.0,
            defaults.min_stable_duration_s,
        ),
        reference_ftp_w=number(
            "reference_ftp", "reference_ftp_w", 5.0, defaults.reference_ftp_w or 0
        ),
    )
    params = replace(params, reference_ftp_w=params.reference_ftp_w or None)
    bucket_s = int(
        st.sidebar.number_input(t("bucket_seconds"), value=5, min_value=1, step=1)
    )

    name = st.sidebar.text_input(t("preset_name"))
    if st.sidebar.button(t("preset_save"), disabled=not name):
        presets[name] = {k: getattr(params, k) or 0.0 for k in PARAM_KEYS}
    return params, bucket_s


def sidebar_upload(t: Translator) -> None:
    """Import de TCX depuis le navigateur (si POSEIDON_ENABLE_UPLOAD)."""
    generation = st.session_state.setdefault("upload_generation", 0)
    with st.sidebar.expander(t("upload")):
        for level, message in st.session_state.pop("upload_messages", []):
            getattr(st, level)(message)
        files = st.file_uploader(
            t("upload_files"),
            type=["tcx"],
            accept_multiple_files=True,
            key=f"upload_{generation}",
        )
        merge = st.checkbox(
            t("upload_merge"),
            disabled=len(files or []) < 2,
            key=f"upload_merge_{generation}",
        )
        name = st.text_input(t("upload_name"), key=f"upload_name_{generation}")
        if st.button(t("upload_run"), disabled=not files, type="primary"):
            with st.spinner(t("upload_run")):
                messages = import_uploads(files, merge, name.strip() or None, t)
            st.session_state["upload_messages"] = messages
            st.session_state["upload_generation"] = generation + 1
            data.clear_cache()
            st.rerun()


def import_uploads(files, merge: bool, name: str | None, t: Translator) -> list:
    messages = []
    params = AnalysisParams.from_env()
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for uploaded in files:
            path = os.path.join(tmp, os.path.basename(uploaded.name))
            with open(path, "wb") as fh:
                fh.write(uploaded.getbuffer())
            paths.append(path)
        groups = [paths] if merge else [[p] for p in paths]
        for group in groups:
            label = name or ", ".join(os.path.basename(p) for p in group)
            try:
                session_id = ingest(group, name=name if len(groups) == 1 else None)
                run_analysis(session_id, params)
                messages.append(("success", t("upload_done", name=label)))
            except DuplicateSessionError:
                messages.append(("warning", t("upload_skipped", name=label)))
            except (TcxError, ValueError) as exc:
                messages.append(
                    ("error", t("upload_failed", name=label, error=str(exc)))
                )
    return messages


# ----------------------------------------------------------------------- body
def _delta(value, other, pattern: str) -> str | None:
    if value is None or other is None or pd.isna(value) or pd.isna(other):
        return None
    text = pattern.format(value - other)
    # Écart nul à l'arrondi : pas de flèche.
    return None if not any(ch in "123456789" for ch in text) else text


def render_summary(
    view: SessionView,
    compare: Optional["SessionView"],
    params: AnalysisParams,
    t: Translator,
) -> None:
    s = view.summary
    c = compare.summary if compare else None

    def kpi(col, label, attr, show: Callable, delta_fmt=None, color="normal", **kw):
        value = getattr(s, attr)
        delta = _delta(value, getattr(c, attr), delta_fmt) if c and delta_fmt else None
        col.metric(t(label), show(value), delta=delta, delta_color=color, **kw)

    def watts(v) -> str:
        return fmt(v, "{:.0f}")

    st.subheader(session_label(view.info, t))
    row = st.columns(5)
    kpi(row[0], "duration", "duration_s", human_duration)
    kpi(row[1], "distance", "distance_km", fmt, "{:+.2f}")
    # Allure : plus bas = plus rapide.
    kpi(row[2], "avg_split", "avg_split_500m_s", split, "{:+.1f} s", "inverse")
    kpi(row[3], "avg_power", "avg_power", watts, "{:+.0f}")
    # Cadence et FC : ni mieux ni moins bien en soi.
    kpi(
        row[4],
        "avg_cadence",
        "avg_cadence",
        lambda v: fmt(v, "{:.1f}"),
        "{:+.1f}",
        "off",
    )
    row = st.columns(5)
    kpi(row[0], "avg_dps", "avg_distance_per_stroke_m", fmt, "{:+.2f}")
    kpi(row[1], "avg_hr", "avg_heart_rate", watts, "{:+.0f}", "off")
    kpi(
        row[2],
        "normalized_power",
        "normalized_power",
        watts,
        "{:+.0f}",
        help=t("tooltip_np"),
    )
    kpi(row[3], "ftp_est", "ftp", watts, "{:+.0f}", help=t("tooltip_ftp"))
    kpi(row[4], "tss", "tss", watts, "{:+.0f}", "off", help=t("tooltip_tss"))

    total = len(view.df)
    removed = total - int(view.df["power_filtered"].count())
    info = t(
        "filter_info",
        threshold=params.max_power,
        removed=removed,
        percent=removed / total * 100 if total else 0,
    )
    st.caption(info)


def render_session_tab(
    view: SessionView, compare: SessionView | None, t: Translator
) -> None:
    df = view.df
    df_cmp = compare.df if compare else None

    st.plotly_chart(charts.power_over_time(df, view.segments, t, compare=df_cmp))
    render_segments(view, df_cmp, t)

    left, right = st.columns(2)
    left.plotly_chart(
        charts.metric_over_time(
            df, "split_500m_s", t("split_over_time"), t("split_s"), t, df_cmp, True
        )
    )
    right.plotly_chart(
        charts.metric_over_time(
            df, "cadence", t("cadence_over_time"), t("cadence_spm"), t, df_cmp
        )
    )
    if view.has_heart_rate:
        st.plotly_chart(
            charts.metric_over_time(
                df, "heart_rate", t("hr_over_time"), t("hr_bpm"), t, df_cmp
            )
        )

    st.markdown(f"### {t('distributions')}")
    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(
            df, x="power_filtered", nbins=40, title=t("power_w")
        ).update_layout(xaxis_title=t("power_w"))
    )
    right.plotly_chart(
        px.histogram(df, x="cadence", nbins=30, title=t("cadence_spm")).update_layout(
            xaxis_title=t("cadence_spm")
        )
    )

    st.markdown(f"### {t('correlations')}")
    left, right = st.columns(2)
    with left:
        st.caption(t("power_vs_cadence"))
        st.plotly_chart(
            charts.scatter_with_fit(
                df,
                "cadence",
                view.fit_cadence,
                t("cadence_spm"),
                t,
                df_cmp,
                compare.fit_cadence if compare else None,
            )
        )
    with right:
        st.caption(t("power_vs_speed"))
        st.plotly_chart(
            charts.scatter_with_fit(
                df,
                "speed_kmh",
                view.fit_speed,
                t("speed_kmh"),
                t,
                df_cmp,
                compare.fit_speed if compare else None,
            )
        )

    render_exports(view, t)


def render_segments(view: SessionView, df_cmp, t: Translator) -> None:
    st.markdown(f"### {t('stable_segments')}")
    if not view.segments:
        st.caption(t("no_stable"))
        return
    table = pd.DataFrame(
        {
            t("start"): [hhmmss(s["elapsed_time_s_start"]) for s in view.segments],
            t("duration"): [human_duration(s["duration_s"]) for s in view.segments],
            t("avg_power"): [round(s["avg_power"]) for s in view.segments],
            t("avg_cadence"): [round(s["avg_cadence"], 1) for s in view.segments],
            t("avg_split"): [
                split(500 / (s["avg_speed_kmh"] / 3.6))
                if s["avg_speed_kmh"]
                else MISSING
                for s in view.segments
            ],
        }
    )
    table.index = range(1, len(table) + 1)
    st.dataframe(table)
    choice = st.selectbox(
        t("select_segment"),
        [None, *range(len(view.segments))],
        format_func=lambda i: t("none") if i is None else f"#{i + 1}",
    )
    if choice is not None:
        seg = view.segments[choice]
        zoom = (seg["elapsed_time_s_start"] - 10, seg["elapsed_time_s_end"] + 10)
        st.plotly_chart(charts.power_over_time(view.df, view.segments, t, df_cmp, zoom))


def render_exports(view: SessionView, t: Translator) -> None:
    st.markdown(f"### {t('export')}")
    cleaned_cols = [
        "time",
        "elapsed_time_s",
        "power",
        "power_filtered",
        "cadence",
        "heart_rate",
        "speed_kmh",
        "split_500m_s",
        "distance_per_stroke_m",
        "altitude_m",
        "distance_m",
    ]
    stem = f"session_{view.info.id}"
    left, middle, right = st.columns(3)
    left.download_button(
        t("cleaned_trackpoints"),
        data=view.df[cleaned_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_cleaned.csv",
        mime="text/csv",
        on_click="ignore",
    )
    middle.download_button(
        t("full_export_csv"),
        data=view.df.to_csv(index=False).encode("utf-8"),
        file_name=f"{stem}_full.csv",
        mime="text/csv",
        on_click="ignore",
    )
    if right.button(t("build_pdf")):
        right.download_button(
            t("download_pdf"),
            data=pdf.build_report(
                view.info.name, view.df, view.summary, view.segments, t
            ),
            file_name=f"{stem}_report.pdf",
            mime="application/pdf",
            on_click="ignore",
            type="primary",
        )


def render_progression_tab(t: Translator) -> None:
    history = data.load_history()
    if history.empty:
        st.info(t("no_progression"))
        return
    pending = int(
        history[["normalized_power", "tss", "ftp_estimated"]].isna().all(axis=1).sum()
    )
    if pending:
        st.info(t("not_analyzed", count=pending))

    history["week"] = (
        history["start_time"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    )
    weekly = (
        history.groupby("week")
        .agg(
            distance_km=("distance_km", "sum"),
            sessions=("id", "count"),
            tss=("tss", "sum"),
            ftp=("ftp_estimated", "mean"),
            np=("normalized_power", "mean"),
        )
        .reset_index()
    )

    st.markdown(f"### {t('weekly_trends')}")
    left, right = st.columns(2)
    labels = {
        "week": t("week"),
        "distance_km": t("distance"),
        "sessions": t("weekly_sessions"),
        "tss": t("tss"),
        "value": t("power_w"),
        "metric": "",
    }
    left.plotly_chart(
        px.bar(
            weekly, x="week", y="distance_km", title=t("weekly_volume"), labels=labels
        )
    )
    right.plotly_chart(
        px.bar(
            weekly, x="week", y="sessions", title=t("weekly_sessions"), labels=labels
        )
    )
    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(weekly, x="week", y="tss", title=t("training_load"), labels=labels)
    )
    trend = weekly.melt(id_vars="week", value_vars=["ftp", "np"], var_name="metric")
    trend["metric"] = trend["metric"].map(
        {"ftp": t("ftp_est"), "np": t("normalized_power")}
    )
    right.plotly_chart(
        px.line(
            trend.dropna(),
            x="week",
            y="value",
            color="metric",
            markers=True,
            labels=labels,
            title=f"{t('ftp_trend')} / {t('np_trend')}",
        )
    )

    st.markdown(f"### {t('records')}")
    st.dataframe(records_table(history, t), hide_index=True)


def records_table(history: pd.DataFrame, t: Translator) -> pd.DataFrame:
    speed = history["avg_speed_kmh"].where(history["avg_speed_kmh"] > 0)
    history = history.assign(split_s=500 / (speed / 3.6))
    specs = [
        ("rec_distance", "distance_km", "max", lambda v: fmt(v)),
        ("rec_duration", "duration_s", "max", human_duration),
        ("rec_avg_power", "avg_power", "max", lambda v: fmt(v, "{:.0f}")),
        ("rec_np", "normalized_power", "max", lambda v: fmt(v, "{:.0f}")),
        ("rec_split", "split_s", "min", split),
        ("rec_ftp", "ftp_estimated", "max", lambda v: fmt(v, "{:.0f}")),
    ]
    rows = []
    for label, column, how, show in specs:
        values = history[column].dropna()
        if values.empty:
            continue
        idx = values.idxmax() if how == "max" else values.idxmin()
        rows.append(
            {
                t("record"): t(label),
                t("value"): show(values[idx]),
                t("date"): history.loc[idx, "start_time"].strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


def render_advanced_tab(view: SessionView, t: Translator) -> None:
    df = view.df
    columns = {
        "power_filtered": t("power_w"),
        "cadence": t("cadence_spm"),
        "split_500m_s": t("split_s"),
        "distance_per_stroke_m": t("dps_m"),
        "heart_rate": t("hr_bpm"),
    }
    if not view.has_heart_rate:
        columns.pop("heart_rate")

    st.markdown(f"### {t('descriptive_stats')}")
    stats = df[list(columns)].agg(["mean", "median", "min", "max", "std"]).T
    stats.columns = [t(c) for c in stats.columns]
    st.dataframe(stats.rename(index=columns).style.format("{:.2f}"))

    st.markdown(f"### {t('boxplots')}")
    for col, (name, label) in zip(
        st.columns(len(columns)), columns.items(), strict=True
    ):
        col.plotly_chart(
            px.box(df, y=name, points="outliers", title=label).update_layout(
                yaxis_title=None
            )
        )

    st.markdown(f"### {t('power_zones')}")
    zones = metrics.time_in_power_zones(df)
    st.dataframe(
        pd.DataFrame(
            {
                t("zone"): zones["zone"],
                t("name"): zones[f"name_{t.lang}"],
                t("from"): zones["from_w"].astype(str) + " W",
                t("to"): zones["to_w"].astype(str) + " W",
                t("time_in_zone"): zones["seconds"].map(human_duration),
            }
        ),
        hide_index=True,
    )

    st.markdown(f"### {t('best_efforts')}")
    best = metrics.best_efforts(df)
    best_df = pd.DataFrame(
        {t("interval"): list(best), t("power_w"): list(best.values())}
    )
    st.plotly_chart(
        px.bar(
            best_df, x=t("interval"), y=t("power_w"), text=t("power_w")
        ).update_traces(texttemplate="%{text:.0f} W", textposition="outside")
    )

    low, high = STREAK_RANGE_W
    st.markdown(f"### {t('longest_streak', low=low, high=high)}")
    streak = metrics.longest_streak_s(df, low, high)
    st.write(t("max_streak", duration=hhmmss(streak)))


# ----------------------------------------------------------------------- main
def run(t: Translator) -> None:
    if st.sidebar.button(t("refresh")):
        data.clear_cache()
    if upload_enabled():
        sidebar_upload(t)

    limit = int(
        st.sidebar.number_input(
            t("sessions_shown"), min_value=10, max_value=1000, value=50, step=10
        )
    )
    sessions = data.load_sessions(limit)
    if not sessions:
        st.info(t("no_sessions"))
        return
    primary, compare = sidebar_sessions(t, sessions)
    params, bucket_s = sidebar_params(t)

    view = SessionView(primary, params, bucket_s)
    if view.df is None:
        st.error(t("no_trackpoints"))
        return
    compare_view = SessionView(compare, params, bucket_s) if compare else None
    if compare_view is not None and compare_view.df is None:
        compare_view = None

    render_summary(view, compare_view, params, t)
    tab_session, tab_progression, tab_advanced = st.tabs(
        [t("tab_session"), t("tab_progression"), t("tab_advanced")]
    )
    with tab_session:
        render_session_tab(view, compare_view, t)
    with tab_progression:
        render_progression_tab(t)
    with tab_advanced:
        render_advanced_tab(view, t)


def main() -> None:
    st.set_page_config(page_title="Poseidon", page_icon="🚣", layout="wide")
    t = sidebar_language()
    st.sidebar.title(t("controls"))
    st.title(t("title"))
    try:
        run(t)
    except (ConfigError, SQLAlchemyError) as exc:
        detail = str(getattr(exc, "orig", None) or exc).strip().splitlines()[0]
        st.error(t("db_error", error=detail))


if __name__ == "__main__":
    main()
