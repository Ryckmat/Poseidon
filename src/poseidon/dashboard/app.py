"""Dashboard Streamlit : streamlit run src/poseidon/dashboard/app.py"""

from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from poseidon.config import AnalysisParams
from poseidon.dashboard import charts, data, pdf
from poseidon.dashboard.formatting import fmt, hhmmss, human_duration
from poseidon.dashboard.i18n import LANGUAGES, Translator
from poseidon.processing import metrics

PARAM_DEFAULTS = AnalysisParams()
STREAK_RANGE_W = (100, 250)


class SessionView:
    """Séance chargée et analysée avec les paramètres courants."""

    def __init__(self, session: dict, params: AnalysisParams, bucket_s: int):
        self.id = session["id"]
        self.start_time = session["start_time"]
        raw = data.load_trackpoints(self.id, bucket_seconds=bucket_s)
        self.df = metrics.prepare(raw, params) if not raw.empty else None
        if self.df is None:
            return
        self.summary = metrics.summarize(self.df)
        self.segments = metrics.detect_stable_segments(self.df, params)
        self.fit_cadence = metrics.linear_fit(
            self.df["power_filtered"], self.df["cadence"]
        )
        self.fit_speed = metrics.linear_fit(
            self.df["power_filtered"], self.df["speed_kmh"]
        )


# -------------------------------------------------------------------- sidebar
def sidebar_language() -> Translator:
    lang = st.sidebar.selectbox(
        "Language / Langue",
        list(LANGUAGES),
        format_func=LANGUAGES.get,
        key="lang",
    )
    return Translator(lang)


def sidebar_params(t: Translator) -> tuple:
    """Paramètres d'analyse, avec presets conservés pour la session navigateur."""
    presets = st.session_state.setdefault("presets", {})
    st.sidebar.markdown(f"### {t('stable_params')}")

    if presets:
        chosen = st.sidebar.selectbox(t("preset_select"), [None, *presets])
        if chosen and st.session_state.get("_applied_preset") != chosen:
            st.session_state.update(presets[chosen])
            st.session_state["_applied_preset"] = chosen

    if st.sidebar.button(t("reset_params")):
        for key in [*vars(PARAM_DEFAULTS), "_applied_preset"]:
            st.session_state.pop(key, None)

    params = AnalysisParams(
        max_power=st.sidebar.number_input(
            t("power_threshold"),
            value=PARAM_DEFAULTS.max_power,
            step=10.0,
            key="max_power",
        ),
        min_stable_power=st.sidebar.number_input(
            t("min_stable_power"),
            value=PARAM_DEFAULTS.min_stable_power,
            step=5.0,
            key="min_stable_power",
        ),
        std_window_s=st.sidebar.number_input(
            t("rolling_std_window"),
            value=PARAM_DEFAULTS.std_window_s,
            step=5.0,
            key="std_window_s",
        ),
        std_threshold=st.sidebar.number_input(
            t("std_threshold"),
            value=PARAM_DEFAULTS.std_threshold,
            step=0.5,
            key="std_threshold",
        ),
        min_stable_duration_s=st.sidebar.number_input(
            t("min_segment_duration"),
            value=PARAM_DEFAULTS.min_stable_duration_s,
            step=10.0,
            key="min_stable_duration_s",
        ),
    )
    bucket_s = int(
        st.sidebar.number_input(t("bucket_seconds"), value=5, min_value=1, step=1)
    )

    name = st.sidebar.text_input(t("preset_name"))
    if st.sidebar.button(t("preset_save")) and name:
        presets[name] = vars(params).copy()
    return params, bucket_s


def sidebar_sessions(t: Translator, sessions: list) -> tuple:
    def label(s: Optional[dict]) -> str:
        return t("none") if s is None else f"{s['start_time']} | {s['id']}"

    primary = st.sidebar.selectbox(t("primary_session"), sessions, format_func=label)
    compare = st.sidebar.selectbox(
        t("compare_to_optional"), [None, *sessions], format_func=label
    )
    if compare is not None and compare["id"] == primary["id"]:
        st.warning(t("self_compare_warning"))
        compare = None
    return primary, compare


# ----------------------------------------------------------------------- body
def render_summary(view: SessionView, params: AnalysisParams, t: Translator) -> None:
    s = view.summary
    st.subheader(f"{t('primary_session')}: {view.id}")
    c = st.columns(4)
    c[0].metric(t("duration"), human_duration(s.duration_s))
    c[1].metric(t("distance"), fmt(s.distance_km))
    c[2].metric(t("elevation_gain"), fmt(s.elevation_gain_m, "{:.1f}"))
    c[3].metric(t("avg_speed"), fmt(s.avg_speed_kmh))
    c = st.columns(3)
    c[0].metric(t("ftp_est"), fmt(s.ftp, "{:.1f}"), help=t("tooltip_ftp"))
    c[1].metric(
        t("normalized_power"), fmt(s.normalized_power, "{:.1f}"), help=t("tooltip_np")
    )
    c[2].metric(t("tss"), fmt(s.tss, "{:.1f}"), help=t("tooltip_tss"))

    total = len(view.df)
    removed = total - view.df["power_filtered"].count()
    info = t(
        "filter_info",
        threshold=params.max_power,
        removed=removed,
        percent=removed / total * 100 if total else 0,
    )
    st.markdown(f"**{t('power_filtering')}:** {info}")


def render_session_tab(
    view: SessionView, compare: Optional[SessionView], t: Translator
) -> None:
    df = view.df
    df_cmp = compare.df if compare else None

    st.markdown(f"## {t('time_series')}")
    st.plotly_chart(
        charts.power_over_time(df, view.segments, t, compare=df_cmp),
        use_container_width=True,
    )
    if view.segments:
        labels = [
            f"{i + 1}: {human_duration(s['duration_s'])} @ {s['avg_power']:.1f}W"
            for i, s in enumerate(view.segments)
        ]
        choice = st.selectbox(
            t("select_segment"),
            [None, *range(len(labels))],
            format_func=lambda i: t("none") if i is None else labels[i],
        )
        if choice is not None:
            seg = view.segments[choice]
            zoom = (seg["elapsed_time_s_start"] - 10, seg["elapsed_time_s_end"] + 10)
            st.plotly_chart(
                charts.power_over_time(df, view.segments, t, df_cmp, zoom=zoom),
                use_container_width=True,
            )

    left, right = st.columns(2)
    left.plotly_chart(
        charts.metric_over_time(
            df, "cadence", t("cadence_over_time"), t("cadence_rpm"), t, df_cmp
        ),
        use_container_width=True,
    )
    right.plotly_chart(
        charts.metric_over_time(
            df, "speed_kmh", t("speed_over_time"), t("speed_kmh"), t, df_cmp
        ),
        use_container_width=True,
    )

    st.markdown(f"### {t('distributions')}")
    left, right = st.columns(2)
    left.plotly_chart(
        px.histogram(df, x="power_filtered", nbins=40, title=t("power_w")),
        use_container_width=True,
    )
    right.plotly_chart(
        px.histogram(df, x="cadence", nbins=30, title=t("cadence_rpm")),
        use_container_width=True,
    )

    st.markdown(f"## {t('correlations')}")
    left, right = st.columns(2)
    with left:
        st.write(t("power_vs_cadence"))
        st.plotly_chart(
            charts.scatter_with_fit(
                df,
                "cadence",
                view.fit_cadence,
                t("cadence_rpm"),
                t,
                df_cmp,
                compare.fit_cadence if compare else None,
            ),
            use_container_width=True,
        )
    with right:
        st.write(t("power_vs_speed"))
        st.plotly_chart(
            charts.scatter_with_fit(
                df,
                "speed_kmh",
                view.fit_speed,
                t("speed_kmh"),
                t,
                df_cmp,
                compare.fit_speed if compare else None,
            ),
            use_container_width=True,
        )

    render_exports(view, t)


def render_exports(view: SessionView, t: Translator) -> None:
    st.markdown(f"## {t('export')}")
    cleaned_cols = [
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
    st.download_button(
        t("cleaned_trackpoints"),
        data=view.df[cleaned_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"session_{view.id}_cleaned.csv",
        mime="text/csv",
    )
    st.download_button(
        t("full_export_csv"),
        data=view.df.to_csv(index=False).encode("utf-8"),
        file_name=f"session_{view.id}_full.csv",
        mime="text/csv",
    )
    if st.button(t("build_pdf")):
        st.download_button(
            "📄 " + t("download_pdf"),
            data=pdf.build_report(view.id, view.df, view.summary, view.segments, t),
            file_name=f"session_{view.id}_report.pdf",
            mime="application/pdf",
        )


def render_progression_tab(
    sessions: list, params: AnalysisParams, bucket_s: int, t: Translator
) -> None:
    st.markdown(f"## {t('weekly_trends')}")
    st.info(t("trends_info"))
    if not st.button(t("trends_run")):
        return
    records = []
    for s in sessions:
        view = SessionView(s, params, bucket_s)
        if view.df is None:
            continue
        records.append(
            {
                "week": pd.to_datetime(s["start_time"]).to_period("W").start_time,
                "ftp": view.summary.ftp,
                "np": view.summary.normalized_power,
                "tss": view.summary.tss,
            }
        )
    if not records:
        st.info(t("no_progression"))
        return
    weekly = (
        pd.DataFrame(records)
        .groupby("week")
        .agg(ftp=("ftp", "mean"), np=("np", "mean"), tss=("tss", "sum"))
        .reset_index()
    )
    st.plotly_chart(
        px.line(weekly, x="week", y="ftp", title=t("ftp_trend"), markers=True),
        use_container_width=True,
    )
    st.plotly_chart(
        px.line(weekly, x="week", y="np", title=t("np_trend"), markers=True),
        use_container_width=True,
    )
    st.plotly_chart(
        px.bar(weekly, x="week", y="tss", title=t("training_load")),
        use_container_width=True,
    )


def render_advanced_tab(view: SessionView, t: Translator) -> None:
    df = view.df
    columns = {
        "power_filtered": t("power_w"),
        "cadence": t("cadence_rpm"),
        "speed_kmh": t("speed_kmh"),
    }

    st.subheader(t("descriptive_stats"))
    stats = df[list(columns)].agg(["mean", "median", "min", "max", "std"]).T
    stats.columns = [t(c) for c in stats.columns]
    st.dataframe(stats.rename(index=columns).style.format("{:.2f}"))

    st.subheader(t("boxplots"))
    for col, (name, label) in zip(st.columns(3), columns.items()):
        col.plotly_chart(
            px.box(df, y=name, points="outliers", title=label),
            use_container_width=True,
        )

    st.subheader(t("power_zones"))
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

    st.subheader(t("best_efforts"))
    best = metrics.best_efforts(df)
    st.markdown(", ".join(f"{k}: {fmt(v, '{:.0f}')}W" for k, v in best.items()))
    best_df = pd.DataFrame(
        {t("interval"): list(best), t("power_w"): list(best.values())}
    )
    st.plotly_chart(
        px.bar(
            best_df, x=t("interval"), y=t("power_w"), text=t("power_w")
        ).update_traces(texttemplate="%{text:.0f}W", textposition="outside"),
        use_container_width=True,
    )

    low, high = STREAK_RANGE_W
    st.subheader(t("longest_streak", low=low, high=high))
    streak = metrics.longest_streak_s(df, low, high)
    st.write(t("max_streak", duration=hhmmss(streak)))


# ----------------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(page_title="Poseidon", layout="wide")
    t = sidebar_language()
    st.sidebar.title(t("controls"))
    st.title(t("title"))

    sessions = data.load_sessions()
    if not sessions:
        st.warning(t("no_sessions"))
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

    render_summary(view, params, t)
    tab_session, tab_progression, tab_advanced = st.tabs(
        [t("tab_session"), t("tab_progression"), t("tab_advanced")]
    )
    with tab_session:
        render_session_tab(view, compare_view, t)
    with tab_progression:
        render_progression_tab(sessions, params, bucket_s, t)
    with tab_advanced:
        render_advanced_tab(view, t)


if __name__ == "__main__":
    main()
