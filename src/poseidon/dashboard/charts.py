"""Figures Plotly du dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from poseidon.dashboard.formatting import hhmmss
from poseidon.dashboard.i18n import Translator
from poseidon.processing.metrics import LinearFit

PRIMARY_COLOR = "#1f6fd1"
COMPARE_COLOR = "#e8833a"
STABLE_COLOR = "#2e9e5b"


def elapsed_ticks(start_s: float, end_s: float) -> tuple[np.ndarray, list[str]]:
    span = max(end_s - start_s, 1)
    step = next(s for s in (30, 60, 300, 600, 1800, 3600) if span / s <= 12)
    first = np.floor(start_s / step) * step
    values = np.arange(first, end_s + step, step)
    return values, [hhmmss(v) for v in values]


def _time_axis(fig: go.Figure, t: Translator, start_s: float, end_s: float) -> None:
    values, labels = elapsed_ticks(start_s, end_s)
    fig.update_xaxes(title=t("elapsed_axis"), tickvals=values, ticktext=labels)
    fig.update_layout(margin=dict(t=50, b=40), hovermode="x unified")


def _max_elapsed(df: pd.DataFrame, compare: pd.DataFrame | None) -> float:
    end = df["elapsed_time_s"].max()
    if compare is not None:
        end = max(end, compare["elapsed_time_s"].max())
    return float(end)


def power_over_time(
    df: pd.DataFrame,
    segments: list[dict],
    t: Translator,
    compare: pd.DataFrame | None = None,
    zoom: tuple[float, float] | None = None,
    show_raw: bool = True,
) -> go.Figure:
    fig = go.Figure()
    if show_raw:
        fig.add_scatter(
            x=df["elapsed_time_s"],
            y=df["power"],
            name=t("power_raw"),
            mode="lines",
            line=dict(color="lightgray"),
            opacity=0.6,
        )
    fig.add_scatter(
        x=df["elapsed_time_s"],
        y=df["power_filtered"],
        name=t("power_filtered"),
        mode="lines",
        line=dict(color=PRIMARY_COLOR),
    )
    for seg in segments:
        fig.add_vrect(
            x0=seg["elapsed_time_s_start"],
            x1=seg["elapsed_time_s_end"],
            fillcolor=STABLE_COLOR,
            opacity=0.12,
            line_width=0,
            annotation_text=t("stable"),
            annotation_position="top left",
        )
    if compare is not None:
        fig.add_scatter(
            x=compare["elapsed_time_s"],
            y=compare["power_filtered"],
            name=f"{t('compare')} : {t('power_filtered')}",
            mode="lines",
            line=dict(color=COMPARE_COLOR, dash="dash"),
        )
    fig.update_layout(title=t("power_over_time"), yaxis_title=t("power_w"))
    if zoom:
        start, end = max(zoom[0], 0), zoom[1]
        fig.update_xaxes(range=[start, end])
    else:
        start, end = 0, _max_elapsed(df, compare)
    _time_axis(fig, t, start, end)
    return fig


def metric_over_time(
    df: pd.DataFrame,
    column: str,
    title: str,
    y_title: str,
    t: Translator,
    compare: pd.DataFrame | None = None,
    reverse_y: bool = False,
) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["elapsed_time_s"],
        y=df[column],
        mode="lines",
        name=t("primary"),
        line=dict(color=PRIMARY_COLOR),
    )
    if compare is not None:
        fig.add_scatter(
            x=compare["elapsed_time_s"],
            y=compare[column],
            mode="lines",
            name=t("compare"),
            line=dict(color=COMPARE_COLOR, dash="dash"),
        )
    fig.update_layout(title=title, yaxis_title=y_title)
    if reverse_y:
        fig.update_yaxes(autorange="reversed")
    _time_axis(fig, t, 0, _max_elapsed(df, compare))
    return fig


def scatter_with_fit(
    df: pd.DataFrame,
    y_col: str,
    fit: LinearFit | None,
    y_title: str,
    t: Translator,
    compare: pd.DataFrame | None = None,
    compare_fit: LinearFit | None = None,
) -> go.Figure:
    fig = go.Figure()
    series = [(df, fit, t("primary"), PRIMARY_COLOR, "solid")]
    if compare is not None:
        series.append((compare, compare_fit, t("compare"), COMPARE_COLOR, "dash"))
    for data, data_fit, label, color, dash in series:
        fig.add_scatter(
            x=data["power_filtered"],
            y=data[y_col],
            mode="markers",
            name=label,
            marker=dict(size=5, opacity=0.5, color=color),
        )
        if data_fit:
            fig.add_scatter(
                x=data_fit.x,
                y=data_fit.y_pred,
                mode="lines",
                name=(
                    f"{t('fit')} {label}: y = {data_fit.slope:.3f}x "
                    f"+ {data_fit.intercept:.1f} (R² {data_fit.r2:.2f})"
                ),
                line=dict(color=color, dash=dash),
            )
    fig.update_layout(
        xaxis_title=t("power_w"),
        yaxis_title=y_title,
        legend=dict(orientation="h", y=-0.25),
        margin=dict(t=30),
    )
    return fig
