"""Figures Plotly du dashboard."""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from poseidon.dashboard.formatting import hhmmss
from poseidon.dashboard.i18n import Translator
from poseidon.processing.metrics import LinearFit


def elapsed_ticks(start_s: float, end_s: float) -> Tuple[np.ndarray, List[str]]:
    step = 300 if (end_s - start_s) > 3600 else 60
    values = np.arange(start_s, end_s + step, step)
    return values, [hhmmss(v) for v in values]


def _time_axis(fig: go.Figure, t: Translator, start_s: float, end_s: float) -> None:
    values, labels = elapsed_ticks(start_s, end_s)
    fig.update_xaxes(title=t("elapsed_axis"), tickvals=values, ticktext=labels)


def power_over_time(
    df: pd.DataFrame,
    segments: List[dict],
    t: Translator,
    compare: Optional[pd.DataFrame] = None,
    zoom: Optional[Tuple[float, float]] = None,
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
            opacity=0.5,
        )
    fig.add_scatter(
        x=df["elapsed_time_s"],
        y=df["power_filtered"],
        name=t("power_filtered"),
        mode="lines",
        line=dict(color="blue"),
    )
    for seg in segments:
        fig.add_vrect(
            x0=seg["elapsed_time_s_start"],
            x1=seg["elapsed_time_s_end"],
            fillcolor="green",
            opacity=0.15,
            line_width=0,
            annotation_text=t("stable"),
            annotation_position="top left",
        )
    if compare is not None:
        fig.add_scatter(
            x=compare["elapsed_time_s"],
            y=compare["power_filtered"],
            name=f"{t('compare')} — {t('power_filtered')}",
            mode="lines",
            line=dict(color="orange", dash="dash"),
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
    compare: Optional[pd.DataFrame] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["elapsed_time_s"], y=df[column], mode="lines", name=t("primary")
    )
    if compare is not None:
        fig.add_scatter(
            x=compare["elapsed_time_s"],
            y=compare[column],
            mode="lines",
            name=t("compare"),
            line=dict(dash="dash", color="red"),
        )
    fig.update_layout(title=title, yaxis_title=y_title)
    _time_axis(fig, t, 0, _max_elapsed(df, compare))
    return fig


def scatter_with_fit(
    df: pd.DataFrame,
    y_col: str,
    fit: Optional[LinearFit],
    y_title: str,
    t: Translator,
    compare: Optional[pd.DataFrame] = None,
    compare_fit: Optional[LinearFit] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(
        x=df["power_filtered"],
        y=df[y_col],
        mode="markers",
        name=t("primary"),
        marker=dict(size=6, opacity=0.6),
    )
    if fit:
        fig.add_scatter(
            x=fit.x,
            y=fit.y_pred,
            mode="lines",
            name=f"{t('fit')} {t('primary')} (R²={fit.r2:.3f})",
            line=dict(color="black"),
        )
        fig.add_annotation(
            x=float(np.mean(fit.x)),
            y=float(np.mean(fit.y_pred)),
            text=f"slope={fit.slope:.4f}, intercept={fit.intercept:.1f}",
            showarrow=False,
            bgcolor="white",
        )
    if compare is not None and compare_fit:
        fig.add_scatter(
            x=compare["power_filtered"],
            y=compare[y_col],
            mode="markers",
            name=t("compare"),
            marker=dict(size=6, opacity=0.4),
        )
        fig.add_scatter(
            x=compare_fit.x,
            y=compare_fit.y_pred,
            mode="lines",
            name=f"{t('fit')} {t('compare')} (R²={compare_fit.r2:.3f})",
            line=dict(dash="dash"),
        )
    fig.update_layout(xaxis_title=t("power_w"), yaxis_title=y_title)
    return fig


def _max_elapsed(df: pd.DataFrame, compare: Optional[pd.DataFrame]) -> float:
    end = df["elapsed_time_s"].max()
    if compare is not None:
        end = max(end, compare["elapsed_time_s"].max())
    return float(end)
