"""Rapport PDF d'une séance."""

import io
from typing import List

import pandas as pd
import plotly.io as pio
from fpdf import FPDF

from poseidon.dashboard import charts
from poseidon.dashboard.formatting import fmt, human_duration
from poseidon.dashboard.i18n import Translator
from poseidon.processing.metrics import SessionSummary

# Les polices standard de fpdf2 sont en latin-1.
_LATIN1_FIXES = str.maketrans(
    {"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "…": "..."}
)


def _latin1(s: str) -> str:
    return s.translate(_LATIN1_FIXES).encode("latin-1", "replace").decode("latin-1")


def build_report(
    session_id,
    df: pd.DataFrame,
    summary: SessionSummary,
    segments: List[dict],
    t: Translator,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    def line(text: str, size: int = 10, style: str = "", height: int = 5) -> None:
        pdf.set_font("helvetica", style, size)
        pdf.cell(0, height, _latin1(text), ln=True)

    line(t("title"), size=16, style="B", height=10)
    line(f"{t('primary_session')}: {session_id}", height=8)
    pdf.ln(2)
    line(t("key_metrics"), size=12, style="B", height=6)
    line(f"{t('duration')}: {human_duration(summary.duration_s)}")
    line(f"{t('distance')}: {fmt(summary.distance_km)}")
    line(f"{t('avg_speed')}: {fmt(summary.avg_speed_kmh)}")
    line(f"{t('ftp_est')}: {fmt(summary.ftp, '{:.1f}')}")
    line(f"{t('normalized_power')}: {fmt(summary.normalized_power, '{:.1f}')}")
    line(f"{t('tss')}: {fmt(summary.tss, '{:.1f}')}")
    pdf.ln(5)

    fig = charts.power_over_time(df, segments, t, show_raw=False)
    try:
        png = pio.to_image(fig, format="png", width=700, height=300)
        pdf.image(io.BytesIO(png), x=10, w=190)
    except Exception as exc:  # kaleido absent ou en échec
        line(f"{t('power_over_time')} ({t('chart_unavailable')})", size=12, style="B")
        pdf.set_font("helvetica", size=9)
        pdf.multi_cell(0, 5, _latin1(str(exc)))

    if segments:
        pdf.ln(5)
        line(t("stable_segments"), size=12, style="B", height=6)
        for s in segments[:5]:
            line(
                f"- {human_duration(s['duration_s'])} @ {s['avg_power']:.1f}W "
                f"(std {s['std_power']:.1f})",
                size=9,
            )

    return bytes(pdf.output())
