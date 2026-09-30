"""Rapport PDF d'une séance."""

import io
import logging

import pandas as pd
import plotly.io as pio
from fpdf import FPDF

from poseidon.dashboard import charts
from poseidon.dashboard.formatting import fmt, human_duration, split
from poseidon.dashboard.i18n import Translator
from poseidon.processing.metrics import SessionSummary

log = logging.getLogger(__name__)

# Les polices standard de fpdf2 sont en latin-1.
_LATIN1_FIXES = str.maketrans(
    {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }
)


def _latin1(s: str) -> str:
    return s.translate(_LATIN1_FIXES).encode("latin-1", "replace").decode("latin-1")


def build_report(
    session_name: str,
    df: pd.DataFrame,
    summary: SessionSummary,
    segments: list[dict],
    t: Translator,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    def line(text: str, size: int = 10, style: str = "", height: int = 5) -> None:
        pdf.set_font("helvetica", style, size)
        pdf.cell(0, height, _latin1(text), new_x="LMARGIN", new_y="NEXT")

    line(t("title"), size=16, style="B", height=10)
    line(f"{t('primary_session')}: {session_name}", height=8)
    pdf.ln(2)
    line(t("key_metrics"), size=12, style="B", height=6)
    rows = [
        (t("duration"), human_duration(summary.duration_s)),
        (t("distance"), fmt(summary.distance_km)),
        (t("avg_split"), split(summary.avg_split_500m_s)),
        (t("avg_power"), fmt(summary.avg_power, "{:.0f}")),
        (t("avg_cadence"), fmt(summary.avg_cadence, "{:.1f}")),
        (t("avg_dps"), fmt(summary.avg_distance_per_stroke_m)),
        (t("avg_hr"), fmt(summary.avg_heart_rate, "{:.0f}")),
        (t("ftp_est"), fmt(summary.ftp, "{:.0f}")),
        (t("normalized_power"), fmt(summary.normalized_power, "{:.0f}")),
        (t("tss"), fmt(summary.tss, "{:.0f}")),
    ]
    for label, value in rows:
        line(f"{label}: {value}")
    pdf.ln(4)

    fig = charts.power_over_time(df, segments, t, show_raw=False)
    try:
        png = pio.to_image(fig, format="png", width=900, height=380)
        pdf.image(io.BytesIO(png), x=10, w=190)
    except Exception as exc:  # moteur d'export d'image absent ou en échec
        log.warning("Graphique non intégré au PDF : %s", exc)
        line(f"{t('power_over_time')} ({t('chart_unavailable')})", size=10, style="I")

    if segments:
        pdf.ln(4)
        line(t("stable_segments"), size=12, style="B", height=6)
        for s in segments[:10]:
            line(
                f"- {human_duration(s['duration_s'])} @ {s['avg_power']:.0f} W "
                f"(std {s['std_power']:.1f})",
                size=9,
            )

    return bytes(pdf.output())
