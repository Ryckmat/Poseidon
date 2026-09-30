import numpy as np
import pandas as pd

from poseidon.config import AnalysisParams
from poseidon.dashboard import pdf
from poseidon.dashboard.i18n import Translator
from poseidon.processing import metrics


def _prepared_session():
    n = 600
    raw = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq="1s"),
            "power": np.where(np.arange(n) < 300, 150.0, 90.0),
            "cadence": np.full(n, 24.0),
            "distance_m": np.arange(n) * 4.0,
            "altitude_m": np.zeros(n),
        }
    )
    params = AnalysisParams()
    df = metrics.prepare(raw, params)
    return df, metrics.summarize(df), metrics.detect_stable_segments(df, params)


def test_report_is_a_pdf_with_chart():
    df, summary, segments = _prepared_session()
    content = pdf.build_report(
        "Séance \u2014 test", df, summary, segments, Translator("fr")
    )
    assert content.startswith(b"%PDF")
    assert b"/Subtype /Image" in content


def test_report_without_chart_engine(monkeypatch):
    df, summary, segments = _prepared_session()

    def broken(*args, **kwargs):
        raise RuntimeError("moteur absent")

    monkeypatch.setattr(pdf.pio, "to_image", broken)
    content = pdf.build_report("s", df, summary, segments, Translator("en"))
    assert content.startswith(b"%PDF")
    assert b"/Subtype /Image" not in content
