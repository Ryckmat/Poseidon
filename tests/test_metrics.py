import numpy as np
import pandas as pd
import pytest

from poseidon.config import AnalysisParams
from poseidon.processing import metrics


def _session(power, step_s=1, heart_rate=None):
    n = len(power)
    df = pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq=f"{step_s}s"),
            "power": power,
            "cadence": np.full(n, 24.0),
            "distance_m": np.arange(n) * 4.0 * step_s,
            "altitude_m": np.zeros(n),
        }
    )
    if heart_rate is not None:
        df["heart_rate"] = heart_rate
    return df


PARAMS = AnalysisParams(min_stable_duration_s=60)


def test_derived_speed_split_and_filter():
    df = metrics.prepare(_session([100.0, 300.0, 120.0]), PARAMS)
    assert np.isnan(df["speed_kmh"].iloc[0])
    assert df["speed_kmh"].iloc[1] == pytest.approx(14.4)
    assert df["pace_min_per_km"].iloc[1] == pytest.approx(1000 / 4 / 60)
    assert df["split_500m_s"].iloc[1] == pytest.approx(125.0)
    assert df["distance_per_stroke_m"].iloc[1] == pytest.approx(10.0)
    assert np.isnan(df["power_filtered"].iloc[1])


def test_missing_optional_columns_are_tolerated():
    raw = pd.DataFrame(
        {"time": pd.date_range("2025-01-01", periods=3, freq="1s"), "power": [1, 2, 3]}
    )
    df = metrics.prepare(raw, PARAMS)
    summary = metrics.summarize(df)
    assert summary.distance_km == 0.0
    assert summary.avg_heart_rate is None
    assert summary.avg_split_500m_s is None


def test_prepare_rejects_empty_frame():
    with pytest.raises(ValueError):
        metrics.prepare(pd.DataFrame(columns=["time", "power"]), PARAMS)


def test_stored_speed_takes_precedence():
    raw = _session([100.0] * 3)
    raw["speed_kmh"] = [np.nan, 10.0, np.nan]
    df = metrics.add_derived(raw)
    assert df["speed_kmh"].tolist()[1:] == pytest.approx([10.0, 14.4])


def test_stable_segment_detected_and_bounded():
    power = [0.0] * 30 + [150.0] * 120 + [0.0] * 30
    df = metrics.prepare(_session(power), PARAMS)
    segments = metrics.detect_stable_segments(df, PARAMS)
    assert len(segments) == 1
    seg = segments[0]
    assert seg["avg_power"] == pytest.approx(150.0)
    assert 60 <= seg["duration_s"] <= 120


def test_short_segment_ignored():
    power = [0.0] * 30 + [150.0] * 40 + [0.0] * 30
    df = metrics.prepare(_session(power), PARAMS)
    assert metrics.detect_stable_segments(df, PARAMS) == []


def test_linear_fit_exact_line():
    fit = metrics.linear_fit([1, 2, 3, np.nan], [3, 5, 7, 100])
    assert fit.slope == pytest.approx(2.0)
    assert fit.intercept == pytest.approx(1.0)
    assert fit.r2 == pytest.approx(1.0)
    assert metrics.linear_fit([1, 1], [2, 3]) is None


def test_constant_power_indicators():
    df = metrics.prepare(_session([200.0] * 3600, heart_rate=150.0), PARAMS)
    summary = metrics.summarize(df)
    assert summary.normalized_power == pytest.approx(200.0)
    assert summary.ftp == pytest.approx(190.0)
    assert summary.tss == pytest.approx(3599 / 3600 * (200 / 190) ** 2 * 100)
    assert summary.avg_heart_rate == pytest.approx(150.0)
    assert summary.avg_split_500m_s == pytest.approx(125.0)
    assert summary.avg_distance_per_stroke_m == pytest.approx(10.0)


def test_reference_ftp_drives_tss():
    df = metrics.prepare(_session([200.0] * 3600), PARAMS)
    summary = metrics.summarize(df, reference_ftp_w=200.0)
    assert summary.ftp == pytest.approx(190.0)
    assert summary.tss == pytest.approx(3599 / 3600 * 100)


def test_training_stress_score_needs_inputs():
    assert metrics.training_stress_score(None, 200, 3600) is None
    assert metrics.training_stress_score(200, 0, 3600) is None


def test_streak_best_efforts_and_zones():
    power = [50.0] * 10 + [150.0] * 20 + [50.0] * 5 + [150.0] * 5
    df = metrics.prepare(_session(power), PARAMS)
    assert metrics.longest_streak_s(df, 100, 250) == 20.0
    assert metrics.best_efforts(df)["5s"] == pytest.approx(150.0)
    zones = metrics.time_in_power_zones(df).set_index("zone")["seconds"]
    assert zones["Z7"] == 25.0
    assert zones["Z3"] == 15.0


def test_start_spike_does_not_inflate_ftp_or_best_efforts():
    # Pic à 250 W pendant 3 s au démarrage, puis 150 W.
    power = [250.0] * 3 + [150.0] * 1500
    df = metrics.prepare(_session(power), PARAMS)
    assert metrics.estimate_ftp(df["power_filtered"], df["time"]) == pytest.approx(
        0.95 * 150, abs=0.5
    )
    best = metrics.best_efforts(df)
    assert best["5s"] == pytest.approx(250 * 3 / 5 + 150 * 2 / 5)
    # Une fenêtre complète de 60 s peut légitimement contenir le pic.
    assert best["1min"] == pytest.approx((3 * 250 + 57 * 150) / 60)


def test_short_session_has_no_ftp_nor_long_efforts():
    df = metrics.prepare(_session([150.0] * 600), PARAMS)
    summary = metrics.summarize(df)
    assert summary.ftp is None
    assert summary.tss is None
    assert summary.normalized_power == pytest.approx(150.0)
    assert metrics.summarize(df, reference_ftp_w=200).tss > 0
    best = metrics.best_efforts(df)
    assert best["5min"] == pytest.approx(150.0)
    assert best["20min"] is None
