import numpy as np
import pandas as pd
import pytest

from poseidon.config import AnalysisParams
from poseidon.processing import metrics


def _session(power, step_s=1):
    n = len(power)
    return pd.DataFrame(
        {
            "time": pd.date_range("2025-01-01", periods=n, freq=f"{step_s}s"),
            "power": power,
            "cadence": np.full(n, 26.0),
            "distance_m": np.arange(n) * 4.0 * step_s,
            "altitude_m": np.zeros(n),
        }
    )


PARAMS = AnalysisParams(min_stable_duration_s=60)


def test_derived_speed_and_filter():
    df = metrics.prepare(_session([100.0, 300.0, 120.0]), PARAMS)
    assert df["speed_kmh"].iloc[1] == pytest.approx(14.4)
    assert df["pace_min_per_km"].iloc[1] == pytest.approx(1000 / 4 / 60)
    assert np.isnan(df["power_filtered"].iloc[1])


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
    df = metrics.prepare(_session([200.0] * 3600), PARAMS)
    summary = metrics.summarize(df)
    assert summary.normalized_power == pytest.approx(200.0)
    assert summary.ftp == pytest.approx(190.0)
    assert summary.tss == pytest.approx(3599 / 3600 * (200 / 190) ** 2 * 100)


def test_streak_best_efforts_and_zones():
    power = [50.0] * 10 + [150.0] * 20 + [50.0] * 5 + [150.0] * 5
    df = metrics.prepare(_session(power), PARAMS)
    assert metrics.longest_streak_s(df, 100, 250) == 20.0
    assert metrics.best_efforts(df)["5s"] == pytest.approx(150.0)
    zones = metrics.time_in_power_zones(df).set_index("zone")["seconds"]
    assert zones["Z7"] == 25.0
    assert zones["Z3"] == 15.0
