import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.dashboard.app import detect_stable_segments, compute_derived_df

def make_df_with_segment():
    # build 120s of low power, then 120s of stable high power, then low
    times = pd.date_range(start="2025-08-01 12:00:00", periods=360, freq="S")
    power = np.concatenate([
        np.full(120, 50.0),
        np.full(120, 200.0),
        np.full(120, 50.0)
    ])
    cadence = np.full(360, 90.0)
    df = pd.DataFrame({
        "time": times,
        "power": power,
        "power_filtered": power,
        "cadence": cadence,
        "speed_kmh": np.full(360, 30.0),
        "altitude_m": np.zeros(360),
        "distance_m": np.linspace(0, 360 * 5 / 3600, 360),  # arbitrary
    })
    df = compute_derived_df(df)
    # simulate rolling std with window = 30s -> stable segment should be middle 120s
    df["power_std"] = pd.Series(df["power_filtered"]).rolling(window=30, min_periods=1).std().to_numpy()
    return df

def test_detect_stable_segment():
    df = make_df_with_segment()
    segments = detect_stable_segments(df, power_col="power_filtered", std_col="power_std",
                                      min_power=150, std_threshold=1.0, min_duration_s=60)
    # should find exactly one stable segment roughly 120s
    assert len(segments) == 1
    seg = segments[0]
    assert seg["duration_s"] >= 120 - 1  # tolerance
    assert pytest.approx(200.0, rel=1e-2) == seg["avg_power"]
