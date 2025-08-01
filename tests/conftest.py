from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_time_series():
    # 1 Hz for 5 minutes
    start = datetime(2025, 8, 1, 12, 0, 0)
    times = [start + timedelta(seconds=i) for i in range(300)]
    return pd.to_datetime(times)


@pytest.fixture
def power_constant(simple_time_series):
    # constant power of 150W
    return pd.Series(150.0, index=simple_time_series)


@pytest.fixture
def power_with_noise(simple_time_series):
    rng = np.random.default_rng(0)
    base = 200.0
    noise = rng.normal(0, 5, len(simple_time_series))
    return pd.Series(base + noise, index=simple_time_series)


@pytest.fixture
def cadence_linear(simple_time_series):
    # cadence increases linearly from 80 to 100
    n = len(simple_time_series)
    return pd.Series(np.linspace(80, 100, n), index=simple_time_series)
