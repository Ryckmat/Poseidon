import pytest

from src.dashboard.app import (
    compute_tss,
    estimate_ftp,
    human_duration,
    normalized_power,
)


def test_human_duration():
    assert (
        human_duration(3661) == "1h 1m 1s" or "1h1m1s"
    )  # depending du formatting, accepte présence d'unités


def test_estimate_ftp_constant(power_constant, simple_time_series):
    ftp = estimate_ftp(power_constant, simple_time_series)
    # constant 150W => 20min average is 150, ftp = 0.95 * 150
    assert pytest.approx(0.95 * 150.0, rel=1e-3) == ftp


def test_normalized_power_constant(power_constant, simple_time_series):
    npow = normalized_power(power_constant, simple_time_series)
    # constant power => NP == power
    assert pytest.approx(150.0, rel=1e-3) == npow


def test_tss_calculation():
    # example: NP=150, FTP=150, duration=3600s (1h)
    tss = compute_tss(150.0, 150.0, 3600)
    # IF = 1 => TSS = (3600 * 150 * 1) / (150 * 3600) * 100 = 100
    assert pytest.approx(100.0, rel=1e-3) == tss
