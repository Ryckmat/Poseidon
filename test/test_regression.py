import numpy as np
from src.dashboard.app import regression_with_ci

def test_regression_perfect_line():
    x = np.linspace(0, 10, 50)
    y = 2 * x + 5  # perfect
    result = regression_with_ci(x, y, n_boot=100)
    assert result is not None
    assert pytest.approx(2.0, rel=1e-2) == result["slope"]
    assert pytest.approx(5.0, rel=1e-2) == result["intercept"]
    assert result["r2"] > 0.999
