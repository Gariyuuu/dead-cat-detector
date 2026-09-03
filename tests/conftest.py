import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def calendar():
    return pd.bdate_range("2010-01-04", periods=400)


@pytest.fixture
def synthetic_panel(calendar):
    """Deterministic two-ticker panel with one engineered crash in AAA."""
    rng = np.random.default_rng(12345)
    n = len(calendar)
    ret_a = rng.normal(0.0, 0.01, n)
    ret_a[200] = -0.20                      # the engineered crash
    ret_a[201:221] = 0.005                  # engineered recovery drift
    ret_b = rng.normal(0.0, 0.01, n)
    close = pd.DataFrame({
        "AAA": 100 * np.cumprod(1 + ret_a),
        "BBB": 50 * np.cumprod(1 + ret_b),
    }, index=calendar)
    bench = pd.Series(200 * np.cumprod(1 + rng.normal(0.0, 0.004, n)),
                      index=calendar, name="SPY")
    return close, bench
