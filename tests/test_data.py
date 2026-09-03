"""Universe reconstruction, ticker-reuse guard and the price-integrity screen."""
import numpy as np
import pandas as pd

from deadcat import data as D


def test_valid_ticker_rejects_nan():
    """NaN is truthy in Python - the guard must not be a bare boolean test."""
    assert D.valid_ticker("AAPL")
    assert not D.valid_ticker(float("nan"))
    assert not D.valid_ticker(None)
    assert not D.valid_ticker("")
    assert not D.valid_ticker("  ")


def test_yahoo_symbol_normalisation():
    assert D.yahoo_symbol("BRK.B") == "BRK-B"
    assert D.yahoo_symbol(" aapl ") == "AAPL"


def test_membership_rollback_reconstructs_windows():
    """A name added then removed gets exactly the window between those dates."""
    current = pd.DataFrame({"ticker": ["AAA", "BBB"], "security": ["a", "b"],
                            "sector": ["Energy", "Energy"]})
    changes = pd.DataFrame({
        "date": pd.to_datetime(["2012-01-10", "2018-06-01", "2020-03-02"]),
        "added": ["CCC", "AAA", None],
        "removed": [None, "CCC", "DDD"],
    })
    w = D.build_membership_windows(current, changes,
                                   pd.Timestamp("2005-01-01"), pd.Timestamp("2026-01-01"))
    ccc = w[w.ticker == "CCC"].iloc[0]
    assert ccc.start == pd.Timestamp("2012-01-10")
    assert ccc.end == pd.Timestamp("2018-06-01")

    aaa = w[w.ticker == "AAA"].iloc[0]
    assert aaa.start == pd.Timestamp("2018-06-01")      # joined here
    assert aaa.end == pd.Timestamp("2026-01-01")

    ddd = w[w.ticker == "DDD"].iloc[0]
    assert ddd.end == pd.Timestamp("2020-03-02")        # left here
    assert ddd.start == pd.Timestamp("2005-01-01")

    bbb = w[w.ticker == "BBB"].iloc[0]                   # never changed: full window
    assert bbb.start == pd.Timestamp("2005-01-01")
    assert bbb.end == pd.Timestamp("2026-01-01")


def test_membership_mask_is_half_open():
    w = pd.DataFrame({"ticker": ["AAA"], "start": [pd.Timestamp("2010-01-05")],
                      "end": [pd.Timestamp("2010-01-08")]})
    idx = pd.bdate_range("2010-01-04", periods=6)
    m = D.membership_mask(w, idx, ["AAA"])
    assert not m.loc[pd.Timestamp("2010-01-04"), "AAA"]
    assert m.loc[pd.Timestamp("2010-01-05"), "AAA"]
    assert m.loc[pd.Timestamp("2010-01-07"), "AAA"]
    assert not m.loc[pd.Timestamp("2010-01-08"), "AAA"]   # end is exclusive


def test_reuse_guard_drops_non_overlapping_history():
    """A recycled ticker whose data starts after the name left the index is dropped."""
    dates = pd.bdate_range("2024-01-01", periods=60)
    prices = pd.DataFrame({"date": dates, "ticker": "SBNY",
                           "open": 1.0, "high": 1.0, "low": 1.0,
                           "close": 1.0, "volume": 1e6})
    windows = pd.DataFrame({"ticker": ["SBNY"], "start": [pd.Timestamp("2015-01-01")],
                            "end": [pd.Timestamp("2023-03-01")]})
    kept, rejects = D.apply_reuse_guard(prices, windows)
    assert kept.empty
    assert rejects.iloc[0]["ticker"] == "SBNY"


def test_reuse_guard_keeps_overlapping_history():
    dates = pd.bdate_range("2016-01-01", periods=200)
    prices = pd.DataFrame({"date": dates, "ticker": "AAA", "open": 1.0, "high": 1.0,
                           "low": 1.0, "close": 1.0, "volume": 1e6})
    windows = pd.DataFrame({"ticker": ["AAA"], "start": [pd.Timestamp("2015-01-01")],
                            "end": [pd.Timestamp("2020-01-01")]})
    kept, rejects = D.apply_reuse_guard(prices, windows)
    assert not kept.empty
    assert rejects.empty


def _series(vals, ticker="X"):
    d = pd.bdate_range("2010-01-04", periods=len(vals))
    return pd.DataFrame({"date": d, "ticker": ticker, "open": vals, "high": vals,
                         "low": vals, "close": vals, "volume": 1e6})


def test_integrity_screen_catches_interleaved_series():
    """A series oscillating between two price levels is rejected."""
    vals = np.tile([100.0, 5.0], 200)
    kept, rej = D.price_integrity_screen(_series(vals))
    assert kept.empty
    assert "round_trip_oscillation" in rej.iloc[0]["reason"]


def test_integrity_screen_catches_quantised_penny_series():
    rng = np.random.default_rng(0)
    vals = rng.choice([0.20, 0.25, 0.30, 0.35, 0.40], 400).astype(float)
    kept, rej = D.price_integrity_screen(_series(vals))
    assert kept.empty
    assert "low_value_diversity" in rej.iloc[0]["reason"]


def test_integrity_screen_keeps_a_genuine_one_off_squeeze():
    """GameStop-shaped path: one violent round trip must survive."""
    rng = np.random.default_rng(1)
    vals = list(20 * np.cumprod(1 + rng.normal(0, 0.02, 600)))
    base = vals[300]
    vals[300:306] = [base * m for m in (2.4, 1.3, 2.1, 1.5, 0.9, 0.8)]
    kept, rej = D.price_integrity_screen(_series(np.array(vals, float)))
    assert not kept.empty
    assert rej.empty


def test_integrity_screen_keeps_a_split_adjusted_growth_series():
    """NVDA-shaped path: sub-dollar early prices are legitimate, not corruption."""
    rng = np.random.default_rng(2)
    vals = 0.30 * np.cumprod(1 + rng.normal(0.0012, 0.02, 3000))
    kept, rej = D.price_integrity_screen(_series(vals))
    assert not kept.empty
    assert rej.empty
