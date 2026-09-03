"""Event indexing, cooldown, uniqueness and benchmark alignment."""
import numpy as np
import pandas as pd

from deadcat import events as E


def _events_at(positions, calendar, ticker="AAA"):
    return pd.DataFrame({
        "ticker": [ticker] * len(positions),
        "event_date": [calendar[p] for p in positions],
        "pos": list(positions),
    })


def test_forward_return_indexing_is_exact(synthetic_panel, calendar):
    """fwd_ret_h must equal P[t+h]/P[t]-1 exactly, for every horizon."""
    close, bench = synthetic_panel
    t = 200
    ev = _events_at([t], calendar)
    out = E.forward_outcomes(ev, close, bench, [1, 5, 10, 20, 60], 20)
    for h in (1, 5, 10, 20, 60):
        expected = close["AAA"].iloc[t + h] / close["AAA"].iloc[t] - 1
        assert np.isclose(out[f"fwd_ret_{h}"].iloc[0], expected), h


def test_car_equals_stock_minus_benchmark(synthetic_panel, calendar):
    close, bench = synthetic_panel
    t = 200
    out = E.forward_outcomes(_events_at([t], calendar), close, bench, [20], 20)
    r_s = close["AAA"].iloc[t + 20] / close["AAA"].iloc[t] - 1
    r_b = bench.iloc[t + 20] / bench.iloc[t] - 1
    assert np.isclose(out["car_20"].iloc[0], r_s - r_b)


def test_benchmark_alignment_is_date_matched(synthetic_panel, calendar):
    """Shifting the benchmark by one day must change CAR - proving alignment binds."""
    close, bench = synthetic_panel
    t = 200
    a = E.forward_outcomes(_events_at([t], calendar), close, bench, [20], 20)["car_20"].iloc[0]
    b = E.forward_outcomes(_events_at([t], calendar), close, bench.shift(1), [20], 20)["car_20"].iloc[0]
    assert not np.isclose(a, b)


def test_forward_outcome_is_nan_past_the_calendar_edge(synthetic_panel, calendar):
    """No wrap-around or clipping at the end of the sample."""
    close, bench = synthetic_panel
    t = len(calendar) - 5
    out = E.forward_outcomes(_events_at([t], calendar), close, bench, [1, 20], 20)
    assert np.isfinite(out["fwd_ret_1"].iloc[0])
    assert np.isnan(out["fwd_ret_20"].iloc[0])
    assert np.isnan(out["car_20"].iloc[0])


def test_mae_mfe_bracket_the_path(synthetic_panel, calendar):
    close, bench = synthetic_panel
    t = 200
    out = E.forward_outcomes(_events_at([t], calendar), close, bench, [20], 20)
    path = close["AAA"].iloc[t + 1:t + 21] / close["AAA"].iloc[t] - 1
    assert np.isclose(out["mfe_20"].iloc[0], path.max())
    assert np.isclose(out["mae_20"].iloc[0], path.min())
    assert out["mae_20"].iloc[0] <= out["mfe_20"].iloc[0]


def test_cooldown_suppresses_near_events(calendar):
    """Breaches inside the cooldown collapse into the first event's episode."""
    pos = [100, 105, 110, 130, 131, 200]
    cands = pd.DataFrame({
        "ticker": ["AAA"] * len(pos),
        "event_date": [calendar[p] for p in pos],
        "crash_z": [-3.5] * len(pos),
    })
    kept = E.apply_cooldown(cands, cooldown=20, calendar=calendar)
    assert list(kept["pos"]) == [100, 130, 200]
    assert list(kept["episode_length"]) == [3, 2, 1]


def test_cooldown_is_ticker_specific(calendar):
    """A cooldown on one ticker must not suppress another ticker's event."""
    cands = pd.DataFrame({
        "ticker": ["AAA", "BBB", "AAA"],
        "event_date": [calendar[100], calendar[102], calendar[104]],
        "crash_z": [-3.5, -3.5, -3.5],
    })
    kept = E.apply_cooldown(cands, cooldown=20, calendar=calendar)
    assert set(zip(kept.ticker, kept.pos)) == {("AAA", 100), ("BBB", 102)}


def test_cooldown_minimum_gap_holds_for_every_pair(calendar):
    rng = np.random.default_rng(3)
    pos = np.sort(rng.choice(np.arange(80, 390), 60, replace=False))
    cands = pd.DataFrame({"ticker": ["AAA"] * len(pos),
                          "event_date": [calendar[p] for p in pos],
                          "crash_z": [-3.2] * len(pos)})
    kept = E.apply_cooldown(cands, cooldown=20, calendar=calendar)
    gaps = np.diff(np.sort(kept["pos"].to_numpy()))
    assert (gaps >= 20).all()


def test_event_uniqueness(synthetic_panel, calendar):
    close, bench = synthetic_panel
    ret = E.daily_returns(close)
    elig = pd.DataFrame(True, index=calendar, columns=close.columns)
    ev = E.build_events(close, ret, elig, bench, z_threshold=-3.0, vol_window=60,
                        min_prior_obs=60, cooldown=20, horizons=[1, 5, 20],
                        primary_h=20, study_start=calendar[0], study_end=calendar[-1])
    assert not ev["event_id"].duplicated().any()
    assert not ev.duplicated(["ticker", "event_date"]).any()


def test_membership_gating_removes_ineligible_events(synthetic_panel, calendar):
    """A name excluded from the index yields no events."""
    close, bench = synthetic_panel
    ret = E.daily_returns(close)
    elig = pd.DataFrame(True, index=calendar, columns=close.columns)
    elig["AAA"] = False
    ev = E.build_events(close, ret, elig, bench, z_threshold=-3.0, vol_window=60,
                        min_prior_obs=60, cooldown=20, horizons=[1, 20], primary_h=20,
                        study_start=calendar[0], study_end=calendar[-1])
    assert (ev["ticker"] != "AAA").all() if len(ev) else True


def test_car_path_starts_at_zero_and_matches_car_20(synthetic_panel, calendar):
    close, bench = synthetic_panel
    t = 200
    ev = _events_at([t], calendar)
    path = E.car_path(ev, close, bench, max_tau=20)
    out = E.forward_outcomes(ev, close, bench, [20], 20)
    assert np.isclose(path[0, 0], 0.0)
    assert np.isclose(path[0, 20], out["car_20"].iloc[0])
