"""The load-bearing tests: nothing that defines an event may see the event."""
import numpy as np
import pandas as pd

from deadcat import events as E, features as F


def test_rolling_moments_exclude_current_and_future(synthetic_panel):
    """mu/sigma at t must be computable from returns up to t-1 only."""
    close, _ = synthetic_panel
    ret = E.daily_returns(close)
    mu, sigma = E.rolling_moments(ret, window=60, min_obs=60)

    t = 200
    manual_mu = ret["AAA"].iloc[t - 60:t].mean()
    manual_sd = ret["AAA"].iloc[t - 60:t].std(ddof=1)
    assert np.isclose(mu["AAA"].iloc[t], manual_mu)
    assert np.isclose(sigma["AAA"].iloc[t], manual_sd)


def test_rolling_moments_invariant_to_future_mutation(synthetic_panel):
    """Corrupting every return from t onward must not change mu/sigma at t."""
    close, _ = synthetic_panel
    ret = E.daily_returns(close)
    t = 200
    mutated = ret.copy()
    mutated.iloc[t:] = mutated.iloc[t:] * 100 + 7.0

    mu_a, sd_a = E.rolling_moments(ret, 60, 60)
    mu_b, sd_b = E.rolling_moments(mutated, 60, 60)
    assert np.isclose(mu_a["AAA"].iloc[t], mu_b["AAA"].iloc[t])
    assert np.isclose(sd_a["AAA"].iloc[t], sd_b["AAA"].iloc[t])


def test_zscore_uses_prior_window_only(synthetic_panel):
    close, _ = synthetic_panel
    ret = E.daily_returns(close)
    mu, sigma = E.rolling_moments(ret, 60, 60)
    z = E.crash_zscore(ret, mu, sigma)
    t = 200
    expected = (ret["AAA"].iloc[t] - ret["AAA"].iloc[t - 60:t].mean()) / \
               ret["AAA"].iloc[t - 60:t].std(ddof=1)
    assert np.isclose(z["AAA"].iloc[t], expected)
    assert z["AAA"].iloc[t] < -3


def test_min_prior_obs_enforced(synthetic_panel):
    """No event may be emitted before `min_prior_obs` returns exist."""
    close, _ = synthetic_panel
    ret = E.daily_returns(close)
    mu, sigma = E.rolling_moments(ret, 60, 60)
    z = E.crash_zscore(ret, mu, sigma)
    # ret[0] is NaN, so the first usable z is at index 61.
    assert z.iloc[:61].isna().all().all()
    assert z.iloc[61:].notna().any().any()


def test_momentum_features_exclude_the_crash_day(synthetic_panel, calendar):
    """Prior-return features must end at t-1, not t."""
    close, bench = synthetic_panel
    ret = E.daily_returns(close)
    t = 200
    ev = pd.DataFrame({"ticker": ["AAA"], "event_date": [calendar[t]],
                       "pos": [t], "raw_return": [ret["AAA"].iloc[t]]})
    vol = pd.DataFrame(1e6, index=calendar, columns=close.columns)
    bc = pd.DataFrame({"SPY": bench, "^VIX": pd.Series(20.0, index=calendar)})
    out = F.build_features(ev, close, close, close, vol, ret, bc,
                           pd.DataFrame(index=calendar), pd.Series(dtype=object))
    expected_m20 = close["AAA"].iloc[t - 1] / close["AAA"].iloc[t - 21] - 1
    assert np.isclose(out["mom_20"].iloc[0], expected_m20)
    # A -20% crash-day move must not appear in the pre-crash momentum.
    assert out["mom_20"].iloc[0] > -0.15


def test_feature_matrix_excludes_outcomes_and_forward_descriptors():
    """No outcome or forward-looking descriptor may reach the predictor matrix."""
    cols = set(F.FEATURE_COLUMNS)
    for pref in E.OUTCOME_PREFIXES:
        assert not any(c.startswith(pref) for c in cols), pref
    for c in E.DESCRIPTIVE_FORWARD:
        assert c not in cols
    for banned in ("recovered_20d", "car_20", "fwd_ret_20", "mfe_20", "mae_20",
                   "days_to_recovery", "episode_length", "episode_min_return"):
        assert banned not in cols
