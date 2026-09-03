"""Inference primitives: determinism, coverage and FDR behaviour."""
import numpy as np
import pandas as pd

from deadcat import statistics as S


def test_bootstrap_is_seed_deterministic():
    x = np.random.default_rng(0).normal(size=400)
    a = S.bootstrap_mean(x, n_boot=500, seed=42)
    b = S.bootstrap_mean(x, n_boot=500, seed=42)
    assert a == b
    c = S.bootstrap_mean(x, n_boot=500, seed=43)
    assert c["lo"] != a["lo"]


def test_bootstrap_ci_brackets_the_sample_mean():
    x = np.random.default_rng(1).normal(0.4, 1.0, 800)
    r = S.bootstrap_mean(x, n_boot=1000, seed=3)
    assert r["lo"] < r["mean"] < r["hi"]
    assert r["lo"] > 0                     # a genuinely non-zero mean is detected


def test_bootstrap_does_not_reject_a_true_null():
    x = np.random.default_rng(5).normal(0.0, 1.0, 2000)
    r = S.bootstrap_mean(x, n_boot=1000, seed=5)
    assert r["lo"] < 0 < r["hi"]


def test_bootstrap_path_column_zero_is_degenerate():
    paths = np.cumsum(np.random.default_rng(2).normal(size=(300, 21)) * 0.01, axis=1)
    paths[:, 0] = 0.0
    out = S.bootstrap_mean_path(paths, n_boot=200, seed=1)
    assert np.isclose(out.loc[0, "mean_car"], 0.0)
    assert (out["lo"] <= out["mean_car"] + 1e-12).all()
    assert (out["hi"] >= out["mean_car"] - 1e-12).all()
    assert len(out) == 21


def test_bootstrap_diff_detects_a_real_difference():
    rng = np.random.default_rng(7)
    r = S.bootstrap_diff(rng.normal(0.5, 1, 500), rng.normal(0.0, 1, 500),
                         n_boot=1000, seed=2)
    assert r["diff"] > 0 and r["lo"] > 0 and r["p_two_sided"] < 0.05


def test_benjamini_hochberg_is_monotone_and_conservative():
    p = np.array([0.001, 0.008, 0.02, 0.04, 0.3, 0.9])
    out = S.benjamini_hochberg(p, alpha=0.05)
    assert (out["q_value"].values >= p - 1e-12).all()
    assert out["q_value"].is_monotonic_increasing
    assert out["reject_fdr"].iloc[0]
    assert not out["reject_fdr"].iloc[-1]


def test_benjamini_hochberg_controls_a_pure_null():
    p = np.random.default_rng(11).uniform(size=500)
    out = S.benjamini_hochberg(p, alpha=0.05)
    assert out["reject_fdr"].sum() <= 5


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(4)
    n = 4000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = 0.5 + 2.0 * x1 - 1.0 * x2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": rng.integers(0, 50, n)})
    r = S.ols_hc3(df, "y", ["x1", "x2"], cluster="g")
    t = r["table"].set_index("term")
    assert abs(t.loc["x1", "coef"] - 2.0) < 0.05
    assert abs(t.loc["x2", "coef"] + 1.0) < 0.05
    assert t.loc["x1", "ci_lo"] < 2.0 < t.loc["x1", "ci_hi"]
    assert "se_cluster_date" in r["table"].columns
    assert r["adj_r2"] > 0.9


def test_logit_odds_ratio_direction():
    rng = np.random.default_rng(6)
    n = 4000
    x = rng.normal(size=n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.8 * x)))).astype(float)
    df = pd.DataFrame({"y": y, "x": x})
    r = S.logit_odds(df, "y", ["x"])
    t = r["table"].set_index("term")
    assert t.loc["x", "odds_ratio_per_sd"] > 1.5
    assert t.loc["x", "or_ci_lo"] > 1.0
    assert t.loc["x", "p_value"] < 1e-6
