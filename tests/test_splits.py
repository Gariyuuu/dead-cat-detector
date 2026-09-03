"""Chronological splitting, embargo and seed determinism."""
import numpy as np
import pandas as pd

from deadcat import models as M


def _dates(n=2000):
    return pd.Series(pd.bdate_range("2007-01-01", periods=n))


def test_split_is_chronological_and_disjoint():
    d = _dates()
    s = M.chronological_split(d, 0.7, 0.15, embargo_days=20)
    assert not (s.train & s.val).any()
    assert not (s.val & s.test).any()
    assert not (s.train & s.test).any()
    assert d[s.train].max() < d[s.val].min() < d[s.test].min()
    assert d[s.val].max() < d[s.test].min()


def test_embargo_creates_a_real_gap():
    d = _dates()
    s = M.chronological_split(d, 0.7, 0.15, embargo_days=20)
    gap = (d[s.val].min() - d[s.train].max()).days
    assert gap > 20, gap
    assert s.boundaries["n_embargoed"] > 0


def test_zero_embargo_leaves_no_gap():
    d = _dates()
    s = M.chronological_split(d, 0.7, 0.15, embargo_days=0)
    assert s.boundaries["n_embargoed"] >= 0
    assert s.train.sum() + s.val.sum() + s.test.sum() <= len(d)


def test_split_fractions_are_approximately_respected():
    d = _dates()
    s = M.chronological_split(d, 0.7, 0.15, embargo_days=20)
    assert 0.60 < s.train.sum() / len(d) < 0.71
    assert 0.10 < s.test.sum() / len(d) < 0.20


def test_expanding_folds_never_train_on_the_future():
    d = _dates()
    for tr, te, meta in M.expanding_window_folds(d, n_folds=5, embargo_days=20):
        assert not (tr & te).any()
        assert d[tr].max() < d[te].min()


def test_expanding_folds_grow():
    d = _dates()
    sizes = [int(tr.sum()) for tr, _, _ in M.expanding_window_folds(d, n_folds=5)]
    assert sizes == sorted(sizes)


def test_model_seeds_are_deterministic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 6)), columns=[f"x{i}" for i in range(6)])
    y = (X["x0"] + rng.normal(0, 0.5, 600) > 0).astype(int)
    for name in ("logistic", "random_forest", "lightgbm"):
        a = M.MODEL_BUILDERS[name](seed=7).fit(X, y).predict_proba(X)[:, 1]
        b = M.MODEL_BUILDERS[name](seed=7).fit(X, y).predict_proba(X)[:, 1]
        assert np.allclose(a, b), name
