"""Predictive experiment: chronological splitting and model fitting.

Splits are strictly chronological and *embargoed*. A naive time split still
leaks: the last training event's 20-day outcome window extends into the
validation period, so its label is partly determined by prices the validation
features already see. An embargo of ``primary_horizon`` trading days between
consecutive blocks removes that overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    boundaries: dict = field(default_factory=dict)


def chronological_split(
    dates: pd.Series,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo_days: int = 20,
) -> Split:
    """Ordered train/val/test masks with an embargo gap between blocks.

    ``embargo_days`` is expressed in *calendar* days derived from the event
    date ordering: events falling inside the embargo window after a boundary
    are dropped from every block, so no training label depends on a price
    observed inside the next block.
    """
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    order = np.argsort(d.values, kind="stable")
    n = len(d)
    i_tr = int(np.floor(n * train_frac))
    i_va = int(np.floor(n * (train_frac + val_frac)))

    cut_tr = d.iloc[order[i_tr]]
    cut_va = d.iloc[order[i_va]]
    emb = pd.Timedelta(days=int(embargo_days * 7 / 5) + 1)  # trading -> calendar

    train = (d < cut_tr - emb).to_numpy()
    val = ((d >= cut_tr) & (d < cut_va - emb)).to_numpy()
    test = (d >= cut_va).to_numpy()
    return Split(train, val, test, {
        "train_end": str(cut_tr.date()), "val_end": str(cut_va.date()),
        "embargo_calendar_days": int(emb.days),
        "n_train": int(train.sum()), "n_val": int(val.sum()), "n_test": int(test.sum()),
        "n_embargoed": int(n - train.sum() - val.sum() - test.sum()),
    })


def expanding_window_folds(dates: pd.Series, n_folds: int = 5,
                           min_train_frac: float = 0.4, embargo_days: int = 20):
    """Expanding-window evaluation folds: train on all history, test the next block."""
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    n = len(d)
    order = np.argsort(d.values, kind="stable")
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, n_folds + 1).astype(int)
    emb = pd.Timedelta(days=int(embargo_days * 7 / 5) + 1)
    folds = []
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        if hi - lo < 50:
            continue
        cut_lo = d.iloc[order[lo]]
        cut_hi = d.iloc[order[hi - 1]]
        tr = (d < cut_lo - emb).to_numpy()
        te = ((d >= cut_lo) & (d <= cut_hi)).to_numpy()
        if tr.sum() > 200 and te.sum() > 50:
            folds.append((tr, te, {"train_end": str((cut_lo - emb).date()),
                                   "test_start": str(cut_lo.date()),
                                   "test_end": str(cut_hi.date()),
                                   "n_train": int(tr.sum()), "n_test": int(te.sum())}))
    return folds


# ------------------------------------------------------------------ models ---
def make_logistic(seed: int, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=C, random_state=seed)),
    ])


def make_random_forest(seed: int, min_samples_leaf: int = 20, n_estimators: int = 500) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            max_features="sqrt", n_jobs=-1, random_state=seed)),
    ])


def make_lightgbm(seed: int, n_estimators: int = 400, learning_rate: float = 0.03,
                  num_leaves: int = 15, min_child_samples: int = 40) -> Pipeline:
    import lightgbm as lgb

    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            num_leaves=num_leaves, min_child_samples=min_child_samples,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1)),
    ])


MODEL_BUILDERS = {
    "logistic": make_logistic,
    "random_forest": make_random_forest,
    "lightgbm": make_lightgbm,
}

# Small, honest hyper-parameter grids searched on the validation block only.
PARAM_GRIDS = {
    "logistic": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "random_forest": [{"min_samples_leaf": m} for m in (10, 20, 50, 100)],
    "lightgbm": [{"n_estimators": n, "learning_rate": lr, "num_leaves": nl}
                 for n, lr, nl in [(200, 0.03, 15), (400, 0.03, 15),
                                   (400, 0.01, 31), (800, 0.01, 15)]],
}


def base_rate_prediction(y_train: np.ndarray, n: int) -> np.ndarray:
    """Constant-probability reference model: the training base rate."""
    return np.full(n, float(np.mean(y_train)))
