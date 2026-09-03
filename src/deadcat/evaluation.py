"""Probabilistic evaluation and model interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """Discrimination, calibration and sharpness. Accuracy is reported last, on purpose."""
    y = np.asarray(y_true, float)
    p = np.clip(np.asarray(y_prob, float), 1e-9, 1 - 1e-9)
    out = {
        "n": int(y.size),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "accuracy": float(accuracy_score(y, (p >= threshold).astype(int))),
    }
    # Skill relative to always predicting the sample base rate.
    base = np.full_like(p, y.mean())
    out["brier_skill_vs_base_rate"] = float(1 - out["brier"] / brier_score_loss(y, base))
    return out


def calibration_table(y_true, y_prob, n_bins: int = 10, strategy: str = "quantile") -> pd.DataFrame:
    """Reliability table: predicted vs realised frequency per probability bin."""
    y = np.asarray(y_true, float)
    p = np.asarray(y_prob, float)
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(p.min(), p.max(), n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin": b, "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                     "observed_freq": float(y[m].mean()),
                     "lo_edge": float(edges[b]), "hi_edge": float(edges[b + 1])})
    tbl = pd.DataFrame(rows)
    if len(tbl):
        w = tbl["n"] / tbl["n"].sum()
        tbl.attrs["ece"] = float((w * (tbl["mean_pred"] - tbl["observed_freq"]).abs()).sum())
    return tbl


def permutation_importances(model, X: pd.DataFrame, y, n_repeats: int = 10,
                            seed: int = 0, scoring: str = "roc_auc") -> pd.DataFrame:
    r = permutation_importance(model, X, y, n_repeats=n_repeats,
                               random_state=seed, scoring=scoring, n_jobs=-1)
    return pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def shap_values_for(model_pipeline, X: pd.DataFrame, max_samples: int = 2000, seed: int = 0):
    """SHAP values for the tree stage of a fitted pipeline.

    The imputer is applied first so SHAP sees exactly the matrix the estimator
    consumed. Returns ``(shap_matrix, X_used)``.
    """
    import shap

    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        Xs = X.iloc[np.sort(idx)]
    else:
        Xs = X
    Xt = pd.DataFrame(model_pipeline[:-1].transform(Xs), columns=Xs.columns, index=Xs.index)
    explainer = shap.TreeExplainer(model_pipeline[-1])
    sv = explainer.shap_values(Xt)
    if isinstance(sv, list):          # older API: one matrix per class
        sv = sv[1]
    elif sv.ndim == 3:                # newer API: (n, features, classes)
        sv = sv[:, :, 1]
    return np.asarray(sv), Xt
