"""Predictive experiment: can recovery be forecast from event-time features?

Chronological, embargoed splits. Hyper-parameters are chosen on the validation
block only; the test block is touched once. A constant base-rate forecast is
carried through as the reference every model must beat.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deadcat import data as D, evaluation as V, features as F, models as M  # noqa: E402
from deadcat.config import PATHS, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("models")


def main() -> None:
    cfg = load_config()
    PATHS.ensure()
    seed = int(cfg.models.seed)
    H = int(cfg.outcomes.primary_horizon)
    target = f"recovered_{H}d"

    ev = D.load_processed("events.parquet")
    df = ev[ev["complete_20"]].copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    feats = [c for c in F.FEATURE_COLUMNS if c in df.columns]
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    y = df[target].astype(int).to_numpy()

    # Guard in code, not just in prose: no outcome may enter the design matrix.
    from deadcat.events import DESCRIPTIVE_FORWARD, OUTCOME_PREFIXES
    for c in feats:
        assert not c.startswith(OUTCOME_PREFIXES), c
        assert c not in DESCRIPTIVE_FORWARD, c

    split = M.chronological_split(df["event_date"], cfg.models.train_frac,
                                  cfg.models.val_frac, embargo_days=H)
    log.info("split: %s", split.boundaries)
    log.info("base rates  train=%.4f val=%.4f test=%.4f",
             y[split.train].mean(), y[split.val].mean(), y[split.test].mean())

    Xtr, ytr = X[split.train], y[split.train]
    Xva, yva = X[split.val], y[split.val]
    Xte, yte = X[split.test], y[split.test]

    results, fitted, chosen = [], {}, {}
    for name, builder in M.MODEL_BUILDERS.items():
        best = None
        for params in M.PARAM_GRIDS[name]:
            mdl = builder(seed=seed, **params).fit(Xtr, ytr)
            p_va = mdl.predict_proba(Xva)[:, 1]
            score = V.classification_metrics(yva, p_va)["roc_auc"]
            if best is None or score > best[0]:
                best = (score, params, mdl)
        val_auc, params, mdl = best
        chosen[name] = params
        # Refit on train+validation for the single test evaluation.
        final = builder(seed=seed, **params).fit(
            pd.concat([Xtr, Xva]), np.concatenate([ytr, yva]))
        fitted[name] = final
        p_te = final.predict_proba(Xte)[:, 1]
        m = V.classification_metrics(yte, p_te)
        m.update({"model": name, "val_roc_auc": val_auc, "params": json.dumps(params)})
        results.append(m)
        log.info("%-14s val_AUC=%.4f  test_AUC=%.4f  PR=%.4f  Brier=%.5f  LL=%.5f  acc=%.4f",
                 name, val_auc, m["roc_auc"], m["pr_auc"], m["brier"], m["log_loss"], m["accuracy"])

    p_base = M.base_rate_prediction(np.concatenate([ytr, yva]), len(yte))
    mb = V.classification_metrics(yte, p_base)
    mb.update({"model": "base_rate", "val_roc_auc": np.nan, "params": "{}"})
    results.append(mb)
    log.info("%-14s test_AUC=%.4f  Brier=%.5f  LL=%.5f", "base_rate",
             mb["roc_auc"], mb["brier"], mb["log_loss"])

    res = pd.DataFrame(results)[
        ["model", "n", "base_rate", "val_roc_auc", "roc_auc", "pr_auc", "brier",
         "log_loss", "brier_skill_vs_base_rate", "accuracy", "params"]]
    res.to_csv(PATHS.tables / "model_comparison.csv", index=False)

    # ---------------- calibration ----------------
    cal_frames = []
    for name, mdl in fitted.items():
        c = V.calibration_table(yte, mdl.predict_proba(Xte)[:, 1], n_bins=10)
        c.insert(0, "model", name)
        c["ece"] = c.attrs.get("ece", np.nan)
        cal_frames.append(c)
    cal = pd.concat(cal_frames, ignore_index=True)
    cal.to_csv(PATHS.tables / "calibration.csv", index=False)

    # ---------------- expanding-window evaluation ----------------
    rows = []
    for k, (tr, te, meta) in enumerate(M.expanding_window_folds(df["event_date"],
                                                               n_folds=5, embargo_days=H)):
        for name, builder in M.MODEL_BUILDERS.items():
            mdl = builder(seed=seed, **chosen[name]).fit(X[tr], y[tr])
            m = V.classification_metrics(y[te], mdl.predict_proba(X[te])[:, 1])
            rows.append({"fold": k, "model": name, **meta, **m})
    exp = pd.DataFrame(rows)
    exp.to_csv(PATHS.tables / "expanding_window.csv", index=False)
    if len(exp):
        log.info("expanding-window mean test AUC:\n%s",
                 exp.groupby("model")[["roc_auc", "brier", "log_loss"]].mean().round(4).to_string())

    # ---------------- interpretability ----------------
    perm = V.permutation_importances(fitted["lightgbm"], Xte, yte, n_repeats=15, seed=seed)
    perm.to_csv(PATHS.tables / "permutation_importance.csv", index=False)
    log.info("top permutation importances (LightGBM):\n%s", perm.head(8).round(5).to_string(index=False))

    sv, Xs = V.shap_values_for(fitted["lightgbm"], Xte, max_samples=2000, seed=seed)
    np.save(PATHS.data_processed / "shap_values.npy", sv)
    Xs.to_parquet(PATHS.data_processed / "shap_X.parquet")
    shap_rank = (pd.DataFrame({"feature": Xs.columns,
                               "mean_abs_shap": np.abs(sv).mean(axis=0)})
                 .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
    shap_rank.to_csv(PATHS.tables / "shap_importance.csv", index=False)
    log.info("top SHAP features:\n%s", shap_rank.head(8).round(6).to_string(index=False))

    # Logistic coefficients on the same split, for the statistical-vs-ML comparison.
    lr = fitted["logistic"]
    coefs = pd.DataFrame({"feature": feats,
                          "coef": lr[-1].coef_[0],
                          "odds_ratio": np.exp(lr[-1].coef_[0])}
                         ).sort_values("coef", key=np.abs, ascending=False)
    coefs.to_csv(PATHS.tables / "logistic_model_coefficients.csv", index=False)

    (PATHS.metrics / "models.json").write_text(json.dumps({
        "config_fingerprint": cfg.fingerprint,
        "target": target,
        "n_features": len(feats),
        "features": feats,
        "split": split.boundaries,
        "chosen_params": chosen,
        "test_results": res.to_dict("records"),
        "expanding_window_mean": (exp.groupby("model")[["roc_auc", "brier", "log_loss"]]
                                  .mean().round(6).to_dict() if len(exp) else {}),
        "top_shap": shap_rank.head(10).to_dict("records"),
        "top_permutation": perm.head(10).to_dict("records"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
