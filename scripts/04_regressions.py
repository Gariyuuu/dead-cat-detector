"""Cross-sectional regressions for CAR_20 and recovery probability.

The pre-registered specification is estimated first, exactly as written in the
research plan. An extended specification using the full feature set follows as
an exploratory model, and every exploratory coefficient enters one
Benjamini-Hochberg family so the multiple-comparison cost is paid explicitly.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deadcat import data as D, features as F, statistics as S  # noqa: E402
from deadcat.config import PATHS, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("regressions")

# The pre-registered right-hand side.
PRIMARY_X = ["crash_z", "avol", "mom_20", "rv_20", "spy_ret_event", "vix"]
PRETTY = {
    "crash_z": "Crash z-score", "avol": "Abnormal volume", "mom_20": "Momentum 20d",
    "rv_20": "Realised vol 20d", "spy_ret_event": "Market return (event day)",
    "vix": "VIX", "const": "Intercept",
}


def main() -> None:
    cfg = load_config()
    PATHS.ensure()
    H = int(cfg.outcomes.primary_horizon)
    y_c, y_b = f"car_{H}", f"recovered_{H}d"

    ev = D.load_processed("events.parquet")
    df = ev[ev["complete_20"]].copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    log.info("estimation sample: %d events, %d tickers, %d distinct dates",
             len(df), df.ticker.nunique(), df.event_date.nunique())

    out: dict = {"config_fingerprint": cfg.fingerprint, "n_events": int(len(df))}

    # ---------------- primary OLS ----------------
    ols = S.ols_hc3(df, y_c, PRIMARY_X, cluster="event_date")
    t = ols["table"].copy()
    t["label"] = t["term"].map(PRETTY).fillna(t["term"])
    t.to_csv(PATHS.tables / "ols_primary_car20.csv", index=False)
    out["ols_primary"] = {"n": ols["n"], "r2": ols["r2"], "adj_r2": ols["adj_r2"],
                          "f_pvalue": ols["f_pvalue"]}
    log.info("PRIMARY OLS  n=%d  adj_R2=%.5f", ols["n"], ols["adj_r2"])
    log.info("\n%s", t[["label", "coef", "std_coef", "se_hc3", "ci_lo", "ci_hi",
                        "p_value", "se_cluster_date", "p_value_cluster"]]
             .round(5).to_string(index=False))

    # ---------------- primary logistic ----------------
    lg = S.logit_odds(df, y_b, PRIMARY_X, cluster="event_date")
    lt = lg["table"].copy()
    lt["label"] = lt["term"].map(PRETTY).fillna(lt["term"])
    lt.to_csv(PATHS.tables / "logit_primary_recovery.csv", index=False)
    out["logit_primary"] = {"n": lg["n"], "pseudo_r2": lg["pseudo_r2"],
                            "base_rate": float(df[y_b].mean())}
    log.info("PRIMARY LOGIT  n=%d  pseudo_R2=%.5f  base_rate=%.4f",
             lg["n"], lg["pseudo_r2"], df[y_b].mean())
    log.info("\n%s", lt[["label", "odds_ratio_per_sd", "or_ci_lo", "or_ci_hi", "p_value"]]
             .round(4).to_string(index=False))

    # ---------------- extended (exploratory) ----------------
    cand_x = [c for c in F.FEATURE_COLUMNS
              if c in df.columns and c not in ("volume", "med_volume_60", "log_volume")]
    ext_x, aliased = S.drop_aliased(df, cand_x)
    if aliased:
        log.info("dropped %d aliased predictors from the extended model: %s",
                 len(aliased), aliased)
    out["extended_dropped_aliased"] = aliased
    ols_ext = S.ols_hc3(df, y_c, ext_x, cluster="event_date")
    lg_ext = S.logit_odds(df, y_b, ext_x, cluster="event_date")
    ols_ext["table"].to_csv(PATHS.tables / "ols_extended_car20.csv", index=False)
    lg_ext["table"].to_csv(PATHS.tables / "logit_extended_recovery.csv", index=False)
    out["ols_extended"] = {"n": ols_ext["n"], "adj_r2": ols_ext["adj_r2"],
                           "n_predictors": len(ext_x)}
    out["logit_extended"] = {"n": lg_ext["n"], "pseudo_r2": lg_ext["pseudo_r2"]}
    log.info("EXTENDED OLS n=%d adj_R2=%.5f | EXTENDED LOGIT pseudo_R2=%.5f",
             ols_ext["n"], ols_ext["adj_r2"], lg_ext["pseudo_r2"])

    # ---------------- one FDR family over all exploratory coefficients ----------------
    fam = []
    for model, tbl, pcol in [
        ("ols_extended", ols_ext["table"], "p_value_cluster"),
        ("logit_extended", lg_ext["table"], "p_value"),
    ]:
        sub = tbl[tbl.term != "const"]
        pc = pcol if pcol in sub.columns else "p_value"
        for r in sub.itertuples(index=False):
            fam.append({"model": model, "term": r.term, "p_value": getattr(r, pc)})
    fam = pd.DataFrame(fam)
    fdr = S.benjamini_hochberg(fam["p_value"].values, alpha=0.05)
    fam["q_value"], fam["reject_fdr"] = fdr["q_value"], fdr["reject_fdr"]
    fam.to_csv(PATHS.tables / "fdr_exploratory_coefficients.csv", index=False)
    out["fdr"] = {"n_tests": int(len(fam)), "n_reject": int(fam.reject_fdr.sum())}
    log.info("FDR family: %d tests, %d survive q<=0.05", len(fam), fam.reject_fdr.sum())
    log.info("\n%s", fam[fam.reject_fdr].sort_values("q_value").round(6).to_string(index=False))

    # ---------------- crash-type fixed effects ----------------
    dd = pd.get_dummies(df["crash_type"], prefix="type", drop_first=False).astype(float)
    df2 = pd.concat([df.reset_index(drop=True), dd.reset_index(drop=True)], axis=1)
    type_x = [c for c in ("type_broad_market", "type_sector") if c in df2.columns]
    ols_type = S.ols_hc3(df2, y_c, type_x + PRIMARY_X, cluster="event_date")
    ols_type["table"].to_csv(PATHS.tables / "ols_crash_type.csv", index=False)
    out["ols_crash_type"] = {"n": ols_type["n"], "adj_r2": ols_type["adj_r2"]}
    log.info("CRASH-TYPE MODEL (idiosyncratic is the omitted base)\n%s",
             ols_type["table"][ols_type["table"].term.isin(type_x)]
             [["term", "coef", "se_hc3", "p_value", "p_value_cluster"]].round(5).to_string(index=False))

    (PATHS.metrics / "regressions.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
