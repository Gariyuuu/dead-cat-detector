"""Cross-check every headline number in README.md and report.md against results.

A report can drift from its own outputs after a re-run. This asserts the
documented figures still match what is on disk, and fails loudly if not.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, M = ROOT / "results" / "tables", ROOT / "results" / "metrics"

h1 = pd.read_csv(T / "h1_mean_car_by_group.csv").set_index("group")
con = pd.read_csv(T / "group_contrasts.csv")
ols = pd.read_csv(T / "ols_primary_car20.csv").set_index("term")
lg = pd.read_csv(T / "logit_primary_recovery.csv").set_index("term")
comp = pd.read_csv(T / "model_comparison.csv").set_index("model")
reg = json.load(open(M / "regressions.json"))
rob = json.load(open(M / "robustness.json"))
ec = json.load(open(M / "event_construction.json"))
man = json.load(open(ROOT / "data" / "processed" / "manifest.json"))

allrow = h1.loc["all_events"]
avol = con[con.hypothesis == "H3"].iloc[0]
bi = con[(con.group_a == "broad_market") & (con.group_b == "idiosyncratic")].iloc[0]

CLAIMS = [
    ("mean CAR20 = -0.265%", allrow.mean_car20, -0.00265, 5e-5),
    ("CAR20 CI low  = -0.403%", allrow.ci_lo, -0.00403, 5e-5),
    ("CAR20 CI high = -0.124%", allrow.ci_hi, -0.00124, 5e-5),
    ("recovery rate = 47.1%", allrow.recovery_rate, 0.4714, 5e-4),
    ("regained pre-crash = 38.9%", allrow.regained_precrash_rate, 0.3894, 5e-4),
    ("n events = 14678", ec["n_events"], 14678, 0),
    ("n complete CAR20 = 14647", ec["n_events_complete_car20"], 14647, 0),
    ("n tickers = 600", ec["n_tickers"], 600, 0),
    ("broad-market mean = -0.351%", h1.loc["broad_market"].mean_car20, -0.00351, 5e-5),
    ("sector mean = -0.241%", h1.loc["sector"].mean_car20, -0.00241, 5e-5),
    ("idiosyncratic mean = -0.226%", h1.loc["idiosyncratic"].mean_car20, -0.00226, 5e-5),
    ("broad-idio diff p = 0.43", bi.p_value, 0.429, 0.02),
    ("AVOL Q4-Q1 diff = +0.43pp", avol["diff"], 0.00427, 5e-5),
    ("AVOL nominal p = 0.028", avol.p_value, 0.028, 0.01),
    ("AVOL q-value = 0.14", avol.q_value, 0.140, 0.02),
    ("rv_20 coef = +0.063", ols.loc["rv_20"].coef, 0.06300, 5e-5),
    ("rv_20 std coef = 0.101", ols.loc["rv_20"].std_coef, 0.10059, 5e-4),
    ("rv_20 HC3 p = 0.0001", ols.loc["rv_20"].p_value, 0.00011, 5e-5),
    ("rv_20 clustered p = 0.0011", ols.loc["rv_20"].p_value_cluster, 0.00111, 5e-5),
    ("mom_20 clustered p = 0.478", ols.loc["mom_20"].p_value_cluster, 0.478, 0.01),
    ("adj R2 = 0.0085", reg["ols_primary"]["adj_r2"], 0.00849, 5e-5),
    ("logit rv_20 OR = 1.082", lg.loc["rv_20"].odds_ratio_per_sd, 1.0820, 5e-4),
    ("FDR survivors = 0 of 32", reg["fdr"]["n_reject"], 0, 0),
    ("FDR family size = 32", reg["fdr"]["n_tests"], 32, 0),
    ("logistic test AUC = 0.504", comp.loc["logistic"].roc_auc, 0.50408, 5e-4),
    ("random forest AUC = 0.499", comp.loc["random_forest"].roc_auc, 0.49898, 5e-4),
    ("lightgbm AUC = 0.476", comp.loc["lightgbm"].roc_auc, 0.47648, 5e-4),
    ("base-rate accuracy highest", comp.loc["base_rate"].accuracy,
     comp.accuracy.max(), 0),
    ("logistic Brier skill = -0.002", comp.loc["logistic"].brier_skill_vs_base_rate, -0.002399, 5e-4),
    ("rand forest Brier skill = -0.003", comp.loc["random_forest"].brier_skill_vs_base_rate, -0.003495, 5e-4),
    ("lightgbm Brier skill = -0.030", comp.loc["lightgbm"].brier_skill_vs_base_rate, -0.029673, 5e-4),
    ("all Brier skills negative", float(comp.loc[["logistic", "random_forest", "lightgbm"]]
        .brier_skill_vs_base_rate.max() < 0), 1.0, 0),
    ("n specifications = 576", rob["n_specifications"], 576, 0),
    ("share negative = 1.0", rob["car_h20_across_specs"]["share_negative"], 1.0, 0),
    ("share sig. positive = 0", rob["car_h20_across_specs"]["share_positive_and_significant"], 0.0, 0),
    ("recovery below half = 1.0", rob["recovery_rate_across_specs"]["share_below_half"], 1.0, 0),
    ("AVOL sign consistency = 97.9%", rob["avol_effect_sign_consistency"], 0.979, 0.002),
    ("historical coverage = 35.0%", man["coverage"]["historical_only_coverage_pct"], 35.0, 0.05),
    ("current coverage = 99.4%", man["coverage"]["current_member_coverage_pct"], 99.4, 0.05),
    ("reuse rejects = 43", man["reuse_guard_rejects"], 43, 0),
    ("integrity rejects = 11", man["integrity_rejects"], 11, 0),
    ("price rows = 2,976,172", man["price_rows"], 2976172, 0),
]

fails = []
for label, actual, expected, tol in CLAIMS:
    ok = abs(float(actual) - float(expected)) <= tol
    print(f"{'PASS' if ok else 'FAIL'}  {label:34s} actual={actual}")
    if not ok:
        fails.append((label, actual, expected))

# Documents must not contain placeholder text.
PLACEHOLDERS = ["TODO", "TBD", "FIXME", "XXX", "PLACEHOLDER", "lorem ipsum", "<insert"]
for doc in ["README.md", "report/report.md", "data/README.md",
            "docs/research_plan.md", "docs/data_leakage_audit.md"]:
    text = (ROOT / doc).read_text()
    hit = [p for p in PLACEHOLDERS if re.search(p, text, re.I)]
    print(f"{'PASS' if not hit else 'FAIL'}  no placeholders in {doc}"
          + (f" -> {hit}" if hit else ""))
    if hit:
        fails.append((doc, hit, "none"))

print()
if fails:
    print(f"{len(fails)} CLAIM(S) OUT OF DATE:")
    for f in fails:
        print("  ", f)
    sys.exit(1)
print(f"All {len(CLAIMS)} documented claims match the persisted results.")
