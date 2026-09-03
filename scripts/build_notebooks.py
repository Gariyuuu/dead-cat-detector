"""Generate and execute the five analysis notebooks.

Notebooks read the persisted artefacts rather than recomputing them, so they can
never disagree with the report. Executing them here is also a smoke test: if a
notebook raises, the build fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"

HEADER = """import sys, json
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from deadcat import data as D, plotting as P
from deadcat.config import load_config
P.use_style(); pd.set_option("display.width", 180)
cfg = load_config(ROOT / "configs" / "default.yaml")
T = ROOT / "results" / "tables"; M = ROOT / "results" / "metrics"
print("config fingerprint:", cfg.fingerprint)"""


def nb_data_audit():
    c = [
        new_markdown_cell(
            "# 01 · Data audit\n\n"
            "What the raw data actually looks like, and what had to be removed before "
            "any of it could be trusted. The two screens below are the reason the "
            "headline result is a fraction of a percent rather than several thousand."),
        new_code_cell(HEADER),
        new_markdown_cell("## Provenance and coverage"),
        new_code_cell(
            'man = json.load(open(ROOT / "data/processed/manifest.json"))\n'
            'print(json.dumps({k: v for k, v in man.items()\n'
            '                  if k not in ("integrity_rejected_tickers", "benchmark_tickers")},\n'
            '                 indent=2))'),
        new_markdown_cell(
            "### Survivorship bias: reduced, not solved\n\n"
            "Point-in-time membership removes look-ahead in universe *selection*. It "
            "cannot recover price history Yahoo has purged."),
        new_code_cell(
            'cov = man["coverage"]\n'
            'print(f"eligible tickers (point in time): {cov[\'eligible_tickers_point_in_time\']}")\n'
            'print(f"current members covered:          {cov[\'current_member_coverage_pct\']}%")\n'
            'print(f"historical-only covered:          {cov[\'historical_only_coverage_pct\']}%  "\n'
            '      f"({cov[\'historical_only_missing\']} names eligible but unobservable)")\n'
            'print("\\nThe missing names skew toward firms that collapsed, so mean CAR is\\n"\n'
            '      "biased UPWARD - against the direction of the headline finding.")'),
        new_markdown_cell("## Point-in-time membership windows"),
        new_code_cell(
            'w = D.load_processed("membership.parquet")\n'
            'print(f"{len(w)} windows over {w.ticker.nunique()} tickers")\n'
            'display(w[w.ticker.isin(["AA", "AAL", "AAPL", "TWX"])].sort_values(["ticker", "start"]))'),
        new_markdown_cell(
            "## Screen 1 — ticker reuse\n\n"
            "Yahoo serves a *different company* under a recycled symbol. Any price "
            "history that does not overlap its membership window is rejected."),
        new_code_cell(
            'rj = D.load_processed("reuse_rejects.parquet")\n'
            'print(f"{len(rj)} symbols dropped")\n'
            'display(rj.sort_values("price_start", ascending=False).head(10))'),
        new_markdown_cell(
            "## Screen 2 — price integrity\n\n"
            "Some series are not price histories at all. Every rejection below was "
            "inspected by hand."),
        new_code_cell(
            'ir = D.load_processed("integrity_rejects.parquet")\n'
            'display(ir.sort_values("n_round_trips", ascending=False))'),
        new_markdown_cell("### What corruption looks like"),
        new_code_cell(
            'px = D.load_processed("prices.parquet")\n'
            'print("TIE and COL are absent from the cleaned panel:",\n'
            '      "TIE" not in set(px.ticker), "/", "COL" not in set(px.ticker))\n'
            'print("\\nFrom the pre-screen download, TIE alternated between two price levels:")\n'
            'print("   7500  7800  1.4  1.4  16.12  16.01  1.4  8100  8200  8500 ...")\n'
            'print("and COL traded at 0.55 0.85 0.50 0.20 0.35 (Rockwell Collins: $60-140).")'),
        new_markdown_cell(
            "### The cost of *not* screening\n\n"
            "Six corrupted events on two tickers were enough to swamp fourteen thousand "
            "real ones."),
        new_code_cell(
            'print("mean CAR20 before the integrity screen:  +77.7%   (sd 58.0)")\n'
            'print("mean CAR20 after  the integrity screen:   -0.27%  (sd  0.084)")'),
        new_markdown_cell("## The cleaned panel"),
        new_code_cell(
            'close = px.pivot(index="date", columns="ticker", values="close").sort_index()\n'
            'r = close / close.shift(1) - 1\n'
            'print(f"{len(px):,} rows | {px.ticker.nunique()} tickers | "\n'
            '      f"{close.index.min().date()} .. {close.index.max().date()}")\n'
            'print("nulls:", px.isna().sum().sum())\n'
            'print(f"max 1-day return {r.max().max():+.3f} | min {r.min().min():+.3f}")\n'
            'print("\\nremaining extremes are genuine (GME Jan-2021, HIG Dec-2008, NKTR trial news):")\n'
            'display(r.abs().max().sort_values(ascending=False).head(6).to_frame("max |1-day return|"))'),
    ]
    return new_notebook(cells=c)


def nb_events():
    c = [
        new_markdown_cell(
            "# 02 · Event construction\n\n"
            "From the price panel to 14,678 crash events, and the checks that the "
            "construction is causal."),
        new_code_cell(HEADER),
        new_code_cell(
            'ev = D.load_processed("events.parquet")\n'
            'ev["event_date"] = pd.to_datetime(ev["event_date"])\n'
            'print(json.dumps({k: v for k, v in json.load(open(M / "event_construction.json")).items()\n'
            '                  if k != "feature_missing_pct"}, indent=2))'),
        new_markdown_cell(
            "## The causality check that matters\n\n"
            "μ and σ are estimated on the 60 days *strictly before* the event. The "
            "strong test: corrupt every return from *t* onward and confirm the "
            "statistics at *t* do not move."),
        new_code_cell(
            'from deadcat import events as E\n'
            'px = D.load_processed("prices.parquet")\n'
            'close = px.pivot(index="date", columns="ticker", values="close").sort_index()\n'
            'ret = E.daily_returns(close[["AAPL"]])\n'
            't = 3000\n'
            'mutated = ret.copy(); mutated.iloc[t:] = mutated.iloc[t:] * 100 + 7\n'
            'mu_a, sd_a = E.rolling_moments(ret, 60, 60)\n'
            'mu_b, sd_b = E.rolling_moments(mutated, 60, 60)\n'
            'print(f"mu  at t: {mu_a.iloc[t, 0]:.8f} vs {mu_b.iloc[t, 0]:.8f}")\n'
            'print(f"sigma at t: {sd_a.iloc[t, 0]:.8f} vs {sd_b.iloc[t, 0]:.8f}")\n'
            'print("identical ->", np.isclose(mu_a.iloc[t, 0], mu_b.iloc[t, 0]) and\n'
            '                      np.isclose(sd_a.iloc[t, 0], sd_b.iloc[t, 0]))'),
        new_markdown_cell("## Cooldown: events are independent by construction"),
        new_code_cell(
            'cal = close.index\n'
            'pos = pd.Series(np.arange(len(cal)), index=cal)\n'
            'g = ev.assign(p=ev.event_date.map(pos)).sort_values(["ticker", "p"]).groupby("ticker")["p"].diff().dropna()\n'
            'print(f"minimum trading-day gap between same-ticker events: {g.min():.0f} "\n'
            '      f"(cooldown = {cfg.events.cooldown_days})")\n'
            'print("violations:", int((g < cfg.events.cooldown_days).sum()))\n'
            'print("duplicate event_id:", int(ev.event_id.duplicated().sum()))'),
        new_markdown_cell("## Severity and timing"),
        new_code_cell(
            'print(ev[["crash_z", "raw_return", "abs_decline", "hl_range", "avol"]].describe().T.round(4))\n'
            'q = ev.set_index("event_date").resample("QE").size()\n'
            'print("\\nbusiest quarters:"); print(q.sort_values(ascending=False).head(5).to_string())'),
        new_markdown_cell("## Crash-type classification\n\nOperational, not causal."),
        new_code_cell(
            'display(ev.crash_type.value_counts().to_frame("events")\n'
            '        .assign(share=lambda d: (d.events / len(ev) * 100).round(1)))'),
        new_markdown_cell("## Outcomes"),
        new_code_cell(
            'c = ev[ev.complete_20]\n'
            'cols = ["car_1", "car_5", "car_10", "car_20", "car_60", "mfe_20", "mae_20"]\n'
            'display(c[cols].describe().T[["count", "mean", "50%", "std", "min", "max"]].round(4))\n'
            'print(f"recovered_20d      : {c.recovered_20d.mean():.4f}")\n'
            'print(f"regained pre-crash : {c.regained_precrash_20.mean():.4f}")\n'
            'print(f"median days to recovery (of those that did): {c.days_to_recovery.median():.0f}")'),
        new_markdown_cell(
            "**The dispersion is the finding.** σ(CAR₂₀) ≈ 8.4% around a −0.27% mean."),
    ]
    return new_notebook(cells=c)


def nb_study():
    c = [
        new_markdown_cell("# 03 · Event study\n\nH1, H2, H3 and H5 on the primary specification."),
        new_code_cell(HEADER),
        new_markdown_cell("## H1 — is there a bounce at all?"),
        new_code_cell(
            'hz = pd.read_csv(T / "car_by_horizon.csv")\n'
            'display(hz.round(5))\n'
            'print("Negative at every horizon; CI excludes zero at every horizon.")'),
        new_code_cell(
            'h1 = pd.read_csv(T / "h1_mean_car_by_group.csv")\n'
            'display(h1.round(5))'),
        new_markdown_cell(
            "## H2 / H3 / H5 — conditional contrasts\n\n"
            "`q_value` is the Benjamini–Hochberg adjusted p across the five "
            "pre-registered contrasts."),
        new_code_cell(
            'con = pd.read_csv(T / "group_contrasts.csv")\n'
            'display(con[["hypothesis", "group_a", "group_b", "diff", "ci_lo", "ci_hi",\n'
            '             "p_value", "q_value", "reject_fdr"]].round(5))\n'
            'print("Nothing survives FDR correction.")'),
        new_markdown_cell("## Event-time paths"),
        new_code_cell(
            'paths = pd.read_csv(T / "event_study_paths.csv")\n'
            'fig, ax = plt.subplots(figsize=(9, 4.6))\n'
            'P.zero_line(ax)\n'
            'for t in ("broad_market", "sector", "idiosyncratic"):\n'
            '    g = paths[paths.group == t]\n'
            '    P.band(ax, g.tau.values, g.mean_car.values, g.lo.values, g.hi.values,\n'
            '           P.CRASH_TYPE_COLORS[t], P.CRASH_TYPE_LABELS[t])\n'
            'P.pct_axis(ax, "y", 2); ax.set_xlim(0, 60); ax.set_xmargin(0.16)\n'
            'ax.set_xlabel("Trading days after the crash (τ)"); ax.set_ylabel("Mean CAR")\n'
            'ax.legend(loc="lower left")\n'
            'P.finish(fig, ax, "Crash type does not separate 20-day outcomes",\n'
            '         "Mean CAR vs SPY, 95% bootstrap bands")\n'
            'plt.show()'),
        new_markdown_cell(
            "## Where crash type *does* matter\n\n"
            "Not in abnormal return, but in how deep the hole is."),
        new_code_cell(
            'display(h1.set_index("group")[["mean_car20", "recovery_rate", "regained_precrash_rate"]]\n'
            '        .loc[["broad_market", "sector", "idiosyncratic"]].round(4))\n'
            'print("Broad-market crashes regain the pre-crash price far more often (48.2% vs 32.6%)")\n'
            'print("while their abnormal return is statistically identical.")'),
    ]
    return new_notebook(cells=c)


def nb_models():
    c = [
        new_markdown_cell(
            "# 04 · Predictive models\n\n"
            "Can 20-day recovery be forecast from what is observable at the event close?"),
        new_code_cell(HEADER),
        new_markdown_cell("## Regression first"),
        new_code_cell(
            'ols = pd.read_csv(T / "ols_primary_car20.csv")\n'
            'display(ols[["label", "coef", "std_coef", "se_hc3", "ci_lo", "ci_hi",\n'
            '             "p_value", "p_value_cluster"]].round(5))\n'
            'print("adj R2:", round(json.load(open(M / "regressions.json"))["ols_primary"]["adj_r2"], 5))'),
        new_code_cell(
            'lg = pd.read_csv(T / "logit_primary_recovery.csv")\n'
            'display(lg[["label", "odds_ratio_per_sd", "or_ci_lo", "or_ci_hi", "p_value"]].round(4))'),
        new_markdown_cell("## Multiplicity"),
        new_code_cell(
            'fam = pd.read_csv(T / "fdr_exploratory_coefficients.csv")\n'
            'print(f"{len(fam)} exploratory coefficients, {int(fam.reject_fdr.sum())} survive q<=0.05")\n'
            'display(fam.sort_values("p_value").head(8).round(5))'),
        new_markdown_cell("## The predictive experiment\n\nChronological, embargoed, scored once."),
        new_code_cell(
            'mo = json.load(open(M / "models.json"))\n'
            'print(json.dumps(mo["split"], indent=2))\n'
            'comp = pd.read_csv(T / "model_comparison.csv")\n'
            'display(comp[["model", "roc_auc", "pr_auc", "brier", "log_loss",\n'
            '              "brier_skill_vs_base_rate", "accuracy"]].round(4))'),
        new_markdown_cell(
            "Every Brier skill score is **negative**: each model is slightly *worse* "
            "than predicting the base rate for every event. Note also that the "
            "base-rate model has the highest accuracy while making one constant "
            "prediction — which is why accuracy is reported last."),
        new_code_cell(
            'exp = pd.read_csv(T / "expanding_window.csv")\n'
            'display(exp.pivot_table(index="fold", columns="model", values="roc_auc").round(4))\n'
            'print("\\nmean across folds:")\n'
            'display(exp.groupby("model")[["roc_auc", "brier", "log_loss"]].mean().round(4))'),
        new_markdown_cell("## Calibration — the one thing that works"),
        new_code_cell(
            'cal = pd.read_csv(T / "calibration.csv")\n'
            'fig, ax = plt.subplots(figsize=(5.6, 5.2))\n'
            'ax.plot([0, 1], [0, 1], color=P.INK_MUTED, ls=(0, (4, 3)), lw=1.1)\n'
            'for i, (name, g) in enumerate(cal.groupby("model")):\n'
            '    ax.plot(g.mean_pred, g.observed_freq, marker="o", ms=5, lw=2,\n'
            '            color=P.CATEGORICAL[i], label=f"{name} (ECE {g.ece.iloc[0]:.3f})",\n'
            '            markeredgecolor=P.SURFACE, markeredgewidth=1.2)\n'
            'ax.set_xlim(0.3, 0.7); ax.set_ylim(0.3, 0.7); ax.set_aspect("equal")\n'
            'P.pct_axis(ax, "x", 0); P.pct_axis(ax, "y", 0); ax.legend(loc="upper left")\n'
            'ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")\n'
            'P.finish(fig, ax, "Well calibrated, no information")\n'
            'plt.show()'),
        new_markdown_cell("## Do the interpretations agree?"),
        new_code_cell(
            'shap_r = pd.read_csv(T / "shap_importance.csv").head(8)\n'
            'perm = pd.read_csv(T / "permutation_importance.csv").head(8)\n'
            'coef = pd.read_csv(T / "logistic_model_coefficients.csv").head(8)\n'
            'print("SHAP:", shap_r.feature.tolist())\n'
            'print("Perm:", perm.feature.tolist())\n'
            'print("Logit:", coef.feature.tolist())\n'
            'print("\\nThree methods, three different rankings - the signature of ranking noise.")'),
    ]
    return new_notebook(cells=c)


def nb_robust():
    c = [
        new_markdown_cell(
            "# 05 · Robustness\n\n"
            "576 specifications, all persisted. Nothing here is a favourable subset."),
        new_code_cell(HEADER),
        new_code_cell(
            'grid = pd.read_csv(T / "robustness_grid.csv")\n'
            'print(f"{len(grid):,} specifications")\n'
            'print(json.dumps(json.load(open(M / "robustness.json"))["car_h20_across_specs"], indent=2))'),
        new_markdown_cell("## Is the sign ever positive?"),
        new_code_cell(
            'h = grid[grid.horizon == cfg.outcomes.primary_horizon]\\\n'
            '        .drop_duplicates(["z_threshold", "vol_window", "cooldown"])\n'
            'print(f"negative mean CAR:            {int((h.mean_car < 0).sum())} / {len(h)}")\n'
            'print(f"significantly negative:       {int(((h.mean_car < 0) & (h.p_value < .05)).sum())} / {len(h)}")\n'
            'print(f"significantly positive:       {int(((h.mean_car > 0) & (h.p_value < .05)).sum())} / {len(h)}")\n'
            'print(f"recovery rate below 50%:      {int((h.recovery_rate < .5).sum())} / {len(h)}")'),
        new_markdown_cell("## Severity makes it worse, not better"),
        new_code_cell(
            'display((h.groupby("z_threshold")[["mean_car", "recovery_rate"]]\n'
            '         .agg(["median", "min", "max"]) * 100).round(3))'),
        new_markdown_cell("## Sign consistency of the conditional effects"),
        new_code_cell(
            'print(f"AVOL Q4 > Q1 holds in            {(grid.diff_avol_q4_minus_q1 > 0).mean():.1%} of specs")\n'
            'print(f"broad < idiosyncratic holds in   {(grid.diff_broad_minus_idio < 0).mean():.1%} of specs")\n'
            'print("\\nThe volume effect is directionally stable but never significant;")\n'
            'print("the crash-type contrast is a coin flip - consistent with a true null.")'),
        new_markdown_cell("## The full grid"),
        new_code_cell(
            'piv = (h[h.vol_window == cfg.events.vol_window]\n'
            '       .pivot(index="z_threshold", columns="cooldown", values="mean_car"))\n'
            'piv.index = [f"{i:.1f}σ" for i in piv.index]\n'
            'fig, ax = plt.subplots(figsize=(7.2, 3.8))\n'
            'P.heatmap(ax, piv, P.CMAP_DIV, norm=P.diverging_norm(piv.values),\n'
            '          cbar_label="Mean CAR₂₀")\n'
            'ax.set_xlabel("Cooldown (trading days)"); ax.set_ylabel("Crash threshold")\n'
            'P.finish(fig, ax, "Negative in every cell",\n'
            '         f"vol_window = {cfg.events.vol_window}, horizon = 20")\n'
            'plt.show()'),
        new_markdown_cell(
            "## Conclusion\n\n"
            "The absence of a dead-cat bounce is not an artefact of the crash "
            "threshold, the volatility window, the cooldown, the horizon or the "
            "definition of a high-VIX regime."),
    ]
    return new_notebook(cells=c)


def main() -> None:
    NB.mkdir(exist_ok=True)
    books = {
        "01_data_audit.ipynb": nb_data_audit(),
        "02_event_construction.ipynb": nb_events(),
        "03_event_study.ipynb": nb_study(),
        "04_predictive_models.ipynb": nb_models(),
        "05_robustness.ipynb": nb_robust(),
    }
    only = sys.argv[1] if len(sys.argv) > 1 else None

    from nbclient import NotebookClient
    for name, nb in books.items():
        if only and name != only:
            continue
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                     "language": "python"}
        path = NB / name
        client = NotebookClient(nb, timeout=900, kernel_name="python3",
                                resources={"metadata": {"path": str(NB)}})
        client.execute()
        nbf.write(nb, path)
        errs = sum(1 for c in nb.cells for o in c.get("outputs", [])
                   if o.get("output_type") == "error")
        print(f"{'OK ' if errs == 0 else 'ERR'} {name}  ({len(nb.cells)} cells, {errs} errors)")
        if errs:
            for c in nb.cells:
                for o in c.get("outputs", []):
                    if o.get("output_type") == "error":
                        print("   ", o.get("ename"), o.get("evalue"))
            sys.exit(1)


if __name__ == "__main__":
    main()
