"""Generate every publication figure from persisted results.

Nothing here recomputes an estimate: each figure reads the tables written by
scripts 03-06, so a figure can never disagree with the numbers in the report.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deadcat import data as D, plotting as P, statistics as S  # noqa: E402
from deadcat.config import PATHS, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("figures")

SRC = "Source: Yahoo Finance daily adjusted OHLCV; point-in-time S&P 500 membership. 2007-2026."


def fig1_frequency(ev, fig_dir):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    by_m = ev.set_index("event_date").resample("QE").size()
    ax.fill_between(by_m.index, 0, by_m.values, color=P.CATEGORICAL[0], alpha=0.18, linewidth=0)
    ax.plot(by_m.index, by_m.values, color=P.CATEGORICAL[0], linewidth=1.8)
    # Anchor each label above the local peak so it never lands on the line.
    for label, when in [("GFC", "2008-10-01"), ("Euro crisis", "2011-08-01"),
                        ("COVID", "2020-03-01"), ("2022 drawdown", "2022-06-01")]:
        d = pd.Timestamp(when)
        win = by_m[(by_m.index >= d - pd.Timedelta(days=200))
                   & (by_m.index <= d + pd.Timedelta(days=200))]
        if win.empty:
            continue
        peak = win.idxmax()
        ax.annotate(label, xy=(peak, win.max()), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=P.INK_2)
    ax.set_ylabel("Crash events per quarter")
    ax.set_xlabel("")
    ax.margins(x=0.01)
    ax.set_ymargin(0.14)
    P.finish(fig, ax, "Crash events cluster in market-wide stress episodes",
             f"Quarterly count of z ≤ {-3.0:.1f} single-day declines, "
             f"{len(ev):,} events, 20-day cooldown", SRC)
    return P.save(fig, fig_dir / "fig01_event_frequency.png")


def fig2_severity(ev, fig_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax1.hist(ev.crash_z.clip(-12, -3), bins=60, color=P.CATEGORICAL[0], alpha=0.85,
             edgecolor=P.SURFACE, linewidth=0.4)
    ax1.set_xlabel("Crash z-score (clipped at -12)")
    ax1.set_ylabel("Events")
    ax1.set_title("Severity in standard deviations", loc="left", fontsize=10.5, color=P.INK)
    ax1.axvline(ev.crash_z.median(), color=P.CATEGORICAL[1], linewidth=1.6,
                linestyle=(0, (4, 3)))
    ax1.annotate(f"median {ev.crash_z.median():.2f}σ",
                 xy=(ev.crash_z.median(), ax1.get_ylim()[1] * 0.9),
                 xytext=(8, 0), textcoords="offset points",
                 color=P.CATEGORICAL[1], fontsize=9)

    ax2.hist(ev.raw_return.clip(-0.5, 0), bins=60, color=P.CATEGORICAL[0], alpha=0.85,
             edgecolor=P.SURFACE, linewidth=0.4)
    ax2.set_xlabel("Event-day return (clipped at -50%)")
    ax2.set_ylabel("Events")
    ax2.set_title("Severity in percent", loc="left", fontsize=10.5, color=P.INK)
    P.pct_axis(ax2, "x", 0)
    ax2.axvline(ev.raw_return.median(), color=P.CATEGORICAL[1], linewidth=1.6,
                linestyle=(0, (4, 3)))
    ax2.annotate(f"median {ev.raw_return.median() * 100:.1f}%",
                 xy=(ev.raw_return.median(), ax2.get_ylim()[1] * 0.9),
                 xytext=(-8, 0), textcoords="offset points", ha="right",
                 color=P.CATEGORICAL[1], fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.suptitle("Distribution of crash severity", x=0.005, ha="left",
                 fontsize=12.5, color=P.INK, y=0.995)
    fig.text(0.0, -0.04, SRC, fontsize=8, color=P.INK_MUTED, ha="left", va="top")
    return P.save(fig, fig_dir / "fig02_severity_distribution.png")


def fig3_car_all(paths, fig_dir):
    g = paths[paths.group == "all_events"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    P.zero_line(ax)
    P.band(ax, g.tau.values, g.mean_car.values, g.lo.values, g.hi.values,
           P.CATEGORICAL[0], "All events", direct_label=False)
    ax.axvline(20, color=P.INK_MUTED, linewidth=0.9, linestyle=(0, (2, 3)))
    ax.annotate("primary horizon", xy=(20, ax.get_ylim()[1]), xytext=(4, -10),
                textcoords="offset points", fontsize=8.5, color=P.INK_MUTED)
    c20 = g.loc[g.tau == 20, "mean_car"].iloc[0]
    ax.annotate(f"CAR₂₀ = {c20 * 100:+.2f}%", xy=(20, c20), xytext=(-95, -62),
                textcoords="offset points", fontsize=9.5, color=P.CATEGORICAL[0],
                fontweight="medium", ha="center",
                arrowprops=dict(arrowstyle="-", color=P.CATEGORICAL[0],
                                linewidth=0.9, shrinkA=2, shrinkB=6))
    ax.scatter([20], [c20], s=34, color=P.CATEGORICAL[0], zorder=5,
               edgecolor=P.SURFACE, linewidth=1.5)
    P.pct_axis(ax, "y", 2)
    ax.set_xlabel("Trading days after the crash (τ)")
    ax.set_ylabel("Mean cumulative abnormal return")
    ax.set_xlim(0, 60)
    P.finish(fig, ax, "Extreme declines do not mean-revert on average",
             f"Mean CAR vs SPY with 95% bootstrap band (2,000 resamples of "
             f"{int(g.n_events.iloc[0]):,} complete events)", SRC)
    return P.save(fig, fig_dir / "fig03_mean_car_with_ci.png")


def fig4_car_by_type(paths, contrasts, fig_dir):
    """Hero figure: does the kind of crash change what follows?"""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    P.zero_line(ax)
    for t in ("broad_market", "sector", "idiosyncratic"):
        g = paths[paths.group == t]
        if g.empty:
            continue
        P.band(ax, g.tau.values, g.mean_car.values, g.lo.values, g.hi.values,
               P.CRASH_TYPE_COLORS[t],
               f"{P.CRASH_TYPE_LABELS[t]} (n={int(g.n_events.iloc[0]):,})")
    ax.axvline(20, color=P.INK_MUTED, linewidth=0.9, linestyle=(0, (2, 3)))
    P.pct_axis(ax, "y", 2)
    ax.set_xlabel("Trading days after the crash (τ)")
    ax.set_ylabel("Mean cumulative abnormal return")
    ax.set_xlim(0, 60)
    ax.set_xmargin(0.14)
    ax.legend(loc="lower left", ncols=1)
    row = contrasts[(contrasts.group_a == "broad_market") & (contrasts.group_b == "idiosyncratic")]
    note = ""
    if len(row):
        r = row.iloc[0]
        note = (f"Broad-market minus idiosyncratic at τ=20: "
                f"{r['diff'] * 100:+.2f}pp, 95% CI [{r['ci_lo'] * 100:+.2f}, "
                f"{r['ci_hi'] * 100:+.2f}], p={r['p_value']:.2f} — not distinguishable")
    P.finish(fig, ax, "Crash type does not separate 20-day outcomes",
             "Mean CAR vs SPY by operational crash classification, 95% bootstrap bands", SRC)
    if note:
        fig.text(0.0, -0.10, note, fontsize=8.5, color=P.INK_2, ha="left", va="top")
    return P.save(fig, fig_dir / "fig04_car_by_crash_type.png")


def fig5_avol_severity(ev, fig_dir):
    piv = ev.pivot_table(index="avol_quartile", columns="severity_quartile",
                         values="car_20", aggfunc="mean", observed=True)
    piv.index = [f"AVOL {q}" for q in piv.index]
    piv.columns = [f"{q}" for q in piv.columns]
    counts = ev.pivot_table(index="avol_quartile", columns="severity_quartile",
                            values="car_20", aggfunc="size", observed=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    P.heatmap(ax, piv, P.CMAP_DIV, norm=P.diverging_norm(piv.values),
              fmt="{:+.2%}", cbar_label="Mean CAR₂₀")
    ax.set_xlabel("Crash severity quartile (S1 = mildest, S4 = most extreme)")
    ax.set_ylabel("Abnormal volume quartile")
    P.finish(fig, ax, "No monotone interaction between volume and severity",
             f"Mean 20-day CAR by abnormal-volume and severity quartile "
             f"(cell n = {int(counts.values.min()):,}–{int(counts.values.max()):,})", SRC)
    return P.save(fig, fig_dir / "fig05_avol_severity_heatmap.png")


def fig6_recovery_by_vix(ev, fig_dir, seed, n_boot):
    ev = ev.dropna(subset=["vix", "recovered_20d"]).copy()
    ev["vix_q"] = pd.qcut(ev["vix"], 5, labels=["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
    rows = []
    for q, grp in ev.groupby("vix_q", observed=True):
        r = S.bootstrap_mean(grp["recovered_20d"].to_numpy(), n_boot=n_boot, seed=seed)
        rows.append({"q": q, "rate": r["mean"], "lo": r["lo"], "hi": r["hi"],
                     "n": r["n"], "vix_lo": grp.vix.min(), "vix_hi": grp.vix.max()})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(t))
    ax.bar(x, t.rate, width=0.62, color=P.CATEGORICAL[0], zorder=3)
    ax.errorbar(x, t.rate, yerr=[t.rate - t.lo, t.hi - t.rate], fmt="none",
                ecolor=P.INK_2, elinewidth=1.3, capsize=4, zorder=4)
    base = ev["recovered_20d"].mean()
    # Reference lines wear text ink, never a series colour.
    ax.axhline(0.5, color=P.INK_2, linewidth=1.3, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("coin flip (50%)", xy=(-0.45, 0.5), xytext=(0, 6),
                textcoords="offset points", ha="left", fontsize=8.5, color=P.INK_2)
    for xi, r in zip(x, t.itertuples()):
        ax.text(xi, r.rate - 0.035, f"{r.rate * 100:.1f}%", ha="center",
                fontsize=9, color="#ffffff", fontweight="medium", zorder=5)
    ax.set_xticks(x, t.q)
    ax.set_ylim(0, 0.62)
    P.pct_axis(ax, "y", 0)
    ax.set_ylabel("P(CAR₂₀ > 0)")
    ax.set_xlabel("Event-day VIX quintile")
    n_above = int((t.lo > 0.5).sum())
    P.finish(fig, ax, "Recovery is a coin flip in every volatility regime",
             f"Probability of positive 20-day abnormal return, 95% bootstrap CI; "
             f"overall base rate {base * 100:.1f}%; "
             f"{n_above} of {len(t)} quintiles significantly above 50%", SRC)
    return P.save(fig, fig_dir / "fig06_recovery_by_vix_regime.png")


def fig7_outcome_distribution(ev, fig_dir):
    x = ev["car_20"].dropna()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    lo, hi = -0.35, 0.35
    bins = np.linspace(lo, hi, 90)
    xc = x.clip(lo, hi)
    ax.hist(xc[xc < 0], bins=bins, color=P.DIVERGING[6], alpha=0.9,
            edgecolor=P.SURFACE, linewidth=0.3, label="Continued underperformance")
    ax.hist(xc[xc >= 0], bins=bins, color=P.CATEGORICAL[0], alpha=0.9,
            edgecolor=P.SURFACE, linewidth=0.3, label="Recovered")
    ax.axvline(0, color=P.INK_MUTED, linewidth=1.1)
    ax.axvline(x.mean(), color=P.INK, linewidth=1.6, linestyle=(0, (4, 3)))
    ax.annotate(f"mean {x.mean() * 100:+.2f}%\nmedian {x.median() * 100:+.2f}%",
                xy=(x.mean(), ax.get_ylim()[1] * 0.82), xytext=(-10, 0),
                textcoords="offset points", ha="right", fontsize=9, color=P.INK)
    share = (x > 0).mean()
    ax.annotate(f"{share * 100:.1f}% of events end above the benchmark",
                xy=(0.16, ax.get_ylim()[1] * 0.55), fontsize=9, color=P.CATEGORICAL[0])
    P.pct_axis(ax, "x", 0)
    ax.set_xlabel("20-day cumulative abnormal return (clipped at ±35%)")
    ax.set_ylabel("Events")
    ax.legend(loc="upper left")
    P.finish(fig, ax, "The 20-day outcome distribution is wide and slightly left-shifted",
             f"{len(x):,} crash events; dispersion (σ = {x.std() * 100:.1f}%) "
             f"dwarfs the mean effect", SRC)
    return P.save(fig, fig_dir / "fig07_outcome_distribution.png")


def fig8_calibration(cal, comp, fig_dir):
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.plot([0, 1], [0, 1], color=P.INK_MUTED, linewidth=1.1, linestyle=(0, (4, 3)),
            zorder=1)
    ax.annotate("perfect calibration", xy=(0.72, 0.72), xytext=(0, -14),
                textcoords="offset points", fontsize=8.5, color=P.INK_MUTED, rotation=39)
    for i, (name, grp) in enumerate(cal.groupby("model")):
        auc = comp.loc[comp.model == name, "roc_auc"]
        lbl = f"{name} (AUC {auc.iloc[0]:.3f}, ECE {grp.ece.iloc[0]:.3f})"
        ax.plot(grp.mean_pred, grp.observed_freq, marker="o", markersize=6,
                color=P.CATEGORICAL[i], label=lbl, linewidth=2.0,
                markeredgecolor=P.SURFACE, markeredgewidth=1.4, zorder=3 + i)
    base = comp.loc[comp.model == "base_rate", "base_rate"].iloc[0]
    ax.axhline(base, color=P.INK_MUTED, linewidth=0.9, linestyle=(0, (1, 3)))
    ax.annotate(f"test base rate {base * 100:.1f}%", xy=(0.02, base), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=P.INK_2)
    ax.set_xlim(0.25, 0.75); ax.set_ylim(0.25, 0.75)
    P.pct_axis(ax, "x", 0); P.pct_axis(ax, "y", 0)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed recovery frequency")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    eces = cal.groupby("model").ece.first()
    P.finish(fig, ax, "Forecasts stay near the base rate and do not discriminate",
             f"Reliability curves on the held-out test block (decile bins); observed "
             f"frequency is flat in the prediction. ECE {eces.min():.3f}–{eces.max():.3f}", SRC)
    return P.save(fig, fig_dir / "fig08_calibration.png")


def fig9_shap(fig_dir, top_n=14):
    sv = np.load(PATHS.data_processed / "shap_values.npy")
    X = pd.read_parquet(PATHS.data_processed / "shap_X.parquet")
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * top_n + 2.2))
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        v = X.iloc[:, j].to_numpy(float)
        finite = np.isfinite(v)
        r = np.full(v.shape, 0.5)
        if finite.sum() > 1:
            lo, hi = np.nanpercentile(v[finite], [5, 95])
            r[finite] = np.clip((v[finite] - lo) / max(hi - lo, 1e-12), 0, 1)
        ax.scatter(sv[:, j], row + rng.uniform(-0.16, 0.16, len(sv)),
                   c=r, cmap=P.CMAP_SEQ, s=5, alpha=0.55, linewidths=0)
    ax.set_yticks(range(len(order)), [X.columns[j] for j in order])
    ax.axvline(0, color=P.INK_MUTED, linewidth=1.0)
    ax.set_xlabel("SHAP value (log-odds contribution to P(recovery))")
    ax.grid(axis="y", visible=False)
    sm = plt.cm.ScalarMappable(cmap=P.CMAP_SEQ)
    cb = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.015, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("Feature value", fontsize=8.5, color=P.INK_2)
    cb.outline.set_visible(False)
    P.finish(fig, ax, "SHAP attributions are small and show no coherent direction",
             f"LightGBM, {len(sv):,} held-out test events; note the narrow "
             f"log-odds scale", SRC)
    return P.save(fig, fig_dir / "fig09_shap_summary.png")


def fig11_interpretability(fig_dir):
    """Logistic coefficients beside permutation importance: do the two agree?"""
    coefs = pd.read_csv(PATHS.tables / "logistic_model_coefficients.csv").head(14)
    perm = pd.read_csv(PATHS.tables / "permutation_importance.csv").head(14)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    c = coefs.iloc[::-1]
    colors = [P.DIVERGING[6] if v < 0 else P.CATEGORICAL[0] for v in c.coef]
    axes[0].barh(range(len(c)), c.coef, color=colors, height=0.68, zorder=3)
    axes[0].axvline(0, color=P.INK_MUTED, linewidth=1.0)
    axes[0].set_yticks(range(len(c)), c.feature)
    axes[0].set_xlabel("Logistic coefficient (standardised features, log-odds)")
    axes[0].set_title("Logistic regression coefficients", loc="left",
                      fontsize=10.5, color=P.INK)
    axes[0].grid(axis="y", visible=False)

    q = perm.iloc[::-1]
    axes[1].barh(range(len(q)), q.importance_mean, height=0.68,
                 color=P.CATEGORICAL[0], zorder=3,
                 xerr=q.importance_std, error_kw=dict(ecolor=P.INK_2, elinewidth=1.0))
    axes[1].axvline(0, color=P.INK_MUTED, linewidth=1.0)
    axes[1].set_yticks(range(len(q)), q.feature)
    axes[1].set_xlabel("Permutation importance (drop in test ROC-AUC)")
    axes[1].set_title("Permutation importance (LightGBM)", loc="left",
                      fontsize=10.5, color=P.INK)
    axes[1].grid(axis="y", visible=False)

    # Lay the axes out first, then hang the title block above the finished figure.
    fig.tight_layout()
    fig.suptitle("Statistical and machine-learning attributions do not agree",
                 x=0.005, ha="left", fontsize=12.5, color=P.INK, y=1.10)
    fig.text(0.005, 1.025, "Neither ranking is stable: every permutation effect is "
             "within noise of zero (note the axis scale)",
             fontsize=9.5, color=P.INK_2, ha="left", va="bottom")
    fig.text(0.0, -0.05, SRC, fontsize=8, color=P.INK_MUTED, ha="left", va="top")
    return P.save(fig, fig_dir / "fig11_interpretability_comparison.png")


def fig12_shap_dependence(fig_dir, top_n=4):
    sv = np.load(PATHS.data_processed / "shap_values.npy")
    X = pd.read_parquet(PATHS.data_processed / "shap_X.parquet")
    order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_n]
    fig, axes = plt.subplots(1, top_n, figsize=(3.4 * top_n, 3.6), sharey=True)
    for ax, j in zip(np.atleast_1d(axes), order):
        name = X.columns[j]
        v = X.iloc[:, j].to_numpy(float)
        lo, hi = np.nanpercentile(v, [1, 99])
        ax.axhline(0, color=P.INK_MUTED, linewidth=0.9)
        ax.scatter(np.clip(v, lo, hi), sv[:, j], s=6, alpha=0.35,
                   color=P.CATEGORICAL[0], linewidths=0)
        # LOESS-free trend: binned median, which is robust and honest here.
        bins = np.quantile(np.clip(v, lo, hi), np.linspace(0, 1, 13))
        bins = np.unique(bins)
        if len(bins) > 3:
            idx = np.clip(np.digitize(np.clip(v, lo, hi), bins[1:-1]), 0, len(bins) - 2)
            med = pd.DataFrame({"b": idx, "y": sv[:, j]}).groupby("b")["y"].median()
            ctr = (bins[:-1] + bins[1:]) / 2
            ax.plot(ctr[med.index.to_numpy()], med.values, color=P.CATEGORICAL[1],
                    linewidth=2.0, zorder=4)
        ax.set_xlabel(name)
        ax.set_title(name, loc="left", fontsize=10, color=P.INK)
    np.atleast_1d(axes)[0].set_ylabel("SHAP value (log-odds)")
    fig.tight_layout()
    fig.suptitle("SHAP dependence for the four highest-attribution features",
                 x=0.005, ha="left", fontsize=12.5, color=P.INK, y=1.17)
    fig.text(0.005, 1.055, "Orange line is the binned median. No feature shows a "
             "stable monotone relationship with recovery odds.",
             fontsize=9.5, color=P.INK_2, ha="left", va="bottom")
    fig.text(0.0, -0.09, SRC, fontsize=8, color=P.INK_MUTED, ha="left", va="top")
    return P.save(fig, fig_dir / "fig12_shap_dependence.png")


def fig10_robustness(grid, cfg, fig_dir):
    h = int(cfg.outcomes.primary_horizon)
    a = (grid[(grid.horizon == h) & (grid.vol_window == cfg.events.vol_window)]
         .drop_duplicates(["z_threshold", "cooldown"])
         .pivot(index="z_threshold", columns="cooldown", values="mean_car"))
    b = (grid[grid.vol_window == cfg.events.vol_window]
         .drop_duplicates(["z_threshold", "horizon"])
         .pivot(index="z_threshold", columns="horizon", values="mean_car"))
    a.index = [f"{i:.1f}σ" for i in a.index]
    b.index = [f"{i:.1f}σ" for i in b.index]
    allv = np.concatenate([a.values.ravel(), b.values.ravel()])
    norm = P.diverging_norm(allv)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6))
    P.heatmap(axes[0], a, P.CMAP_DIV, norm=norm, fmt="{:+.2%}", cbar=False)
    axes[0].set_xlabel("Cooldown (trading days)")
    axes[0].set_ylabel("Crash threshold")
    axes[0].set_title("Threshold × cooldown (h = 20)", loc="left", fontsize=10.5, color=P.INK)

    im = P.heatmap(axes[1], b, P.CMAP_DIV, norm=norm, fmt="{:+.2%}", cbar=False)
    axes[1].set_xlabel("Outcome horizon (trading days)")
    axes[1].set_ylabel("Crash threshold")
    axes[1].set_title("Threshold × horizon", loc="left", fontsize=10.5, color=P.INK)

    # One scale, one colorbar - both panels share `norm`.
    cb = fig.colorbar(im, ax=axes, fraction=0.026, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=P.GRID)
    cb.ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 100:+.2f}%"))
    cb.set_label("Mean CAR", fontsize=8.5, color=P.INK_2)

    n_neg = int((grid[grid.horizon == h].drop_duplicates(
        ["z_threshold", "vol_window", "cooldown"]).mean_car < 0).sum())
    n_tot = int(len(grid[grid.horizon == h].drop_duplicates(
        ["z_threshold", "vol_window", "cooldown"])))
    fig.suptitle("The sign of the effect survives every specification", x=0.005,
                 ha="left", fontsize=12.5, color=P.INK, y=1.13)
    fig.text(0.005, 1.055, f"Mean CAR across {len(grid):,} persisted specifications; "
             f"negative in {n_neg}/{n_tot} threshold×window×cooldown cells at h=20",
             fontsize=9.5, color=P.INK_2, ha="left", va="bottom",
             transform=fig.transFigure)
    fig.text(0.0, -0.06, SRC, fontsize=8, color=P.INK_MUTED, ha="left", va="top")
    return P.save(fig, fig_dir / "fig10_robustness_heatmap.png")


def main() -> None:
    cfg = load_config()
    PATHS.ensure()
    P.use_style()
    fig_dir = PATHS.figures

    ev = D.load_processed("events.parquet")
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    evc = ev[ev.complete_20]
    paths = pd.read_csv(PATHS.tables / "event_study_paths.csv")
    contrasts = pd.read_csv(PATHS.tables / "group_contrasts.csv")
    comp = pd.read_csv(PATHS.tables / "model_comparison.csv")
    cal = pd.read_csv(PATHS.tables / "calibration.csv")
    grid = pd.read_csv(PATHS.tables / "robustness_grid.csv")

    made = [
        fig1_frequency(ev, fig_dir),
        fig2_severity(ev, fig_dir),
        fig3_car_all(paths, fig_dir),
        fig4_car_by_type(paths, contrasts, fig_dir),
        fig5_avol_severity(evc, fig_dir),
        fig6_recovery_by_vix(evc, fig_dir, int(cfg.study.seed), int(cfg.bootstrap.n_boot)),
        fig7_outcome_distribution(evc, fig_dir),
        fig8_calibration(cal, comp, fig_dir),
        fig9_shap(fig_dir),
        fig10_robustness(grid, cfg, fig_dir),
        fig11_interpretability(fig_dir),
        fig12_shap_dependence(fig_dir),
    ]
    for m in made:
        log.info("wrote %s", m)
    (PATHS.metrics / "figures.json").write_text(json.dumps(
        {"figures": [str(m.name) for m in made],
         "config_fingerprint": cfg.fingerprint}, indent=2))


if __name__ == "__main__":
    main()
