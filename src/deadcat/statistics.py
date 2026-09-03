"""Inference: bootstrap event-study bands, robust regression, FDR control.

Two standard-error families are reported throughout. HC3 is the
heteroskedasticity-robust default requested by the research plan. Because
crash events cluster heavily in calendar time - hundreds of names can breach
the threshold on the same market-wide day - residuals are strongly correlated
within a date, so date-clustered standard errors are reported alongside as the
more conservative inferential benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


# --------------------------------------------------------------- bootstrap ---
def bootstrap_mean(x, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Percentile bootstrap for the mean of a single vector."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0, "p_two_sided": np.nan}
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    alpha = (1 - ci) / 2
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"mean": float(x.mean()), "lo": float(np.quantile(boots, alpha)),
            "hi": float(np.quantile(boots, 1 - alpha)), "n": int(x.size),
            "p_two_sided": float(min(p, 1.0))}


def bootstrap_mean_path(paths: np.ndarray, n_boot: int = 2000, ci: float = 0.95,
                        seed: int = 0) -> pd.DataFrame:
    """Bootstrap the whole event-time path by resampling complete events.

    Resampling events (rows) rather than individual observations preserves the
    within-event autocorrelation of a cumulative return path.
    """
    paths = np.asarray(paths, dtype=float)
    P = paths[np.isfinite(paths).all(axis=1)]
    n, T = P.shape
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, T))
    for b in range(n_boot):
        boots[b] = P[rng.integers(0, n, n)].mean(axis=0)
    alpha = (1 - ci) / 2
    return pd.DataFrame({
        "tau": np.arange(T),
        "mean_car": P.mean(axis=0),
        "lo": np.quantile(boots, alpha, axis=0),
        "hi": np.quantile(boots, 1 - alpha, axis=0),
        "n_events": n,
    })


def bootstrap_diff(a, b, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> dict:
    """Bootstrap the difference in means between two independent groups."""
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {"diff": np.nan, "lo": np.nan, "hi": np.nan, "p_two_sided": np.nan,
                "n_a": int(a.size), "n_b": int(b.size)}
    rng = np.random.default_rng(seed)
    da = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    db = b[rng.integers(0, b.size, (n_boot, b.size))].mean(axis=1)
    d = da - db
    alpha = (1 - ci) / 2
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"diff": float(a.mean() - b.mean()), "lo": float(np.quantile(d, alpha)),
            "hi": float(np.quantile(d, 1 - alpha)), "n_a": int(a.size), "n_b": int(b.size),
            "p_two_sided": float(min(p, 1.0))}


# -------------------------------------------------------------- regression ---
def _clean(df: pd.DataFrame, y: str, xs: list[str]):
    d = df[[y] + xs].replace([np.inf, -np.inf], np.nan).dropna()
    return d[y].astype(float), d[xs].astype(float), d.index


def ols_hc3(df: pd.DataFrame, y: str, xs: list[str], cluster: str | None = None) -> dict:
    """OLS with HC3 standard errors; optionally date-clustered SEs alongside."""
    Y, Xr, idx = _clean(df, y, xs)
    X = sm.add_constant(Xr, has_constant="add")
    fit = sm.OLS(Y, X).fit(cov_type="HC3")
    sdx, sdy = Xr.std(ddof=0), Y.std(ddof=0)
    ci = fit.conf_int(alpha=0.05)
    tbl = pd.DataFrame({
        "term": fit.params.index,
        "coef": fit.params.values,
        "std_coef": [np.nan if t == "const" else fit.params[t] * sdx[t] / sdy
                     for t in fit.params.index],
        "se_hc3": fit.bse.values,
        "ci_lo": ci[0].values,
        "ci_hi": ci[1].values,
        "t": fit.tvalues.values,
        "p_value": fit.pvalues.values,
    })
    if cluster and cluster in df.columns:
        g = pd.factorize(df.loc[idx, cluster])[0]
        cl = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": g})
        cci = cl.conf_int(alpha=0.05)
        tbl["se_cluster_date"] = cl.bse.values
        tbl["p_value_cluster"] = cl.pvalues.values
        tbl["ci_lo_cluster"] = cci[0].values
        tbl["ci_hi_cluster"] = cci[1].values
    return {"table": tbl, "n": int(fit.nobs), "r2": float(fit.rsquared),
            "adj_r2": float(fit.rsquared_adj), "f_pvalue": float(fit.f_pvalue), "model": fit}


def logit_odds(df: pd.DataFrame, y: str, xs: list[str], cluster: str | None = None) -> dict:
    """Logistic regression; odds ratios are per one-standard-deviation move."""
    Y, Xr, idx = _clean(df, y, xs)
    Xs = (Xr - Xr.mean()) / Xr.std(ddof=0)
    X = sm.add_constant(Xs, has_constant="add")
    if cluster and cluster in df.columns:
        g = pd.factorize(df.loc[idx, cluster])[0]
        fit = sm.Logit(Y, X).fit(disp=False, cov_type="cluster", cov_kwds={"groups": g})
    else:
        fit = sm.Logit(Y, X).fit(disp=False)
    ci = fit.conf_int(alpha=0.05)
    tbl = pd.DataFrame({
        "term": fit.params.index,
        "coef_per_sd": fit.params.values,
        "odds_ratio_per_sd": np.exp(fit.params.values),
        "or_ci_lo": np.exp(ci[0].values),
        "or_ci_hi": np.exp(ci[1].values),
        "se": fit.bse.values,
        "z": fit.tvalues.values,
        "p_value": fit.pvalues.values,
    })
    return {"table": tbl, "n": int(fit.nobs), "pseudo_r2": float(fit.prsquared),
            "llf": float(fit.llf), "model": fit}


# --------------------------------------------------------- multiple testing ---
def benjamini_hochberg(pvals, alpha: float = 0.05) -> pd.DataFrame:
    """Benjamini-Hochberg step-up FDR control."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.clip(np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1], 0, 1)
    out = np.empty(n); out[order] = q
    return pd.DataFrame({"p_value": p, "q_value": out, "reject_fdr": out <= alpha})


def drop_aliased(df: pd.DataFrame, xs: list[str], tol: float = 1e-10) -> tuple[list[str], list[str]]:
    """Remove columns that are exact linear combinations of earlier ones.

    Several features are identities by construction - ``abs_decline`` is
    ``-raw_return`` because every event has a negative return, and
    ``excess_vs_sector`` is ``raw_return - sector_ret_event``. Left in, they
    make the design singular. A rank-revealing QR keeps the first column of
    each aliased set and reports what was dropped rather than silently
    reshuffling the specification.
    """
    d = df[xs].replace([np.inf, -np.inf], np.nan).dropna()
    kept: list[str] = []
    dropped: list[str] = []
    for c in xs:
        trial = kept + [c]
        M = np.column_stack([np.ones(len(d))] + [d[k].to_numpy(float) for k in trial])
        if np.linalg.matrix_rank(M, tol=tol) == len(trial) + 1:
            kept.append(c)
        else:
            dropped.append(c)
    return kept, dropped


def clustered_mean_se(x, clusters) -> dict:
    """Mean with a cluster-robust standard error.

    Used across the robustness grid where a 2,000-draw bootstrap per cell would
    be prohibitive. Clustering on the event date is the binding correction:
    crash events are heavily concentrated on market-wide days, so treating them
    as independent overstates precision badly.
    """
    d = pd.DataFrame({"x": np.asarray(x, float), "g": np.asarray(clusters)}).dropna()
    n = len(d)
    if n < 2:
        return {"mean": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "t": np.nan, "p_value": np.nan, "n": n, "n_clusters": 0}
    m = d["x"].mean()
    dev = d["x"] - m
    g_sums = dev.groupby(d["g"]).sum()
    se = float(np.sqrt((g_sums ** 2).sum()) / n)
    G = int(g_sums.size)
    # Small-cluster correction, as in standard cluster-robust practice.
    se *= np.sqrt(G / max(G - 1, 1))
    t = m / se if se > 0 else np.nan
    from scipy import stats as sps
    p = float(2 * sps.t.sf(abs(t), df=max(G - 1, 1))) if np.isfinite(t) else np.nan
    crit = float(sps.t.ppf(0.975, df=max(G - 1, 1)))
    return {"mean": float(m), "se": se, "ci_lo": float(m - crit * se),
            "ci_hi": float(m + crit * se), "t": float(t), "p_value": p,
            "n": n, "n_clusters": G}
