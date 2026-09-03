"""Crash-event detection, episode grouping and forward outcome measurement.

Every quantity used to *define* an event is computed from observations
strictly preceding the event date. Every quantity measured *after* the event
is prefixed or documented as an outcome and is excluded from the predictor
matrix (see ``docs/data_leakage_audit.md`` and ``tests/test_leakage.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that describe what happened *after* the event and must never be
# offered to a model as a predictor.
OUTCOME_PREFIXES = ("fwd_", "car_", "mfe_", "mae_", "recovered_", "regained_", "days_to_")
DESCRIPTIVE_FORWARD = ("episode_length", "episode_min_return", "episode_end_date")


def to_wide(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    """Long tidy prices -> ``date x ticker`` matrix."""
    return prices.pivot(index="date", columns="ticker", values=field).sort_index()


def daily_returns(close: pd.DataFrame) -> pd.DataFrame:
    r"""Simple daily return :math:`r_{i,t} = P_{i,t}/P_{i,t-1} - 1`."""
    return close / close.shift(1) - 1.0


def rolling_moments(ret: pd.DataFrame, window: int, min_obs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Rolling mean and volatility over the ``window`` days *strictly before* ``t``.

    The trailing ``.shift(1)`` is what makes the estimate causal: the window
    ending at :math:`t-1` never contains :math:`r_{i,t}` itself.
    """
    mu = ret.rolling(window, min_periods=min_obs).mean().shift(1)
    sigma = ret.rolling(window, min_periods=min_obs).std(ddof=1).shift(1)
    return mu, sigma


def crash_zscore(ret: pd.DataFrame, mu: pd.DataFrame, sigma: pd.DataFrame) -> pd.DataFrame:
    r""":math:`z_{i,t} = (r_{i,t} - \mu^{w}_{i,t-1}) / \sigma^{w}_{i,t-1}`."""
    sig = sigma.where(sigma > 0)
    return (ret - mu) / sig


def candidate_crashes(
    z: pd.DataFrame,
    threshold: float,
    eligible: pd.DataFrame,
    study_start: pd.Timestamp,
    study_end: pd.Timestamp,
) -> pd.DataFrame:
    """All ``(ticker, date)`` pairs breaching the threshold while index-eligible."""
    hit = (z <= threshold) & z.notna()
    hit = hit & eligible.reindex_like(hit).fillna(False)
    in_window = (hit.index >= study_start) & (hit.index <= study_end)
    hit = hit.loc[in_window]
    zz = z.loc[in_window]
    idx = np.argwhere(hit.to_numpy())
    if idx.size == 0:
        return pd.DataFrame(columns=["ticker", "event_date", "crash_z"])
    dates = hit.index.to_numpy()[idx[:, 0]]
    tickers = np.asarray(hit.columns)[idx[:, 1]]
    zvals = zz.to_numpy()[idx[:, 0], idx[:, 1]]
    return pd.DataFrame({"ticker": tickers, "event_date": dates, "crash_z": zvals}).sort_values(
        ["ticker", "event_date"]
    ).reset_index(drop=True)


def apply_cooldown(cands: pd.DataFrame, cooldown: int, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Greedy ticker-specific cooldown; consecutive breaches collapse to one episode.

    The first breach in an episode is kept. Any later breach within
    ``cooldown`` *trading days* of the kept event is absorbed into that
    episode rather than counted as an independent observation.
    """
    if cands.empty:
        return cands.assign(episode_length=[], episode_min_return=[])
    pos = pd.Series(np.arange(len(calendar)), index=calendar)
    cands = cands.copy()
    cands["pos"] = cands["event_date"].map(pos)
    cands = cands.dropna(subset=["pos"]).sort_values(["ticker", "pos"])

    keep_idx, episode_of = [], {}
    for tkr, grp in cands.groupby("ticker", sort=False):
        anchor_pos, anchor_row = None, None
        for row in grp.itertuples():
            if anchor_pos is None or (row.pos - anchor_pos) >= cooldown:
                anchor_pos, anchor_row = row.pos, row.Index
                keep_idx.append(row.Index)
                episode_of[row.Index] = [row.Index]
            else:
                episode_of[anchor_row].append(row.Index)
    out = cands.loc[keep_idx].copy()
    out["episode_length"] = [len(episode_of[i]) for i in keep_idx]
    out["episode_breach_idx"] = [episode_of[i] for i in keep_idx]
    return out.sort_values(["event_date", "ticker"]).reset_index(drop=True)


def attach_episode_stats(events: pd.DataFrame, cands: pd.DataFrame, ret: pd.DataFrame,
                         cooldown: int, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Minimum daily return observed inside each episode window (descriptive)."""
    ret_np = ret.to_numpy()
    col = {c: i for i, c in enumerate(ret.columns)}
    mins, ends = [], []
    n = len(calendar)
    for row in events.itertuples(index=False):
        j = col.get(row.ticker)
        p0 = int(row.pos)
        p1 = min(p0 + cooldown, n)
        seg = ret_np[p0:p1, j] if j is not None else np.array([np.nan])
        mins.append(np.nanmin(seg) if np.isfinite(seg).any() else np.nan)
        ends.append(calendar[min(p1 - 1, n - 1)])
    out = events.copy()
    out["episode_min_return"] = mins
    out["episode_end_date"] = ends
    return out


def forward_outcomes(
    events: pd.DataFrame,
    close: pd.DataFrame,
    bench_close: pd.Series,
    horizons: list[int],
    primary_h: int,
) -> pd.DataFrame:
    r"""Forward returns, benchmark-adjusted CARs and path statistics.

    :math:`CAR_{i,t,h} = R_{i,t:t+h} - R_{SPY,t:t+h}`, where both legs are
    measured from the event close, so the crash-day move itself is excluded
    from the outcome.
    """
    cal = close.index
    n = len(cal)
    col = {c: i for i, c in enumerate(close.columns)}
    cl = close.to_numpy()
    bench = bench_close.reindex(cal).to_numpy()

    pos = events["pos"].to_numpy().astype(int)
    jj = np.array([col.get(t, -1) for t in events["ticker"]])
    valid_j = jj >= 0

    out = {}
    p0 = cl[pos, np.where(valid_j, jj, 0)]
    b0 = bench[pos]
    p0 = np.where(valid_j, p0, np.nan)

    for h in horizons:
        tgt = pos + h
        ok = (tgt < n) & valid_j
        ph = np.full(len(pos), np.nan)
        bh = np.full(len(pos), np.nan)
        ph[ok] = cl[tgt[ok], jj[ok]]
        bh[ok] = bench[tgt[ok]]
        r_stock = ph / p0 - 1.0
        r_bench = bh / b0 - 1.0
        out[f"fwd_ret_{h}"] = r_stock
        out[f"bench_ret_{h}"] = r_bench
        out[f"car_{h}"] = r_stock - r_bench

    # Path statistics over the primary horizon, measured from the event close.
    H = primary_h
    paths = np.full((len(pos), H), np.nan)
    for k in range(1, H + 1):
        tgt = pos + k
        ok = (tgt < n) & valid_j
        pk = np.full(len(pos), np.nan)
        pk[ok] = cl[tgt[ok], jj[ok]]
        paths[:, k - 1] = pk / p0 - 1.0
    with np.errstate(invalid="ignore"):
        out[f"mfe_{H}"] = np.nanmax(paths, axis=1)
        out[f"mae_{H}"] = np.nanmin(paths, axis=1)

    # Secondary outcome: did the close regain its *pre-crash* level (t-1)?
    pre_pos = np.clip(pos - 1, 0, n - 1)
    pre = np.full(len(pos), np.nan)
    pre[valid_j] = cl[pre_pos[valid_j], jj[valid_j]]
    prices_path = paths * 0.0
    for k in range(1, H + 1):
        tgt = np.clip(pos + k, 0, n - 1)
        pk = np.full(len(pos), np.nan)
        m = valid_j & ((pos + k) < n)
        pk[m] = cl[tgt[m], jj[m]]
        prices_path[:, k - 1] = pk
    reached = prices_path >= pre[:, None]
    any_reached = np.nansum(np.where(np.isnan(prices_path), 0, reached), axis=1) > 0
    first = np.where(reached & ~np.isnan(prices_path), np.arange(1, H + 1), np.nan)
    with np.errstate(invalid="ignore"):
        days_to = np.nanmin(np.where(np.isnan(first), np.inf, first), axis=1)
    days_to = np.where(np.isfinite(days_to), days_to, np.nan)

    out["precrash_close"] = pre
    out[f"regained_precrash_{H}"] = np.where(np.isnan(out[f"car_{H}"]), np.nan, any_reached.astype(float))
    out["days_to_recovery"] = days_to
    out[f"recovered_{H}d"] = np.where(np.isnan(out[f"car_{H}"]), np.nan, (out[f"car_{H}"] > 0).astype(float))

    res = events.copy()
    for k, v in out.items():
        res[k] = v
    return res


def build_events(
    close: pd.DataFrame,
    ret: pd.DataFrame,
    eligible: pd.DataFrame,
    bench_close: pd.Series,
    *,
    z_threshold: float,
    vol_window: int,
    min_prior_obs: int,
    cooldown: int,
    horizons: list[int],
    primary_h: int,
    study_start: pd.Timestamp,
    study_end: pd.Timestamp,
) -> pd.DataFrame:
    """Full event table for one specification."""
    mu, sigma = rolling_moments(ret, vol_window, min_prior_obs)
    z = crash_zscore(ret, mu, sigma)
    cands = candidate_crashes(z, z_threshold, eligible, study_start, study_end)
    if cands.empty:
        return pd.DataFrame()
    ev = apply_cooldown(cands, cooldown, close.index)
    ev = attach_episode_stats(ev, cands, ret, cooldown, close.index)

    # Raw event-day return, attached from the same causal matrices.
    col = {c: i for i, c in enumerate(ret.columns)}
    rnp = ret.to_numpy()
    jj = np.array([col.get(t, -1) for t in ev["ticker"]])
    pp = ev["pos"].to_numpy().astype(int)
    ev["raw_return"] = np.where(jj >= 0, rnp[pp, np.where(jj >= 0, jj, 0)], np.nan)

    ev = forward_outcomes(ev, close, bench_close, horizons, primary_h)
    ev = ev.drop(columns=["episode_breach_idx"], errors="ignore")
    ev.insert(0, "event_id", [f"{t}_{pd.Timestamp(d).date()}" for t, d in
                              zip(ev["ticker"], ev["event_date"])])
    return ev.reset_index(drop=True)


def car_path(events: pd.DataFrame, close: pd.DataFrame, bench_close: pd.Series,
             max_tau: int = 60) -> np.ndarray:
    r"""Benchmark-adjusted return path in event time, :math:`\tau = 0 \dots T`.

    Row ``i``, column ``tau`` holds
    :math:`(P_{i,t+\tau}/P_{i,t} - 1) - (B_{t+\tau}/B_t - 1)`, so column 0 is
    identically zero and the crash-day move itself is excluded.
    """
    cal = close.index
    n = len(cal)
    col = {c: i for i, c in enumerate(close.columns)}
    cl = close.to_numpy()
    bench = bench_close.reindex(cal).to_numpy()

    pos = events["pos"].to_numpy().astype(int)
    jj = np.array([col.get(t, -1) for t in events["ticker"]])
    valid = jj >= 0
    p0 = np.where(valid, cl[pos, np.where(valid, jj, 0)], np.nan)
    b0 = bench[pos]

    out = np.full((len(pos), max_tau + 1), np.nan)
    for tau in range(max_tau + 1):
        tgt = pos + tau
        ok = valid & (tgt < n)
        pk = np.full(len(pos), np.nan)
        bk = np.full(len(pos), np.nan)
        pk[ok] = cl[tgt[ok], jj[ok]]
        bk[ok] = bench[tgt[ok]]
        out[:, tau] = (pk / p0 - 1.0) - (bk / b0 - 1.0)
    return out
