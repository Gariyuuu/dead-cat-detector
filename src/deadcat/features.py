"""Event-time feature construction.

Contract: every column produced here is computable from information available
at the *close of the event day*. Trailing windows end at ``t-1`` except for
same-day quantities (the crash return itself, event-day volume, event-day
market and sector returns, event-day VIX), which are observable at the close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    # crash
    "crash_z", "raw_return", "abs_decline", "hl_range",
    # volume
    "volume", "log_volume", "med_volume_60", "avol",
    # momentum (strictly pre-crash)
    "mom_5", "mom_20", "mom_60", "mom_252",
    # volatility (strictly pre-crash)
    "rv_20", "rv_60", "rv_ratio",
    # market
    "spy_ret_event", "spy_mom_20", "vix", "vix_chg",
    # sector
    "sector_ret_event", "excess_vs_sector",
]


def _gather(mat: pd.DataFrame, pos: np.ndarray, jj: np.ndarray, offset: int = 0) -> np.ndarray:
    """Value of ``mat`` at row ``pos+offset`` for each event's column."""
    arr = mat.to_numpy()
    n = len(mat)
    tgt = pos + offset
    ok = (tgt >= 0) & (tgt < n) & (jj >= 0)
    out = np.full(len(pos), np.nan)
    out[ok] = arr[tgt[ok], jj[ok]]
    return out


def _series_at(s: pd.Series, index: pd.DatetimeIndex, pos: np.ndarray, offset: int = 0) -> np.ndarray:
    arr = s.reindex(index).to_numpy()
    tgt = pos + offset
    ok = (tgt >= 0) & (tgt < len(arr))
    out = np.full(len(pos), np.nan)
    out[ok] = arr[tgt[ok]]
    return out


def build_sector_series(bench_close: pd.DataFrame, sector_etfs: dict, fallbacks: dict) -> pd.DataFrame:
    """Sector ETF close series, back-filled with a long-history proxy.

    XLRE (2015) and XLC (2018) post-date the study window, so IYR and IYZ
    stand in before inception. The substitution is recorded, not silent.
    """
    out = {}
    for sector, etf in sector_etfs.items():
        if etf not in bench_close.columns:
            continue
        s = bench_close[etf].copy()
        fb = fallbacks.get(etf)
        if fb and fb in bench_close.columns:
            s = s.combine_first(bench_close[fb])
        out[sector] = s
    return pd.DataFrame(out, index=bench_close.index)


def expanding_low_pctile(ret: pd.Series, pctile: float, min_periods: int = 252) -> pd.Series:
    """Historical ``pctile``-th percentile using strictly prior observations."""
    q = ret.expanding(min_periods=min_periods).quantile(pctile / 100.0)
    return q.shift(1)


def build_features(
    events: pd.DataFrame,
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    ret: pd.DataFrame,
    bench_close: pd.DataFrame,
    sector_close: pd.DataFrame,
    ticker_sector: pd.Series,
    *,
    market: str = "SPY",
    vix: str = "^VIX",
    vol_window: int = 60,
    benchmark_pctile: float = 5.0,
) -> pd.DataFrame:
    cal = close.index
    col = {c: i for i, c in enumerate(close.columns)}
    pos = events["pos"].to_numpy().astype(int)
    jj = np.array([col.get(t, -1) for t in events["ticker"]])

    f = pd.DataFrame(index=events.index)
    f["abs_decline"] = np.abs(events["raw_return"].to_numpy())

    prev_close = _gather(close, pos, jj, -1)
    hi = _gather(high, pos, jj, 0)
    lo = _gather(low, pos, jj, 0)
    f["hl_range"] = (hi - lo) / prev_close

    # --- volume -------------------------------------------------------------
    vol_pos = volume.where(volume > 0)
    v_t = _gather(vol_pos, pos, jj, 0)
    med60 = vol_pos.rolling(vol_window, min_periods=max(20, vol_window // 3)).median().shift(1)
    v_med = _gather(med60, pos, jj, 0)
    f["volume"] = v_t
    f["log_volume"] = np.log(v_t)
    f["med_volume_60"] = v_med
    f["avol"] = np.log(v_t) - np.log(v_med)

    # --- momentum: windows end at t-1, so the crash day is excluded ---------
    for w in (5, 20, 60, 252):
        c_prev = _gather(close, pos, jj, -1)
        c_back = _gather(close, pos, jj, -1 - w)
        f[f"mom_{w}"] = c_prev / c_back - 1.0

    # --- realised volatility, annualised, strictly pre-crash ----------------
    for w in (20, 60):
        rv = ret.rolling(w, min_periods=max(10, w // 2)).std(ddof=1).shift(1) * np.sqrt(252)
        f[f"rv_{w}"] = _gather(rv, pos, jj, 0)
    f["rv_ratio"] = f["rv_20"] / f["rv_60"]

    # --- market -------------------------------------------------------------
    spy = bench_close[market]
    spy_ret = spy / spy.shift(1) - 1.0
    f["spy_ret_event"] = _series_at(spy_ret, cal, pos, 0)
    f["spy_mom_20"] = _series_at(spy.shift(1) / spy.shift(21) - 1.0, cal, pos, 0)
    vixs = bench_close[vix]
    f["vix"] = _series_at(vixs, cal, pos, 0)
    f["vix_chg"] = _series_at(vixs - vixs.shift(1), cal, pos, 0)

    # --- sector -------------------------------------------------------------
    sect = events["ticker"].map(ticker_sector)
    sec_ret = sector_close / sector_close.shift(1) - 1.0
    sec_vals = np.full(len(events), np.nan)
    for s in sec_ret.columns:
        m = (sect == s).to_numpy()
        if m.any():
            sec_vals[m] = _series_at(sec_ret[s], cal, pos[m], 0)
    f["sector_ret_event"] = sec_vals
    f["excess_vs_sector"] = events["raw_return"].to_numpy() - sec_vals

    out = pd.concat([events.reset_index(drop=True), f.reset_index(drop=True)], axis=1)
    out["sector"] = sect.to_numpy()

    # --- crash-type classification -----------------------------------------
    spy_cut = expanding_low_pctile(spy_ret, benchmark_pctile)
    spy_cut_at = _series_at(spy_cut, cal, pos, 0)
    broad = out["spy_ret_event"].to_numpy() <= spy_cut_at

    sec_cut_at = np.full(len(events), np.nan)
    for s in sec_ret.columns:
        m = (sect == s).to_numpy()
        if m.any():
            cut = expanding_low_pctile(sec_ret[s], benchmark_pctile)
            sec_cut_at[m] = _series_at(cut, cal, pos[m], 0)
    sector_shock = (out["sector_ret_event"].to_numpy() <= sec_cut_at) & ~broad

    ctype = np.where(broad, "broad_market",
             np.where(sector_shock, "sector",
              np.where(np.isnan(spy_cut_at), "unclassified", "idiosyncratic")))
    # A missing sector reading cannot rule sector shock in or out.
    unknown_sector = np.isnan(sec_cut_at) | np.isnan(out["sector_ret_event"].to_numpy())
    ctype = np.where((ctype == "idiosyncratic") & unknown_sector, "unclassified", ctype)
    out["crash_type"] = ctype
    out["spy_pctile_cut"] = spy_cut_at
    out["sector_pctile_cut"] = sec_cut_at
    return out


def add_regimes(events: pd.DataFrame, vix_high_quantile: float) -> pd.DataFrame:
    """VIX and abnormal-volume regime labels used for conditional event studies."""
    out = events.copy()
    vix_cut = out["vix"].quantile(vix_high_quantile)
    regime = pd.Series(pd.NA, index=out.index, dtype="object")
    regime[out["vix"].notna() & (out["vix"] >= vix_cut)] = "high_vix"
    regime[out["vix"].notna() & (out["vix"] < vix_cut)] = "low_vix"
    out["vix_regime"] = regime
    out["vix_high_cut"] = vix_cut
    out["avol_quartile"] = pd.qcut(out["avol"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    out["severity_quartile"] = pd.qcut(out["crash_z"], 4, labels=["S1", "S2", "S3", "S4"])
    return out


def feature_matrix(events: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    cols = columns or FEATURE_COLUMNS
    return events[[c for c in cols if c in events.columns]].copy()
