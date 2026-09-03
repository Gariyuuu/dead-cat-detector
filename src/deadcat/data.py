"""Universe construction, price acquisition and on-disk caching.

The universe is the S&P 500 reconstructed *point in time*: the current
constituent list is rolled backwards through the recorded index add/remove
events, so a stock is only eligible for event detection on dates it actually
belonged to the index. This removes look-ahead in universe *selection*.

It does not remove survivorship bias entirely. Yahoo Finance has purged the
price history of most companies that were acquired or delisted, so those names
are eligible but unobservable. The magnitude of that gap is measured in
:func:`universe_coverage_report` and reported in ``data/README.md``.

Ticker reuse is an active hazard: Yahoo serves a *different* company under a
recycled symbol (SBNY, for example, returns a post-2024 listing rather than the
failed Signature Bank). :func:`apply_reuse_guard` drops any price history that
does not overlap the symbol's recorded membership window.
"""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import PATHS, Config

log = logging.getLogger(__name__)

WIKI_CURRENT = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_CHANGES = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
_UA = {"User-Agent": "Mozilla/5.0 (dead-cat-detector; academic research)"}

# Yahoo uses '-' where the index uses '.' (BRK.B -> BRK-B).
def yahoo_symbol(sym: str) -> str:
    return str(sym).strip().upper().replace(".", "-")


def valid_ticker(x) -> bool:
    """True only for a real symbol. NaN is truthy in Python - never test bare."""
    return isinstance(x, str) and bool(x.strip()) and x.strip().lower() != "nan"


# ---------------------------------------------------------------- universe ---
def _read_wiki_tables(url: str) -> list[pd.DataFrame]:
    req = urllib.request.Request(url, headers=_UA)
    html = urllib.request.urlopen(req, timeout=90).read().decode("utf-8")
    return pd.read_html(io.StringIO(html))


def fetch_sp500_current() -> pd.DataFrame:
    """Current S&P 500 constituents with GICS sector."""
    tbl = _read_wiki_tables(WIKI_CURRENT)[0]
    out = tbl.rename(
        columns={"Symbol": "ticker", "Security": "security", "GICS Sector": "sector"}
    )[["ticker", "security", "sector"]].copy()
    out["ticker"] = out["ticker"].map(yahoo_symbol)
    return out.drop_duplicates("ticker").reset_index(drop=True)


def fetch_sp500_changes() -> pd.DataFrame:
    """Recorded index additions/removals, one row per change."""
    tbl = _read_wiki_tables(WIKI_CHANGES)[0]
    tbl.columns = [f"{a}_{b}" if a != b else a for a, b in tbl.columns]
    out = tbl.rename(
        columns={
            "Effective Date": "date",
            "Added_Ticker": "added",
            "Removed_Ticker": "removed",
        }
    )[["date", "added", "removed"]].copy()
    out["date"] = pd.to_datetime(out["date"], format="mixed", errors="coerce")
    for c in ("added", "removed"):
        out[c] = out[c].map(lambda v: yahoo_symbol(v) if valid_ticker(v) else None)
        out[c] = out[c].astype("object").where(out[c].notna(), None)
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def build_membership_windows(
    current: pd.DataFrame, changes: pd.DataFrame, floor: pd.Timestamp, ceiling: pd.Timestamp
) -> pd.DataFrame:
    """Roll the current list backwards into ``[start, end)`` membership windows.

    Walking the change log in reverse: an *addition* on date ``d`` closes the
    ticker's window at its left edge (it joined on ``d``); a *removal* on ``d``
    opens a new window whose right edge is ``d`` (it was a member until then).
    """
    ceiling = pd.Timestamp(ceiling)
    floor = pd.Timestamp(floor)

    live = set(current["ticker"])
    open_end: dict[str, pd.Timestamp] = {t: ceiling for t in live}
    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    anomalies = 0

    for row in changes.sort_values("date", ascending=False).itertuples(index=False):
        d, added, removed = row.date, row.added, row.removed
        added = added if valid_ticker(added) else None
        removed = removed if valid_ticker(removed) else None
        if added:
            if added in live:
                windows[added].append((d, open_end.pop(added)))
                live.discard(added)
            else:
                anomalies += 1  # add with no matching later removal on record
        if removed:
            if removed in live:
                anomalies += 1  # removal of a name already treated as a member
            else:
                live.add(removed)
                open_end[removed] = d

    # Anything still live predates the change log's reliable coverage.
    for t in list(live):
        windows[t].append((floor, open_end.pop(t)))

    rows = [
        {"ticker": t, "start": s, "end": e}
        for t, spans in windows.items()
        for (s, e) in spans
        if e > floor and s < ceiling
    ]
    out = pd.DataFrame(rows).sort_values(["ticker", "start"]).reset_index(drop=True)
    out["start"] = out["start"].clip(lower=floor)
    out["end"] = out["end"].clip(upper=ceiling)
    log.info("membership: %d windows / %d tickers (%d log anomalies)",
             len(out), out.ticker.nunique(), anomalies)
    return out


def membership_mask(windows: pd.DataFrame, index: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """Boolean ``date x ticker`` frame: True where the name was in the index."""
    mask = pd.DataFrame(False, index=index, columns=sorted(tickers))
    for row in windows.itertuples(index=False):
        if row.ticker in mask.columns:
            mask.loc[(index >= row.start) & (index < row.end), row.ticker] = True
    return mask


# ---------------------------------------------------------------- download ---
def _download_batch(tickers: list[str], start: str, end: str, retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers, start=start, end=end, auto_adjust=True,
                progress=False, threads=True, group_by="column",
            )
            if raw is not None and len(raw):
                return raw
        except Exception as exc:  # pragma: no cover - network
            log.warning("batch failed (attempt %d): %s", attempt + 1, exc)
        time.sleep(2 * (attempt + 1))
    return pd.DataFrame()


def _to_long(raw: pd.DataFrame, fields=("Open", "High", "Low", "Close", "Volume")) -> pd.DataFrame:
    """Wide yfinance output -> tidy long frame."""
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker", *[f.lower() for f in fields]])
    frames = []
    for f in fields:
        if f not in raw.columns.get_level_values(0):
            continue
        sub = raw[f]
        if isinstance(sub, pd.Series):
            sub = sub.to_frame()
        frames.append(sub.stack(future_stack=True).rename(f.lower()))
    out = pd.concat(frames, axis=1).reset_index()
    out.columns = ["date", "ticker", *[c for c in out.columns[2:]]]
    return out.dropna(subset=["close"])


def download_prices(tickers: list[str], start: str, end: str, batch_size: int = 60) -> pd.DataFrame:
    """Download adjusted daily OHLCV for ``tickers`` in batches."""
    tickers = sorted(set(tickers))
    chunks = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        log.info("download %d-%d / %d", i, i + len(batch), len(tickers))
        chunks.append(_to_long(_download_batch(batch, start, end)))
        time.sleep(0.4)
    if not chunks:
        return pd.DataFrame()
    out = pd.concat(chunks, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def apply_reuse_guard(prices: pd.DataFrame, windows: pd.DataFrame,
                      min_overlap_days: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop price history that does not overlap the symbol's membership window.

    Protects against recycled tickers where Yahoo returns an unrelated company.
    Returns ``(kept_prices, rejection_report)``.
    """
    spans = prices.groupby("ticker")["date"].agg(["min", "max", "count"])
    rejects = []
    keep = []
    for tkr, grp in windows.groupby("ticker"):
        if tkr not in spans.index:
            continue
        lo, hi = spans.loc[tkr, "min"], spans.loc[tkr, "max"]
        overlap = any(
            (min(hi, r.end) - max(lo, r.start)).days > 0 for r in grp.itertuples(index=False)
        )
        n_in = int(
            sum(
                ((prices.ticker == tkr) & (prices.date >= r.start) & (prices.date < r.end)).sum()
                for r in grp.itertuples(index=False)
            )
        )
        if overlap and n_in >= min_overlap_days:
            keep.append(tkr)
        else:
            rejects.append(
                {"ticker": tkr, "price_start": lo, "price_end": hi,
                 "membership_start": grp["start"].min(), "membership_end": grp["end"].max(),
                 "overlap_obs": n_in, "reason": "no_membership_overlap"}
            )
    return (prices[prices.ticker.isin(keep)].reset_index(drop=True),
            pd.DataFrame(rejects))


def price_integrity_screen(
    prices: pd.DataFrame,
    jump_factor: float = 2.0,
    roundtrip_days: int = 5,
    roundtrip_band: tuple[float, float] = (0.67, 1.5),
    min_round_trips: int = 3,
    max_extreme_moves: int = 12,
    extreme_move: float = 0.50,
    min_distinct_ratio: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject price series that are not credible price histories.

    Yahoo's history for some delisted or split-adjusted symbols is corrupt in
    ways no winsorisation can repair - two securities interleaved under one
    symbol (TIE alternates between ~$14 and ~$8,000), a botched split
    adjustment (MNST oscillates between ~$47 and ~$95 through July 2026), or a
    quantised penny series served under a large-cap's ticker (COL trades at
    $0.20-$0.85 where Rockwell Collins traded $60-$140). Because the defect is
    the *series*, the whole ticker is dropped rather than individual days.

    Three independent rules, each verified against the full rejection list:

    ``round trips``
        A day whose price moves by ``jump_factor`` or more and returns to
        within ``roundtrip_band`` of its pre-jump level inside
        ``roundtrip_days``. Requiring ``min_round_trips`` separates repeated
        mechanical oscillation from a genuine one-off squeeze - GameStop
        round-trips once in January 2021 and is retained.
    ``extreme-move frequency``
        More than ``max_extreme_moves`` days moving over ``extreme_move``.
        Genuine crisis names top out around ten (GME 10, GNW 6, AIG 3).
    ``value diversity``
        Distinct closes as a share of observations. Real daily series are
        near-unique (median 0.93 across this universe); quantised junk is
        below 0.25. Deliberately *not* a price-level rule: NVDA's split
        adjusted median close is $0.80 and is entirely legitimate.

    Returns ``(kept_prices, rejection_report)``.
    """
    import numpy as np

    lo_f, hi_f = roundtrip_band
    reports = []
    for tkr, grp in prices.groupby("ticker", sort=False):
        c = grp.set_index("date")["close"].sort_index()
        if len(c) < 10:
            continue
        vals = c.to_numpy()
        rel = vals[1:] / vals[:-1]
        hits = np.flatnonzero((rel >= jump_factor) | (rel <= 1.0 / jump_factor))
        n_rt = 0
        for h in hits:
            i = h + 1
            pre = vals[i - 1]
            fwd = vals[i + 1 : i + 1 + roundtrip_days]
            if fwd.size and np.any((fwd / pre > lo_f) & (fwd / pre < hi_f)):
                n_rt += 1
        n_big = int(np.sum(np.abs(rel - 1.0) > extreme_move))
        ratio = c.nunique() / len(c)

        failed = []
        if n_rt >= min_round_trips:
            failed.append("round_trip_oscillation")
        if n_big > max_extreme_moves:
            failed.append("extreme_move_frequency")
        if ratio < min_distinct_ratio:
            failed.append("low_value_diversity")
        if failed:
            reports.append(
                {"ticker": tkr, "n_round_trips": int(n_rt), "n_extreme_moves": n_big,
                 "distinct_ratio": round(float(ratio), 4), "obs": int(len(c)),
                 "min_close": float(vals.min()), "max_close": float(vals.max()),
                 "reason": "+".join(failed)}
            )
    rejects = pd.DataFrame(reports)
    bad = set(rejects["ticker"]) if len(rejects) else set()
    return prices[~prices.ticker.isin(bad)].reset_index(drop=True), rejects


def universe_coverage_report(windows: pd.DataFrame, prices: pd.DataFrame,
                             current: pd.DataFrame) -> dict:
    """How much of the point-in-time universe actually has price data."""
    eligible = set(windows["ticker"])
    have = set(prices["ticker"].unique())
    cur = set(current["ticker"])
    historical_only = eligible - cur
    return {
        "eligible_tickers_point_in_time": len(eligible),
        "current_members": len(cur),
        "historical_only_members": len(historical_only),
        "tickers_with_price_data": len(have & eligible),
        "historical_only_with_data": len(historical_only & have),
        "historical_only_missing": len(historical_only - have),
        "historical_only_coverage_pct": round(
            100 * len(historical_only & have) / max(len(historical_only), 1), 1
        ),
        "current_member_coverage_pct": round(100 * len(cur & have) / max(len(cur), 1), 1),
    }


# ------------------------------------------------------------------- cache ---
def _p(name: str) -> Path:
    return PATHS.data_processed / name


def save_processed(name: str, df: pd.DataFrame) -> Path:
    PATHS.ensure()
    path = _p(name)
    df.to_parquet(path, index=False)
    return path


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(_p(name))


def write_manifest(payload: dict) -> Path:
    PATHS.ensure()
    payload = dict(payload)
    payload["generated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = _p("manifest.json")
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def build_benchmark_panel(bench_long: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Wide close/volume panel for SPY, VIX and the sector ETFs (+ fallbacks)."""
    close = bench_long.pivot(index="date", columns="ticker", values="close").sort_index()
    return close
