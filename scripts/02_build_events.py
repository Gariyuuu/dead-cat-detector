"""Detect crash events, attach features and outcomes, persist the event table."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deadcat import data as D, events as E, features as F  # noqa: E402
from deadcat.config import PATHS, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("events")

# yfinance sector labels -> GICS sector labels used by the SPDR sector ETFs.
YF_TO_GICS = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Basic Materials": "Materials",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
}


def resolve_sectors(tickers: list[str], current: pd.DataFrame) -> pd.Series:
    """GICS sector per ticker: Wikipedia for current members, yfinance for the rest."""
    cache = PATHS.data_processed / "sector_map.parquet"
    known = current.set_index("ticker")["sector"].to_dict()
    if cache.exists():
        prev = pd.read_parquet(cache).set_index("ticker")["sector"].to_dict()
        known = {**prev, **known}
    missing = [t for t in tickers if t not in known or not isinstance(known.get(t), str)]
    if missing:
        import yfinance as yf
        log.info("resolving sector for %d historical-only tickers via yfinance", len(missing))
        for t in missing:
            sec = None
            try:
                sec = yf.Ticker(t).info.get("sector")
            except Exception:
                sec = None
            known[t] = YF_TO_GICS.get(sec) if sec else None
    ser = pd.Series({t: known.get(t) for t in tickers}, name="sector")
    ser.index.name = "ticker"
    ser.reset_index().to_parquet(cache, index=False)
    return ser


def main() -> None:
    cfg = load_config()
    PATHS.ensure()

    prices = D.load_processed("prices.parquet")
    bench_long = D.load_processed("benchmarks.parquet")
    windows = D.load_processed("membership.parquet")
    current = D.load_processed("sp500_current.parquet")

    close = E.to_wide(prices, "close")
    high = E.to_wide(prices, "high")
    low = E.to_wide(prices, "low")
    volume = E.to_wide(prices, "volume")
    bench_close = bench_long.pivot(index="date", columns="ticker", values="close").sort_index()
    bench_close = bench_close.reindex(close.index).ffill(limit=3)

    ret = E.daily_returns(close)
    eligible = D.membership_mask(windows, close.index, close.columns)
    log.info("panel: %d dates x %d tickers | eligible cells %.1fM",
             *close.shape, eligible.to_numpy().sum() / 1e6)

    ev = E.build_events(
        close, ret, eligible, bench_close[cfg.benchmarks.market],
        z_threshold=cfg.events.z_threshold,
        vol_window=cfg.events.vol_window,
        min_prior_obs=cfg.events.min_prior_obs,
        cooldown=cfg.events.cooldown_days,
        horizons=list(cfg.outcomes.horizons),
        primary_h=cfg.outcomes.primary_horizon,
        study_start=pd.Timestamp(cfg.study.start),
        study_end=pd.Timestamp(cfg.study.end),
    )
    log.info("events after cooldown: %d", len(ev))

    sectors = resolve_sectors(sorted(close.columns), current)
    log.info("sector known for %d / %d tickers", sectors.notna().sum(), len(sectors))

    sector_close = F.build_sector_series(
        bench_close, dict(cfg.benchmarks.sector_etfs), dict(cfg.benchmarks.sector_fallbacks)
    )
    ev = F.build_features(
        ev, close, high, low, volume, ret, bench_close, sector_close, sectors,
        market=cfg.benchmarks.market, vix=cfg.benchmarks.vix,
        vol_window=cfg.events.vol_window,
        benchmark_pctile=cfg.crash_type.benchmark_pctile,
    )
    ev = F.add_regimes(ev, cfg.regimes.vix_high_quantile)
    ev["complete_20"] = ev[f"car_{cfg.outcomes.primary_horizon}"].notna()

    D.save_processed("events.parquet", ev)

    summary = {
        "config_fingerprint": cfg.fingerprint,
        "n_events": int(len(ev)),
        "n_events_complete_car20": int(ev["complete_20"].sum()),
        "n_tickers": int(ev.ticker.nunique()),
        "date_min": str(ev.event_date.min().date()),
        "date_max": str(ev.event_date.max().date()),
        "crash_type_counts": ev.crash_type.value_counts().to_dict(),
        "sector_known_pct": round(100 * ev.sector.notna().mean(), 1),
        "mean_crash_z": float(ev.crash_z.mean()),
        "mean_raw_return": float(ev.raw_return.mean()),
        "feature_missing_pct": {
            c: round(100 * ev[c].isna().mean(), 2) for c in F.FEATURE_COLUMNS if c in ev
        },
    }
    (PATHS.metrics / "event_construction.json").write_text(json.dumps(summary, indent=2))
    log.info("summary: %s", json.dumps({k: v for k, v in summary.items()
                                        if k != "feature_missing_pct"}, indent=2))


if __name__ == "__main__":
    main()
