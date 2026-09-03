"""Build the point-in-time universe and download all price data.

Outputs (data/processed/):
  sp500_current.parquet     current constituents + GICS sector
  sp500_changes.parquet     recorded index add/remove log
  membership.parquet        reconstructed [ticker, start, end) windows
  prices.parquet            adjusted daily OHLCV for the universe
  benchmarks.parquet        SPY, ^VIX, sector ETFs
  reuse_rejects.parquet     symbols dropped by the ticker-reuse guard
  manifest.json             provenance + coverage statistics
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deadcat import data as D  # noqa: E402
from deadcat.config import PATHS, load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build")


def main() -> None:
    cfg = load_config()
    PATHS.ensure()
    floor = pd.Timestamp(cfg.study.warmup_start)
    ceiling = pd.Timestamp(cfg.study.end) + pd.Timedelta(days=1)

    log.info("fetching index membership from Wikipedia")
    current = D.fetch_sp500_current()
    changes = D.fetch_sp500_changes()
    D.save_processed("sp500_current.parquet", current)
    D.save_processed("sp500_changes.parquet", changes)
    log.info("current=%d changes=%d (%s..%s)", len(current), len(changes),
             changes.date.min().date(), changes.date.max().date())

    windows = D.build_membership_windows(current, changes, floor, ceiling)
    D.save_processed("membership.parquet", windows)
    tickers = sorted(windows.ticker.unique())
    log.info("point-in-time universe: %d distinct tickers", len(tickers))

    log.info("downloading equity prices ...")
    prices = D.download_prices(tickers, cfg.study.warmup_start, cfg.study.end)
    log.info("raw rows=%d tickers=%d", len(prices), prices.ticker.nunique())

    prices, rejects = D.apply_reuse_guard(prices, windows)
    D.save_processed("reuse_rejects.parquet", rejects)
    log.info("reuse guard dropped %d symbols", len(rejects))

    prices, integrity = D.price_integrity_screen(prices)
    D.save_processed("integrity_rejects.parquet", integrity)
    log.info("integrity screen dropped %d symbols: %s", len(integrity),
             sorted(integrity.ticker.tolist()) if len(integrity) else [])

    D.save_processed("prices.parquet", prices)

    bench = sorted(
        {cfg.benchmarks.market, cfg.benchmarks.vix}
        | set(dict(cfg.benchmarks.sector_etfs).values())
        | set(dict(cfg.benchmarks.sector_fallbacks).values())
    )
    log.info("downloading benchmarks: %s", bench)
    bench_long = D.download_prices(bench, cfg.study.warmup_start, cfg.study.end)
    D.save_processed("benchmarks.parquet", bench_long)

    cov = D.universe_coverage_report(windows, prices, current)
    px_span = (prices.date.min(), prices.date.max())
    manifest = {
        "config_fingerprint": cfg.fingerprint,
        "source": "Yahoo Finance via yfinance (prices); Wikipedia (index membership)",
        "study_window": [cfg.study.start, cfg.study.end],
        "download_window": [cfg.study.warmup_start, cfg.study.end],
        "price_rows": int(len(prices)),
        "price_tickers": int(prices.ticker.nunique()),
        "price_date_min": str(px_span[0].date()),
        "price_date_max": str(px_span[1].date()),
        "benchmark_rows": int(len(bench_long)),
        "benchmark_tickers": sorted(bench_long.ticker.unique().tolist()),
        "index_changes_rows": int(len(changes)),
        "index_changes_span": [str(changes.date.min().date()), str(changes.date.max().date())],
        "reuse_guard_rejects": int(len(rejects)),
        "integrity_rejects": int(len(integrity)),
        "integrity_rejected_tickers": sorted(integrity.ticker.tolist()) if len(integrity) else [],
        "coverage": cov,
    }
    D.write_manifest(manifest)
    log.info("coverage: %s", cov)
    log.info("done")


if __name__ == "__main__":
    main()
