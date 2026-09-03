"""Shared loading helpers so every script sees an identical panel."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import data as D, events as E, features as F
from .config import Config


class Panel:
    """The full price/benchmark panel plus everything derived from it."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        prices = D.load_processed("prices.parquet")
        bench_long = D.load_processed("benchmarks.parquet")
        self.windows = D.load_processed("membership.parquet")
        self.current = D.load_processed("sp500_current.parquet")

        self.close = E.to_wide(prices, "close")
        self.high = E.to_wide(prices, "high")
        self.low = E.to_wide(prices, "low")
        self.volume = E.to_wide(prices, "volume")
        self.bench = (bench_long.pivot(index="date", columns="ticker", values="close")
                      .sort_index().reindex(self.close.index).ffill(limit=3))
        self.ret = E.daily_returns(self.close)
        self.eligible = D.membership_mask(self.windows, self.close.index, self.close.columns)
        self.spy = self.bench[cfg.benchmarks.market]
        self.sector_close = F.build_sector_series(
            self.bench, dict(cfg.benchmarks.sector_etfs), dict(cfg.benchmarks.sector_fallbacks)
        )
        smap = D.load_processed("sector_map.parquet")
        self.sectors = smap.set_index("ticker")["sector"]

    def build(self, *, z_threshold=None, vol_window=None, cooldown=None,
              vix_high_quantile=None) -> pd.DataFrame:
        """Construct a full event table for one (possibly overridden) specification."""
        c = self.cfg
        ev = E.build_events(
            self.close, self.ret, self.eligible, self.spy,
            z_threshold=z_threshold if z_threshold is not None else c.events.z_threshold,
            vol_window=vol_window if vol_window is not None else c.events.vol_window,
            min_prior_obs=vol_window if vol_window is not None else c.events.min_prior_obs,
            cooldown=cooldown if cooldown is not None else c.events.cooldown_days,
            horizons=list(c.outcomes.horizons),
            primary_h=c.outcomes.primary_horizon,
            study_start=pd.Timestamp(c.study.start),
            study_end=pd.Timestamp(c.study.end),
        )
        if ev.empty:
            return ev
        ev = F.build_features(
            ev, self.close, self.high, self.low, self.volume, self.ret,
            self.bench, self.sector_close, self.sectors,
            market=c.benchmarks.market, vix=c.benchmarks.vix,
            vol_window=vol_window if vol_window is not None else c.events.vol_window,
            benchmark_pctile=c.crash_type.benchmark_pctile,
        )
        ev = F.add_regimes(ev, vix_high_quantile if vix_high_quantile is not None
                           else c.regimes.vix_high_quantile)
        ev["complete_20"] = ev[f"car_{c.outcomes.primary_horizon}"].notna()
        return ev
