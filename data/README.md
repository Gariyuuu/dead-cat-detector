# Data

Everything in this study is derived from two public sources. No proprietary or
licensed data is used, and no raw dataset is committed to the repository.

| | |
|---|---|
| **Price source** | Yahoo Finance, retrieved with [`yfinance`](https://github.com/ranaroussi/yfinance) 1.7.0 |
| **Universe source** | Wikipedia: *List of S&P 500 companies* and *Historical components of the S&P 500* |
| **Download date** | 2026-09-02 (manifest records the exact UTC timestamp) |
| **Download window** | 2005-06-01 → 2026-08-31 |
| **Study window** | 2007-01-01 → 2026-08-31 (19.7 years) |
| **Panel after cleaning** | 2,976,172 ticker-days, 620 tickers, 5,345 trading dates |
| **Storage** | Parquet in `data/processed/` (git-ignored) |

Reproduce with `make data`. Provenance, row counts and coverage statistics are
written to `data/processed/manifest.json` on every build.

---

## Files

| File | Contents |
|---|---|
| `prices.parquet` | Adjusted daily OHLCV, long format: `date, ticker, open, high, low, close, volume` |
| `benchmarks.parquet` | SPY, `^VIX`, eleven SPDR sector ETFs, two long-history proxies |
| `membership.parquet` | Reconstructed point-in-time index windows: `ticker, start, end` (half-open) |
| `sp500_current.parquet` | 503 current constituents with GICS sector |
| `sp500_changes.parquet` | 407 recorded index additions/removals, 1976-2026 |
| `sector_map.parquet` | Ticker → GICS sector, including yfinance backfill for historical names |
| `reuse_rejects.parquet` | 43 symbols dropped by the ticker-reuse guard |
| `integrity_rejects.parquet` | 11 symbols dropped by the price-integrity screen |
| `events.parquet` | The event table: 14,678 crash events with features and outcomes |
| `manifest.json` | Source, dates, row counts, coverage, config fingerprint |

## Field definitions

**Prices.** `open/high/low/close` are **split- and dividend-adjusted** closing
levels (`yfinance` `auto_adjust=True`). Because the whole history is restated
on every download, a series pulled today differs slightly from the same series
pulled before a subsequent corporate action; the manifest timestamp is what
makes a run reproducible. `volume` is raw share volume, unadjusted.

**Returns.** Simple daily returns, `r = P_t / P_{t-1} - 1`, computed on the
adjusted close, so dividends and splits do not create artificial jumps.

**Benchmarks.** `SPY` is the market benchmark for all abnormal returns. `^VIX`
is the index level (not a tradeable series). Sector ETFs are mapped from GICS
sector: XLK, XLV, XLF, XLY, XLP, XLE, XLI, XLB, XLU, XLRE, XLC. **XLRE launched
in October 2015 and XLC in June 2018**, both inside the study window; IYR and
IYZ stand in before those inception dates. The substitution is explicit in
`configs/default.yaml` and applied in `features.build_sector_series`.

## Universe construction

The universe is the S&P 500 **reconstructed point in time**. The current
constituent list is rolled backwards through the recorded change log: walking
in reverse, an addition on date *d* closes a ticker's membership window at its
left edge, and a removal on *d* opens a window whose right edge is *d*. This
yields 868 membership windows over 846 distinct tickers.

A stock is eligible for event detection **only on dates it actually belonged to
the index**. This removes look-ahead in universe *selection* — the study never
detects a "crash" in a company that had not yet joined the index, nor after it
left.

Membership windows are floored at 2005-06-01. Names already in the index before
the change log's reliable coverage begins receive a window starting at that
floor.

**Why the study starts in 2007.** The Wikipedia change log is dense and
consistent with reality from 2007 onward (11-30 recorded changes per year,
matching the ~20/yr the index actually experiences). Before 2007 it is sparse —
two changes in 2005, one in 2006 — which is clearly incomplete. Reconstructed
membership before 2007 would be unreliable, so the study window begins there.
39 change-log rows could not be reconciled against the current list and are
counted as anomalies in the build log.

## Data-quality screens

Two screens run before any analysis. Both drop whole tickers rather than
individual days, because in each case the defect is the *series*.

### 1. Ticker-reuse guard — 43 symbols dropped

Yahoo serves a **different company** under a recycled symbol. `SBNY` returns a
listing beginning in August 2024, seventeen months after Signature Bank failed;
`BBBY` returns data starting July 2026. Any price history that does not overlap
its symbol's recorded membership window is rejected. Examples caught: SBNY,
BBBY, APC, ADCT, BEAM, BTU, ADT, AV, ABS.

### 2. Price-integrity screen — 11 symbols dropped

Some Yahoo series are not credible price histories at all. Three independent
rules, each verified by inspecting every rejection:

| Rule | Catches | Threshold |
|---|---|---|
| Round-trip oscillation | Two securities interleaved under one symbol — TIE alternates between ~$14 and ~$8,000 day to day; a botched 2026 split adjustment makes MNST oscillate between ~$47 and ~$95 | ≥ 3 round trips (2× move reversed within 5 sessions) |
| Extreme-move frequency | Persistently implausible series | > 12 days moving more than ±50% |
| Value diversity | Quantised junk served under a large-cap ticker — COL trades at $0.20-$0.85 where Rockwell Collins traded $60-$140 | distinct closes / observations < 0.25 |

Dropped: **BMC, CBE, CFC, COL, CPWR, EP, GLK, GR, MEE, MNST, TIE**.

The thresholds were set so that genuine extremes survive: GameStop's January
2021 squeeze (one round trip, 10 days over ±50%), Genworth's 2008 collapse,
Hartford's +102% day in December 2008, and Nektar's ±60-156% trial-news moves
are all retained. The value-diversity rule deliberately replaced an earlier
price-level rule, which falsely flagged **NVDA** — its split-adjusted median
close of $0.80 is legitimate, not corruption.

## Missingness

The panel has no missing values in `date, ticker, open, high, low, close,
volume`. 20,996 rows (0.7%) carry zero volume, which are treated as missing for
volume features rather than logged as `-inf`.

Feature-level missingness in the event table:

| Feature | Missing |
|---|---|
| `volume`, `log_volume` | 0.01% |
| `med_volume_60`, `avol` | 0.3% |
| `mom_252` | 0.4% |
| `sector_ret_event`, `excess_vs_sector` | 1.0% |

Sector is known for 605 of 620 tickers (99.1% of events): from Wikipedia for
current members, and by a cached best-effort `yfinance` lookup for
historical-only names. Events whose sector is unknown cannot be tested against
the sector-shock condition and are labelled `unclassified` (89 events, 0.6%)
rather than being defaulted into `idiosyncratic`.

## Survivorship bias — what is and is not solved

**Solved:** look-ahead in universe selection. Point-in-time membership is
genuinely incorporated, so eligibility on any date reflects the index as it
actually stood.

**Not solved, and not claimed to be:** Yahoo has purged the price history of
most companies that were acquired or delisted. Of 343 tickers that belonged to
the index during the window but are not current members, only **120 (35.0%)
have usable price data**; 223 are eligible but unobservable. Current-member
coverage is 99.4%.

The surviving historical names therefore over-represent firms that were
acquired at a healthy price and under-represent firms that collapsed. Because
this study measures *post-crash* behaviour, the omission plausibly biases mean
CAR **upward** — the worst outcomes are the ones most likely to be missing. The
headline result is that post-crash CAR is negative, so this bias works against
the finding rather than producing it. That asymmetry is the reason the result is
reported as directionally safe rather than precise.

One further residual: for a small number of acquired names Yahoo's history
appears to correspond to a successor or foreign listing (HAR trades at
2,170-35,618 where Harman International traded $40-110 in USD). These series are
internally smooth and pass both screens, but they are not necessarily the
S&P 500 constituent's own history.

## Licensing and terms

Yahoo Finance data is retrieved through `yfinance`, an unaffiliated open-source
client, and is used here for **non-commercial academic research**. Yahoo's terms
of service govern redistribution, which is why **no raw or processed price data
is committed to this repository** — only the code that reproduces it. Wikipedia
content is CC BY-SA 4.0. The code in this repository is MIT licensed.

## Known limitations

1. Delisted-name coverage is 35%; residual survivorship bias remains (above).
2. Yahoo's adjusted history is restated over time, so exact reproduction
   requires the same download date; the manifest records it.
3. Index membership comes from a crowd-edited source, not S&P's official
   record, and is unreliable before 2007.
4. No intraday data: crash detection is close-to-close, so a stock that fell
   30% intraday and closed flat is not an event.
5. No delisting returns. A stock removed mid-window simply stops contributing;
   the terminal value to shareholders is not modelled. This is a second channel
   through which the worst outcomes are under-counted.
6. Sector ETFs proxy for GICS sectors and are imperfect before XLRE (2015) and
   XLC (2018) existed.
7. The universe is large-cap US equity only. Nothing here extends to small caps,
   non-US markets, or other asset classes.
