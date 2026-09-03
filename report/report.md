# Dead Cat Detector

### When extreme stock selloffs reverse — and when they keep falling

*A conditional event study of 14,678 extreme single-day declines in S&P 500
constituents, 2007–2026.*

---

## Abstract

Extreme single-day equity declines are widely believed to be followed by a
bounce. Using 14,678 crash events — days on which an S&P 500 constituent fell
at least three standard deviations relative to its own trailing 60-day
distribution — drawn from a point-in-time reconstruction of index membership
between 2007 and 2026, this study finds no such bounce. Mean 20-day cumulative
abnormal return against SPY is **−0.27%** (95% bootstrap CI [−0.40%, −0.12%]),
and only **47.1%** of events end the 20-day window above the benchmark. The
result is small in magnitude but directionally unanimous: across 48 combinations
of crash threshold, volatility window and cooldown, mean CAR is negative in
**48 of 48** and the recovery rate is below one-half in **all** of them. Neither
crash type, abnormal volume, pre-crash momentum nor the VIX regime separates
recovering events from continuing decliners at the 20-day horizon after
correction for multiple comparisons, and **none of 32 exploratory coefficients
survives a Benjamini–Hochberg correction**. Three model families — logistic
regression, random forest and LightGBM — fail to beat a constant base-rate
forecast on a chronologically embargoed hold-out block. Post-crash behaviour is
best described as a slightly-worse-than-coin-flip draw from a wide distribution.

---

## 1. Motivation

"Dead cat bounce" is one of the most durable pieces of market folklore: a stock
that falls violently is supposed to rebound, at least briefly, because the
selling was excessive. The intuition has respectable academic ancestry.
[Short-horizon reversal](https://doi.org/10.1093/rfs/3.2.175) is one of the
oldest documented anomalies in the cross-section of equity returns, and
microstructure arguments — liquidity provision, inventory risk, price pressure
from forced sellers — give a mechanism for why a violent decline might overshoot.

The countervailing literature is just as respectable. Post-earnings-announcement
drift and momentum both imply that bad news is *under*-reacted to, not
over-reacted to, and that prices continue in the direction of the shock. Whether
an individual extreme decline reverses or continues is therefore genuinely
ambiguous ex ante, and the answer plausibly depends on what kind of decline it
was.

## 2. Research question

> After a liquid U.S. equity experiences an unusually severe one-day decline,
> which observable characteristics distinguish subsequent 20-trading-day mean
> reversion from continued underperformance?

The interest is not in whether a strategy would have made money — no transaction costs, borrowing costs or
capacity constraints are modelled, and none of what follows is a trading signal.
The interest is in whether the *observable characteristics of a crash at the
moment it happens* carry information about what comes next. If they do,
post-crash returns are a conditional phenomenon worth modelling. If they do not,
the folklore is describing noise.

## 3. Hypotheses

Five hypotheses were pre-registered in `docs/research_plan.md` before
estimation.

* **H1** — Extreme declines do not universally mean-revert.
* **H2** — Broad-market crashes recover differently from idiosyncratic ones.
* **H3** — Abnormal volume carries information about subsequent behaviour.
* **H4** — Pre-crash momentum matters.
* **H5** — Market volatility (VIX) changes the relationship.

None was framed directionally. A null was an acceptable answer to each, and in
four cases a null is the answer.

## 4. Data

Daily split- and dividend-adjusted OHLCV comes from Yahoo Finance; index
membership from Wikipedia's current-constituent list and its record of
historical index changes. The study window is 2007-01-01 to 2026-08-31 — 19.7
years — with price history downloaded from 2005-06-01 so that 252-day momentum
and 60-day volatility are fully warmed up on the first eligible event date.

## 5. Point-in-time universe

The S&P 500 is reconstructed *point in time*. The current list is
rolled backwards through the change log — walking in reverse, an addition closes
a ticker's membership window at its left edge, a removal opens one whose right
edge is the removal date — producing 868 membership windows across 846 tickers.
A stock is eligible for event detection only on dates it actually belonged to
the index. The window begins in 2007 because the change log is dense and
realistic from then on (11–30 changes per year) and demonstrably incomplete
before (two changes recorded in 2005, one in 2006).

## 6. Data quality

**Data quality was the single largest engineering problem in this study**, and
it is worth being concrete, because the raw data is materially corrupt in ways
that silently destroy results. Two screens run before any analysis, and both
drop whole tickers rather than individual days, because in each case the defect
is the series rather than an observation.

The first is a *ticker-reuse guard*. Yahoo serves a different company under a
recycled symbol: `SBNY` returns a listing beginning in August 2024, seventeen
months after Signature Bank failed. Any price history that does not overlap its
symbol's membership window is rejected — 43 symbols, including SBNY, BBBY, APC,
ADCT and BEAM.

The second is a *price-integrity screen*. Some series are not price histories at
all. `TIE` alternates day to day between roughly $14 and roughly $8,000 — two
securities interleaved under one symbol. `CBE` alternates between $52 and $0.08.
`COL` trades at $0.20–$0.85 where Rockwell Collins traded $60–$140. A botched
2026 split adjustment makes `MNST` oscillate between about $47 and $95. Three
rules — repeated round-trip level discontinuities, implausible frequency of
±50% days, and low distinct-value diversity — remove eleven tickers (BMC, CBE,
CFC, COL, CPWR, EP, GLK, GR, MEE, MNST, TIE). Every rejection was inspected
individually. The thresholds are set so that genuine extremes survive: GameStop's
January 2021 squeeze, Genworth's 2008 collapse, Hartford's +102% day in December
2008 and Nektar's trial-news moves are all retained. An earlier version of the
screen used a price-level rule and falsely flagged **NVDA**, whose split-adjusted
median close of $0.80 is entirely legitimate; that rule was replaced.

Before these screens, mean CAR₂₀ was **+77.7%** with a standard deviation of
**58** — six corrupted events on two tickers were overwhelming fourteen thousand
real ones. Afterwards the same statistic is −0.27% with a standard deviation of
0.084. No amount of winsorisation would have been an honest substitute for
identifying and removing series that were never real prices.

The cleaned panel holds 2,976,172 ticker-days across 620 tickers and 5,345
trading dates.

**Survivorship bias is reduced, not solved.** Point-in-time membership removes
look-ahead in universe *selection*. It does not conjure data that Yahoo has
purged: of 343 tickers that belonged to the index during the window but are not
current members, only **120 (35.0%)** have usable price history. Current members
are 99.4% covered. Surviving historical names therefore over-represent firms
acquired at a healthy price and under-represent firms that collapsed. Because
this study measures post-crash behaviour, that omission plausibly biases mean
CAR *upward* — the worst outcomes are the ones most likely to be missing. The
headline finding is that post-crash CAR is negative, so the bias works against
the result rather than manufacturing it. That asymmetry is why the finding is
presented as directionally safe rather than precisely calibrated.

## 7. Event construction

A **crash event** is a day on which

$$z_{i,t} = \frac{r_{i,t} - \mu^{60}_{i,t-1}}{\sigma^{60}_{i,t-1}} \le -3.0,$$

where μ and σ are estimated over the 60 trading days *strictly preceding* *t*.
The trailing shift is what makes the estimate causal, and it is enforced by a
test that mutates every return from *t* onward and asserts that μ and σ at *t*
are unchanged.

Consecutive breaches are not independent observations, so a **ticker-specific
cooldown** of 20 trading days collapses an episode to its first day. On the real
event table the minimum trading-day gap between two events on the same ticker is
exactly 20, with zero violations.

This yields **14,678 events** on 600 tickers, a median of −3.75σ and a median
event-day return of **−5.4%**. Events cluster sharply in market-wide stress:
the 2020 first quarter alone contributes 629.

Outcomes are measured **from the event close** — the earliest moment the event
is observable — so the crash-day move is excluded from the outcome. The primary
outcome is

$$CAR_{i,t,20} = R_{i,t:t+20} - R_{SPY,t:t+20},$$

and the primary classification target is `recovered_20d = 1[CAR₂₀ > 0]`.

Twenty-one features are computed, all observable at the event close: crash
severity, volume and abnormal volume, momentum at four horizons, realised
volatility, market and sector context. Trailing windows end at *t−1*.
Crash types are assigned operationally — *broad-market* when SPY's event-day
return is at or below the 5th percentile of its own expanding prior
distribution, *sector* when the sector ETF is and SPY is not, *idiosyncratic*
when neither. These labels describe co-movement and are explicitly **not** a
claim about what caused the decline.

Inference uses 2,000-resample bootstraps (resampling whole events, to preserve
within-path autocorrelation) and OLS with HC3 standard errors. One addition to
the plan, declared in advance of interpretation: because hundreds of names can
breach the threshold on the same market-wide day, residuals are strongly
correlated within a date and HC3 alone overstates precision. **Date-clustered
standard errors are reported alongside HC3 throughout, and where the two
disagree the clustered result governs.**

## 8. Event study

**Extreme declines do not mean-revert.** Mean CAR against SPY drifts steadily
*downward* through event time. At the primary horizon it is **−0.265%**, with a
95% bootstrap CI of **[−0.403%, −0.124%]** and p < 0.001. The term structure is
negative at every horizon tested:

| Horizon | Mean CAR | 95% CI | n |
|---|---|---|---|
| 1 day | −0.101% | [−0.141%, −0.064%] | 14,676 |
| 5 days | −0.268% | [−0.342%, −0.192%] | 14,675 |
| 10 days | −0.187% | [−0.283%, −0.089%] | 14,668 |
| **20 days** | **−0.265%** | **[−0.403%, −0.124%]** | **14,647** |
| 60 days | −0.425% | [−0.658%, −0.200%] | 14,539 |

**H1 is supported.** There is no bounce on average; there is mild continued
underperformance that deepens slowly out to three months.

The proportion of events ending above the benchmark is **47.1%**, and only
**38.9%** regain their pre-crash closing price within 20 sessions. Of those that
do recover, the median takes 9 trading days.

The magnitudes deserve emphasis. A −0.27% mean sits inside a distribution with a
standard deviation of **8.4%** — the dispersion is thirty times the central
tendency. Mean adverse excursion over the 20 days is −5.9% and mean favourable
excursion is +5.7%. The honest description of a post-crash outcome is a very
wide, almost symmetric draw whose centre is displaced slightly to the left, not
a systematic tendency in either direction.

**H2 is not supported at the 20-day horizon.** Broad-market crashes average
−0.351%, sector −0.241%, idiosyncratic −0.226%. Every pairwise contrast is
indistinguishable from zero: broad-market minus idiosyncratic is −0.12pp
(95% CI [−0.44, +0.17], p = 0.43); sector minus idiosyncratic is −0.02pp
(p = 0.95); broad minus sector is −0.11pp (p = 0.68). None survives FDR
correction. This is the hero figure of the study, and what it shows is three
confidence bands sitting on top of one another.

At 60 days a suggestion of separation appears — sector crashes turn positive
(+0.32%) while idiosyncratic ones deepen to −0.59% — but the sector band is wide
(n = 1,425) and this contrast was not pre-registered at that horizon. It is
reported as a hypothesis for future work, not a finding.

The one place crash type matters clearly is the *secondary* outcome. The share
of events regaining their pre-crash price within 20 sessions is **48.2%** after
a broad-market crash but only **32.6%** after an idiosyncratic one. Broad-market
declines are shallower and are more often mechanically undone when the market
rebounds; a firm-specific collapse resets the price level. That difference is
about the depth of the hole, not about differential abnormal performance out of
it — CAR₂₀ is the same in both cases.

**H3 is directionally consistent but does not survive correction.** The highest
abnormal-volume quartile averages −0.138% against −0.565% for the lowest, a
difference of **+0.43pp** (95% CI [+0.05, +0.81], nominal p = 0.028). After
Benjamini–Hochberg correction across the five pre-registered contrasts,
**q = 0.14** — not significant. It is nonetheless the most robust of the
conditional effects: the sign holds in **97.9%** of the 576 robustness
specifications. The reasonable reading is a small real effect that this sample
cannot establish at conventional thresholds.

**H5 is not supported.** High-VIX events average −0.182%, low-VIX −0.294%, a
difference of +0.11pp (p = 0.57). Across VIX quintiles the recovery probability
ranges only from 44.2% to 51.5%, and **no quintile is significantly above 50%**.

## 9. Regression results

The pre-registered specification, n = 14,615:

| Term | Coef | Std. coef | SE (HC3) | 95% CI | p (HC3) | p (clustered) |
|---|---|---|---|---|---|---|
| Intercept | −0.0078 | — | 0.0044 | [−0.016, +0.001] | 0.080 | 0.185 |
| Crash z-score | −0.0002 | −0.005 | 0.0009 | [−0.002, +0.002] | 0.798 | 0.797 |
| Abnormal volume | −0.0022 | −0.015 | 0.0023 | [−0.007, +0.002] | 0.339 | 0.415 |
| Momentum 20d | −0.0214 | −0.019 | 0.0249 | [−0.070, +0.027] | 0.389 | 0.478 |
| **Realised vol 20d** | **+0.0630** | **+0.101** | 0.0163 | [+0.031, +0.095] | **0.0001** | **0.0011** |
| Market return (event day) | +0.1293 | +0.030 | 0.1166 | [−0.099, +0.358] | 0.268 | 0.402 |
| VIX | −0.0003 | −0.037 | 0.0002 | [−0.001, +0.000] | 0.087 | 0.254 |

Adjusted R² = **0.0085**.

One predictor survives: **20-day realised volatility**, positively signed and
significant under both HC3 and date-clustered errors, with the largest
standardised coefficient (0.101). Stocks that were already volatile before the
crash have marginally better subsequent abnormal returns. The effect is
economically modest — a one-standard-deviation increase in prior volatility
moves expected CAR₂₀ by about 0.85 percentage points — and it is as easily a
risk-compensation artefact as a behavioural one, since `CAR` is market-adjusted
but not risk-adjusted.

**H4 is not supported.** The `mom_20` coefficient is −0.021 with a
date-clustered p of 0.478.

The logistic regression for `recovered_20d` tells the same story
(pseudo-R² = 0.0011, base rate 47.1%). Realised volatility is the only
significant predictor, at an odds ratio of **1.082 per standard deviation**
(95% CI [1.014, 1.154], p = 0.017). Every other odds ratio is within a few
percent of one.

Two features were
dropped automatically from the extended specification as exact linear
identities (`abs_decline` is `−raw_return` because every event has a negative
return; `excess_vs_sector` is `raw_return − sector_ret_event`), which a
rank-revealing check reports rather than silently reshuffling.

One conditional result deserves a caveat. Adding crash-type dummies with
idiosyncratic as the omitted base gives a broad-market coefficient of **+0.96pp**
(HC3 p = 0.0017) — the opposite sign to the raw group means, because the
event-day market return is held fixed. Under date-clustered errors that becomes
p = 0.076, and by the study's declared rule the clustered result governs: this
is not evidence of a crash-type effect.

## 10. Multiple testing

The extended 16-predictor specification raises adjusted R² only to 0.0103 and
pseudo-R² to 0.0021. Pooling all 32 exploratory coefficients from the extended
OLS and logistic models into a single Benjamini–Hochberg family, **none survives
at q ≤ 0.05**.

This is the study's most important statistical guard, and it belongs in the body
rather than an appendix. Read without correction, the exploratory models offer
several coefficients at nominal p < 0.05 — a tempting set of "findings" about
what predicts recovery. But 32 tests against a near-null outcome are expected to
produce roughly that many false positives by construction. Reporting any of them
as a result would have been an artefact of the number of tests performed rather
than evidence about markets. The correction is what converts a suggestive-looking
table into an honest null.

The same logic governs H3. Its nominal p of 0.028 across the five pre-registered
contrasts becomes q = 0.14 after correction, and the appropriate description is
*suggestive but statistically insufficient* — not that abnormal volume predicts
recovery.

## 11. Predictive experiment

The classification task is `recovered_20d` from the 21 event-time features,
split chronologically with a 20-trading-day embargo: train through 2021-10-21
(n = 10,189), validate through 2024-06-24 (n = 2,160), test thereafter
(n = 2,199), with 99 events discarded to the embargo. Base rates are 47.4% /
46.8% / 46.9%. Hyper-parameters were chosen on validation only; the test block
was scored once.

| Model | Val AUC | **Test AUC** | PR-AUC | Brier | Log loss | Brier skill | Accuracy |
|---|---|---|---|---|---|---|---|
| Logistic | 0.502 | **0.504** | 0.477 | 0.2496 | 0.6924 | −0.002 | 0.524 |
| Random forest | 0.519 | **0.499** | 0.477 | 0.2499 | 0.6930 | −0.003 | 0.516 |
| LightGBM | 0.512 | **0.476** | 0.436 | 0.2564 | 0.7064 | −0.030 | 0.493 |
| *Base rate* | — | *0.500* | *0.469* | *0.2490* | *0.6912* | *0.000* | *0.531* |

**No model beats a constant forecast of the base rate.** Every Brier skill score
is negative — each model is fractionally *worse* than predicting the training
base rate for every event. The best test AUC, 0.504, is indistinguishable from
0.5. Expanding-window evaluation across five folds agrees: mean AUCs of 0.506
(logistic), 0.513 (random forest) and 0.505 (LightGBM), with per-fold values
ranging from 0.457 to 0.548 — a spread consistent with noise.

Note that the highest *accuracy* in the table belongs to the base-rate model,
which makes the same prediction every time. This is exactly why the research
plan demoted accuracy: with a base rate of 47%, predicting "no recovery"
always scores 53.1%, and any model that looks accurate is mostly reproducing
the class imbalance.

## 12. Calibration

Calibration is the one genuinely positive result. Expected calibration error is
0.028 for logistic regression, 0.034 for the random forest and 0.063 for
LightGBM — the models are honest about their own uncertainty. They are well
calibrated around 47% and carry no discriminating information, which is the
correct behaviour for a model fitted to an unpredictable target.

## 13. Interpretability

SHAP attributions on the LightGBM model rank `vix_chg`, `spy_mom_20`, `vix` and
`sector_ret_event` highest, with mean absolute values of 0.085, 0.052, 0.042
and 0.040 **log-odds**. A 0.085 log-odds shift moves a 47% probability to about
49%. Permutation importance ranks a different set — `avol`, `sector_ret_event`,
`excess_vs_sector` — with every effect under 0.006 AUC and error bars spanning
zero. The logistic coefficients rank a third set, led by `rv_60` and `mom_5`.

**Statistical and machine-learning interpretations do not agree**, which is the
expected signature of three methods ranking noise. The only point of contact is
that volatility-related features appear near the top of both the regression and
the SHAP ranking, consistent with the one surviving regression coefficient.

SHAP dependence plots for the four top features show no stable monotone
relationship with recovery odds; the binned medians wander without direction.

This warrants an explicit methodological caution. The SHAP rankings above were
extracted from a model with a test AUC of 0.476 — worse than chance. Attributions
from such a model describe what an uninformative fit happened to latch onto
in-sample; they are **not** substantive economics, and presenting them as
"what drives recovery" would be a category error. Disagreement among the three
interpretation methods, combined with chance-level out-of-sample performance, is
consistent with unstable feature ranking over noise rather than with real signal
that the models merely failed to exploit. An interpretability plot inherits the
credibility of the model beneath it, and here that credibility is nil.

## 14. Robustness

The grid crosses crash threshold (−2.5σ to −4.0σ) × volatility window (40, 60,
120) × cooldown (5, 10, 20, 40) × horizon (5, 10, 20, 60) × high-VIX definition
(0.60, 0.75, 0.90) — **576 specifications, every one persisted** to
`results/tables/robustness_grid.csv`.

At the 20-day horizon, across all 48 threshold × window × cooldown combinations:

* mean CAR is **negative in 48 of 48** (range −0.56% to −0.15%, median −0.32%);
* it is **significantly negative in 50%** and **significantly positive in 0%**;
* the recovery rate is **below 50% in 48 of 48** (range 46.3% to 48.2%).

There is a clean monotonicity: the more extreme the threshold, the more negative
the outcome. Taking the median across volatility windows and cooldowns, mean CAR₂₀ is −0.19% at −2.5σ, −0.26% at −3.0σ, −0.36% at −3.5σ and −0.49% at −4.0σ.
Severity does not buy a bounce; it predicts a slightly deeper continued decline.
This gradient is visible across the whole grid even though the crash z-score is
not a significant *within-sample* regressor — the threshold selects a different
population, whereas the regressor measures variation inside one.

The abnormal-volume effect holds its sign in **97.9%** of specifications. The
broad-versus-idiosyncratic contrast holds its sign in only **66.7%**, consistent
with a genuine null.

## 15. Limitations

1. **Coverage bias.** Only 35% of historical-only index members have usable
   price data. This plausibly biases mean CAR upward, which works against the
   headline finding rather than producing it, but it does mean the point
   estimate should be read as a mild overstatement of post-crash performance.
2. **No delisting returns.** A stock removed mid-window stops contributing
   rather than realising a terminal value, again under-counting bad outcomes.
3. **Market-adjusted, not risk-adjusted.** `CAR` subtracts SPY, not a factor
   model. The surviving realised-volatility coefficient is exactly the kind of
   result a beta or volatility factor might absorb.
4. **Close-to-close only.** A stock that fell 30% intraday and closed flat is
   not an event here.
5. **Statistical significance is not economic significance.** A −0.27% mean
   with an 8.4% standard deviation is a statistically detectable displacement
   of an almost symmetric distribution, nothing more. Nothing here supports a
   trading conclusion, and no costs are modelled.
6. **Membership provenance.** Wikipedia is crowd-edited, not S&P's official
   record; 39 change-log rows could not be reconciled, and the record is
   unreliable before 2007.
7. **Restated adjusted prices.** Yahoo restates history after corporate actions,
   so exact reproduction requires the same download date.
8. **One market, one regime.** Large-cap US equity, 2007–2026 — a period
   dominated by a secular bull market and two sharp crises. Nothing here
   extrapolates to small caps, other markets, or other asset classes.
9. **A near-null is not proof of no effect.** Absence of predictability in 21
   price-and-volume features does not mean none exists. Fundamentals, news text,
   options-implied data, order flow and short interest are all unexamined.

## 16. Conclusion

The evidence separates into three claims that should not be collapsed into one
another.

**On the universal effect: there is no average dead-cat bounce.** Across 14,678
extreme single-day declines in S&P 500 constituents over nearly twenty years,
mean 20-day abnormal return against SPY is **−0.27%** (95% CI [−0.40%, −0.12%])
and only **47.1%** of events end above the benchmark. The displacement is small
but stubborn: it holds its sign at every horizon from one day to three months,
and in all 48 primary robustness specifications. Severity deepens it rather than
reversing it. Whatever the folklore describes, the sample does not contain a
systematic rebound.

**On heterogeneity: individual outcomes are extremely dispersed.** The −0.27%
mean sits inside a distribution with a standard deviation of **8.4%** — roughly
thirty times the central tendency — with mean adverse excursion of −5.9% against
mean favourable excursion of +5.7%. Almost every event is dramatically better or
worse than the average. The average is a real and measurable property of the
distribution's centre, and it is nearly irrelevant to any particular crash.
Heterogeneity, not the mean, is where the economic content lives.

**On predictability: the studied features do not sort those outcomes.** Neither
the operational type of the crash, nor abnormal volume, nor pre-crash momentum,
nor the prevailing volatility regime separates recovering events from continuing
decliners at 20 days once multiple comparisons are accounted for. None of 32
exploratory coefficients survives FDR correction. Three model families spanning
linear and non-linear function classes fail to beat a constant base-rate forecast
on embargoed hold-out data, with negative Brier skill throughout, and their
feature attributions disagree with one another — the signature of methods ranking
noise. The conclusion is specific to what was measured: **the available
event-time price-and-volume features do not contain stable out-of-sample
information for distinguishing recovery from continued underperformance.**

Two weak regularities survive. Stocks that were already volatile before the
crash do marginally better afterwards — roughly 0.85 percentage points of CAR₂₀
per standard deviation of prior volatility — though this is as plausibly
risk compensation as behaviour, given that `CAR` is market-adjusted and not
risk-adjusted. And abnormal volume points in a consistent direction, with
high-volume crashes underperforming less, in 97.9% of specifications while never
reaching significance in any single one. Both are reported as leads, not results.

The practical implication is about description. A severe one-day decline is
neither an entry signal nor a warning; it is an invitation to a very wide
distribution whose centre sits a fraction of a percent below zero. The 8.4%
standard deviation around a −0.27% mean *is* the finding. Folklore in both
directions — that crashes bounce, that crashes cascade — reads structure into
what is, conditional on everything observable in this design, close to noise.

These conclusions hold within the studied design: large-cap US equity,
2007–2026, close-to-close events, and a price-and-volume feature set. They do not
establish that no predictable structure exists, only that none was found where
this study looked.

---
