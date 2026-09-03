# Research plan (pre-registration)

**Question.** After a liquid U.S. equity experiences an unusually severe
one-day decline, which observable characteristics distinguish subsequent
20-trading-day mean reversion from continued underperformance?

This document fixes the hypotheses, the event definition, the outcome, and the
analysis plan **before** estimation. Everything not listed here as primary is
exploratory and is reported as such, with a Benjamini-Hochberg correction
applied across the exploratory family.

---

## 1. Hypotheses

| | Hypothesis | Primary test | Direction |
|---|---|---|---|
| **H1** | Extreme declines do not universally mean-revert | Is mean `CAR_20` different from zero? | Two-sided |
| **H2** | Broad-market crashes recover differently from idiosyncratic crashes | Difference in mean `CAR_20` between crash types | Two-sided |
| **H3** | Abnormal volume carries information | Difference in mean `CAR_20`, highest vs lowest AVOL quartile | Two-sided |
| **H4** | Pre-crash momentum matters | Coefficient on `mom_20` in the primary regression | Two-sided |
| **H5** | Market volatility changes the relationship | Difference in mean `CAR_20`, high vs low VIX | Two-sided |

None of these is a directional bet. The interesting outcome is as likely to be
a null as an effect, and a null is reported as a null.

## 2. Event definition

Daily return

$$r_{i,t} = P_{i,t}/P_{i,t-1} - 1$$

with mean and volatility estimated over the 60 trading days **strictly
preceding** *t*:

$$z_{i,t} = \frac{r_{i,t} - \mu^{60}_{i,t-1}}{\sigma^{60}_{i,t-1}}$$

A **crash event** is $z_{i,t} \le -3.0$, requiring at least 60 prior
observations and requiring the stock to be an index member on *t*.

**Sensitivity:** thresholds −2.5σ, −3.0σ, −3.5σ, −4.0σ; volatility windows 40,
60, 120.

## 3. Episodes and independence

Consecutive breaches are not independent observations. A **ticker-specific
cooldown** of 20 trading days is applied greedily: the first breach in an
episode is kept, and later breaches within the cooldown are absorbed into that
episode. Each retained event stores `episode_length` and `episode_min_return`.

**Sensitivity:** cooldowns of 5, 10, 20, 40 trading days.

`episode_length`, `episode_min_return` and `episode_end_date` describe what
happened *after* the event and are **descriptive only** — they are excluded
from the predictor matrix by an explicit test.

## 4. Outcomes

Forward cumulative returns at horizons 1, 5, 10, 20, 60, measured **from the
event close**, so the crash-day move is not part of the outcome.

$$CAR_{i,t,h} = R_{i,t:t+h} - R_{SPY,t:t+h}$$

* **Primary outcome:** `CAR_20`
* **Primary classification target:** `recovered_20d = 1[CAR_20 > 0]`
* **Secondary outcome:** did the close regain its pre-crash (*t−1*) level within
  20 sessions?
* Also recorded: maximum favourable excursion, maximum adverse excursion, days
  until recovery.

## 5. Features

Every feature must be computable at the **close of the event day**. Trailing
windows end at *t−1*; same-day quantities (the crash return, event-day volume,
event-day market and sector returns, event-day VIX) are observable at the close.

* **Crash:** z-score, raw return, absolute decline, high-low range
* **Volume:** volume, log volume, 60-day median volume, and
  $AVOL = \log V_t - \log \mathrm{median}(V_{t-60:t-1})$
* **Momentum:** prior 5, 20, 60, 252-day returns, all ending at *t−1*
* **Volatility:** 20-day and 60-day realised volatility, and their ratio
* **Market:** SPY event-day return, SPY 20-day momentum, VIX, VIX daily change
* **Sector:** sector-ETF event-day return, stock-minus-sector return

## 6. Crash-type classification

Transparent, operational, and **not a claim about cause**:

* **Broad-market shock** — SPY's event-day return is at or below the 5th
  percentile of its own history, computed on an expanding window using strictly
  prior data.
* **Sector shock** — the sector ETF is below its 5th percentile and the
  broad-market condition does not hold.
* **Idiosyncratic** — the stock breaches its threshold and neither benchmark
  condition holds.
* **Unclassified** — sector is unknown, so the sector condition can be neither
  confirmed nor ruled out.

These labels describe *co-movement*, not causation. A stock can be labelled
idiosyncratic and still have fallen for a macro reason.

## 7. Analysis plan

**Event study.** Mean CAR through event time τ = 0…60, with 95% confidence
bands from 2,000 bootstrap resamples of complete events (resampling events, not
observations, to preserve within-path autocorrelation). Groups compared: all
events; broad-market / sector / idiosyncratic; highest and lowest abnormal-volume
quartile; high and low VIX.

**Regression.** The pre-registered specification is

$$CAR_{20} = \beta_0 + \beta_1 z + \beta_2 AVOL + \beta_3 mom_{20} + \beta_4 RV_{20} + \beta_5 r_{SPY} + \beta_6 VIX + \varepsilon$$

estimated by OLS with **HC3** robust standard errors, reporting coefficient,
standardised coefficient, 95% CI, p-value, n and adjusted R².

*Addition to the plan, stated in advance of interpretation:* crash events
cluster heavily in calendar time — hundreds of names can breach the threshold
on the same market-wide day — so HC3 alone overstates precision.
**Date-clustered standard errors are reported alongside HC3 throughout**, and
where the two disagree the clustered result governs the conclusion.

Then a logistic regression for `recovered_20d`, reporting odds ratios per
one-standard-deviation move.

**Multiplicity.** All exploratory coefficients enter a single
Benjamini-Hochberg family at α = 0.05.

## 8. Predictive experiment

Chronological splits only: earliest 70% train, next 15% validation, latest 15%
test, with a **20-trading-day embargo** between blocks so that no training
label depends on prices observed inside the next block. Expanding-window
evaluation over five folds as a secondary check.

Models: logistic regression, random forest, LightGBM. Hyper-parameters are
chosen on the validation block only; the test block is evaluated once. A
constant base-rate forecast is carried as the reference every model must beat.

Metrics: ROC-AUC, PR-AUC, Brier score, log loss, calibration. **Accuracy is
secondary** and is reported last, because with a base rate near 50% it is
almost uninformative.

## 9. Robustness

The grid crosses crash threshold × volatility window × cooldown × outcome
horizon × high-VIX definition — 576 specifications. **Every specification is
persisted**, and the robustness figure is built from the complete grid, not a
favourable subset.

## 10. What would falsify each hypothesis

* **H1** is not supported if the 95% CI for mean `CAR_20` contains zero, or if
  the sign is unstable across the robustness grid.
* **H2/H3/H5** are not supported if the bootstrap CI for the group difference
  contains zero after FDR correction.
* **H4** is not supported if the `mom_20` coefficient is indistinguishable from
  zero under date-clustered standard errors.
* The predictive experiment fails if no model beats a constant base-rate
  forecast on the held-out block.

## 11. Declared analytical choices

Stated so they are visible rather than buried:

1. Study starts in 2007 because index-membership records are unreliable before
   then.
2. Whole tickers, not individual days, are dropped by the data-quality screens.
3. `SPY` is the sole benchmark; no factor model is estimated. `CAR` is therefore
   a market-adjusted return, not a risk-adjusted alpha.
4. Outcomes are measured from the event **close**, which is the earliest point
   at which the event is observable.
5. No transaction costs, borrow costs, or liquidity constraints are modelled.
   This is a study of conditional return behaviour, not of a strategy.
