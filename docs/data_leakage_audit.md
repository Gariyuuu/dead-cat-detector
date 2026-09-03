# Data-leakage audit

Every way this study could accidentally see the future, and what stops it.
Each control names the test that enforces it; `make test` runs all 43.

---

## 1. The event definition must not see the event

**Risk.** If μ and σ are computed over a window that includes day *t*, the
z-score is contaminated by the very return it is supposed to score, and extreme
days become mechanically less extreme.

**Control.** `events.rolling_moments` computes a rolling mean and standard
deviation and then applies a trailing `.shift(1)`, so the window ending at
*t−1* never contains $r_t$.

**Tests.**
* `test_rolling_moments_exclude_current_and_future` — μ and σ at *t* equal the
  hand-computed statistics of `ret[t-60:t]`.
* `test_rolling_moments_invariant_to_future_mutation` — multiplying every
  return from *t* onward by 100 and adding 7 leaves μ and σ at *t* unchanged.
  This is the strong form: it would fail for any window that peeked forward.
* `test_zscore_uses_prior_window_only`
* `test_min_prior_obs_enforced` — no z-score exists before 60 prior returns.

## 2. Forward returns must be indexed correctly

**Risk.** An off-by-one in the forward index silently shifts every outcome by a
day; clipping at the end of the calendar silently invents outcomes that have
not happened yet.

**Control.** `events.forward_outcomes` gathers $P_{t+h}$ by integer position
and writes `NaN` wherever `t+h` exceeds the calendar, with no clipping or
wrap-around.

**Tests.**
* `test_forward_return_indexing_is_exact` — for h ∈ {1,5,10,20,60},
  `fwd_ret_h` equals `P[t+h]/P[t] - 1` exactly.
* `test_forward_outcome_is_nan_past_the_calendar_edge` — an event five days
  from the end has a valid 1-day outcome and a `NaN` 20-day outcome.
* `test_mae_mfe_bracket_the_path`
* `test_car_path_starts_at_zero_and_matches_car_20`

## 3. Benchmark alignment

**Risk.** A benchmark misaligned by one day produces a spurious abnormal
return that looks like a real effect.

**Control.** The benchmark is reindexed onto the equity calendar and gathered
at the same integer positions as the stock.

**Tests.**
* `test_car_equals_stock_minus_benchmark`
* `test_benchmark_alignment_is_date_matched` — shifting the benchmark by one
  day *must* change CAR. A test that only checks equality would pass even if
  both legs were misaligned together; this one proves alignment binds.

## 4. Features must be observable at the event close

**Risk.** A momentum window that ends at *t* instead of *t−1* embeds the crash
return itself, so "pre-crash momentum" partly *is* the crash.

**Control.** All momentum windows run from *t−1* backwards; realised
volatility is shifted by one day. Same-day values (crash return, event-day
volume, event-day market/sector return, event-day VIX) are retained because
they are genuinely observable at the close.

**Test.** `test_momentum_features_exclude_the_crash_day` — with a −20% crash
engineered at *t*, `mom_20` equals `close[t-1]/close[t-21] - 1` and is not
dragged below −15%.

## 5. Outcomes must not enter the predictor matrix

**Risk.** The most common catastrophic leak: an outcome column reaching the
model.

**Control.** `features.FEATURE_COLUMNS` is an explicit allow-list.
`events.OUTCOME_PREFIXES` (`fwd_`, `car_`, `mfe_`, `mae_`, `recovered_`,
`regained_`, `days_to_`) and `events.DESCRIPTIVE_FORWARD` (`episode_length`,
`episode_min_return`, `episode_end_date`) name everything forbidden.

`episode_length` and `episode_min_return` deserve emphasis: they are computed
over the cooldown window *after* the event and so are forward-looking. They are
stored because they describe the episode, and they are barred from the model.

**Controls in two places.**
* `test_feature_matrix_excludes_outcomes_and_forward_descriptors` — no feature name starts
  with an outcome prefix or appears in the descriptive-forward list.
* `scripts/05_models.py` asserts the same at runtime, so the guarantee holds
  for the matrix actually fitted, not merely for the constant.

## 6. Universe selection must not look ahead

**Risk.** Using today's S&P 500 list over a 20-year history means every "event"
belongs to a company that survived to 2026 — a textbook survivorship leak.

**Control.** Point-in-time membership: a stock is eligible only on dates it was
actually an index member (`data.build_membership_windows`,
`data.membership_mask`).

**Tests.**
* `test_membership_rollback_reconstructs_windows` — a name added in 2012 and
  removed in 2018 gets exactly that window.
* `test_membership_mask_is_half_open` — `end` is exclusive.
* `test_membership_gating_removes_ineligible_events`

**Residual, not solved:** 65% of historical-only names have no price data at
Yahoo. See `data/README.md`. This is a coverage gap, not a leak — it biases the
mean CAR *upward*, against the direction of the headline finding.

## 7. Ticker reuse

**Risk.** Yahoo serving a *different, later* company under a recycled symbol
inserts post-hoc data into a historical slot — a leak in the most literal
sense. `SBNY` returns a listing starting seventeen months after Signature Bank
failed.

**Control.** `data.apply_reuse_guard` rejects price history that does not
overlap its symbol's membership window. 43 symbols dropped.

**Tests.** `test_reuse_guard_drops_non_overlapping_history`,
`test_reuse_guard_keeps_overlapping_history`.

## 8. Event independence

**Risk.** Without a cooldown, one multi-day collapse contributes a dozen
correlated "independent" observations and inflates apparent significance.

**Control.** Greedy ticker-specific cooldown, default 20 trading days.

**Tests.** `test_cooldown_suppresses_near_events`,
`test_cooldown_is_ticker_specific`,
`test_cooldown_minimum_gap_holds_for_every_pair`, `test_event_uniqueness`.
Verified on the real event table: the minimum trading-day gap between two
events on the same ticker is exactly 20, with zero violations, and there are no
duplicate `event_id` or `(ticker, event_date)` pairs.

## 9. Train/test contamination

**Risk 1 — ordering.** Random splits let the model train on the future.
**Control.** `models.chronological_split` orders strictly by event date.

**Risk 2 — horizon overlap.** Even a correct time split leaks: the last
training event's 20-day outcome window extends into the validation period, so
its label is partly determined by prices the validation features already see.
**Control.** A 20-trading-day **embargo** between blocks. In the primary run
this drops 99 events (train 10,189 / val 2,160 / test 2,199).

**Risk 3 — tuning on the test set.** **Control.** Hyper-parameters are chosen
on validation only; the test block is scored once, after refitting on
train+validation.

**Tests.** `test_split_is_chronological_and_disjoint`,
`test_embargo_creates_a_real_gap`, `test_split_fractions_are_approximately_respected`,
`test_expanding_folds_never_train_on_the_future`, `test_expanding_folds_grow`.

## 10. Preprocessing leakage

**Risk.** Fitting an imputer or scaler on the full sample leaks test-set
distribution into training.

**Control.** Imputation and scaling live inside a scikit-learn `Pipeline`, so
they are fit on training folds only and applied to test.

## 11. Determinism

**Risk.** Results that move between runs cannot be audited.

**Control.** A single seed (`20260902`) flows from config into the bootstrap,
the splitters, all three model builders and the SHAP sampler.

**Tests.** `test_model_seeds_are_deterministic` (all three model families),
`test_bootstrap_is_seed_deterministic`.

## 12. Multiple comparisons

**Risk.** Testing enough conditional cuts guarantees a "finding".

**Control.** Benjamini-Hochberg across the exploratory family, plus a 576-cell
robustness grid in which every specification is persisted. Of 32 exploratory
coefficients, **none** survive FDR correction — reported as such.

**Tests.** `test_benjamini_hochberg_is_monotone_and_conservative`,
`test_benjamini_hochberg_controls_a_pure_null`.

---

## Known residual risks

1. **Delisting/coverage bias.** 35% coverage of historical-only names. Biases
   mean CAR upward, i.e. against the headline result.
2. **No delisting returns.** A removed stock stops contributing rather than
   realising a terminal value, again under-counting bad outcomes.
3. **Restated adjusted prices.** Yahoo restates history after corporate
   actions, so exact reproduction requires the same download date.
4. **Wikipedia membership.** Crowd-edited; 39 change-log rows could not be
   reconciled.
5. **Close-to-close only.** No intraday path; a 30% intraday drop that closes
   flat is not an event.
6. **Successor-listing contamination.** A few acquired names (e.g. HAR) may
   carry a successor or foreign listing's history. These pass both screens
   because they are internally consistent.
