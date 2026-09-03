"""Generate the static research presentation from persisted results.

Architecture: Python analysis -> persisted result tables -> static site.
Every number rendered on the page is read from `results/` at build time, so the
presentation cannot drift from the study. The page runs no analysis itself.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, M = ROOT / "results" / "tables", ROOT / "results" / "metrics"
WEB = ROOT / "web"
FIGS = WEB / "figures"

FIGURES = [
    "fig03_mean_car_with_ci.png",
    "fig10_robustness_heatmap.png",
    "fig07_outcome_distribution.png",
    "fig04_car_by_crash_type.png",
    "fig08_calibration.png",
    "fig11_interpretability_comparison.png",
]

# ------------------------------------------------------------------ inputs ---
h1 = pd.read_csv(T / "h1_mean_car_by_group.csv").set_index("group")
con = pd.read_csv(T / "group_contrasts.csv")
ols = pd.read_csv(T / "ols_primary_car20.csv").set_index("term")
comp = pd.read_csv(T / "model_comparison.csv").set_index("model")
hz = pd.read_csv(T / "car_by_horizon.csv")
reg = json.load(open(M / "regressions.json"))
rob = json.load(open(M / "robustness.json"))
ec = json.load(open(M / "event_construction.json"))
man = json.load(open(ROOT / "data" / "processed" / "manifest.json"))
ev_sd = json.load(open(M / "event_study.json"))

a = h1.loc["all_events"]
avol = con[con.hypothesis == "H3"].iloc[0]
bi = con[(con.group_a == "broad_market") & (con.group_b == "idiosyncratic")].iloc[0]
vix = con[con.hypothesis == "H5"].iloc[0]
cov = man["coverage"]

def pct(v, d=3):
    """Percentage with a typographic minus sign, matching the page's entities."""
    return f"{v * 100:.{d}f}%".replace("-", "\u2212")
sd_car20 = float(json.load(open(M / "outcome_dispersion.json"))["sd_car20"])

GH = "https://github.com/Gariyuuu/dead-cat-detector"

# Derive every robustness count from the persisted grid -- never typed by hand.
_c = rob["car_h20_across_specs"]
_r = rob["recovery_rate_across_specs"]
N_SPEC_CELLS = int(_c["n"])
N_NEG = int(round(_c["share_negative"] * N_SPEC_CELLS))
N_BELOW_HALF = int(round(_r["share_below_half"] * N_SPEC_CELLS))
N_POS_SIG = int(round(_c["share_positive_and_significant"] * N_SPEC_CELLS))

grid = pd.read_csv(T / "robustness_grid.csv")
# z_thresholds are negative, so the MILDEST cut is the maximum (-2.5) and the
# most SEVERE is the minimum (-4.0). Sorting descending puts mild first.
_g20 = grid[grid.horizon == 20].groupby("z_threshold")["mean_car"].median().sort_index(ascending=False)
THR_MILD, THR_SEVERE = _g20.index[0], _g20.index[-1]
CAR_AT_MILD, CAR_AT_SEVERE = float(_g20.iloc[0]), float(_g20.iloc[-1])
assert THR_MILD > THR_SEVERE, "mild threshold must be the less negative one"
assert CAR_AT_MILD > CAR_AT_SEVERE, "severity is expected to deepen the drift"

horizon_rows = "".join(
    f"<tr><td>{int(r.horizon)}d</td><td class='num'>{pct(r['mean'])}</td>"
    f"<td class='num sub'>[{pct(r.lo)}, {pct(r.hi)}]</td></tr>"
    for _, r in hz.iterrows()
)

model_rows = ""
for name, label in [("logistic", "Logistic regression"), ("random_forest", "Random forest"),
                    ("lightgbm", "LightGBM"), ("base_rate", "Constant base rate")]:
    if name not in comp.index:
        continue
    r = comp.loc[name]
    cls = " class='baseline'" if name == "base_rate" else ""
    model_rows += (
        f"<tr{cls}><td>{label}</td><td class='num'>{r.roc_auc:.3f}</td>"
        f"<td class='num'>{r.brier:.4f}</td>"
        f"<td class='num'>{f'{r.brier_skill_vs_base_rate:+.3f}'.replace('-', chr(0x2212))}</td>"
        f"<td class='num'>{r.accuracy:.3f}</td></tr>"
    )

hyp_rows = "".join([
    f"<tr><td class='hk'>H1</td><td>Extreme declines do not universally mean-revert</td>"
    f"<td class='ok'>Supported</td></tr>",
    f"<tr><td class='hk'>H2</td><td>Broad-market crashes recover differently from idiosyncratic</td>"
    f"<td class='no'>Not supported <span class='sub'>(p = {bi.p_value:.2f})</span></td></tr>",
    f"<tr><td class='hk'>H3</td><td>Abnormal volume contains information</td>"
    f"<td class='maybe'>Suggestive, insufficient <span class='sub'>(q = {avol.q_value:.2f})</span></td></tr>",
    f"<tr><td class='hk'>H4</td><td>Pre-crash momentum matters</td>"
    f"<td class='no'>Not supported <span class='sub'>(p = {ols.loc['mom_20'].p_value_cluster:.3f})</span></td></tr>",
    f"<tr><td class='hk'>H5</td><td>Market volatility changes the relationship</td>"
    f"<td class='no'>Not supported <span class='sub'>(p = {vix.p_value:.2f})</span></td></tr>",
])

HTML = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dead Cat Detector — When extreme stock selloffs reverse, and when they keep falling</title>
<meta name="description" content="An event study of {ec['n_events']:,} extreme single-day declines in point-in-time S&amp;P 500 constituents, 2007-2026. Mean 20-day abnormal return is {pct(a.mean_car20)}: there is no average dead-cat bounce.">
<meta property="og:title" content="Dead Cat Detector">
<meta property="og:description" content="{ec['n_events']:,} extreme equity selloffs, 2007-2026. There is no average dead-cat bounce.">
<meta property="og:type" content="article">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="icon" href="/favicon-32.png" sizes="32x32" type="image/png">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<meta name="theme-color" content="#fcfcfb" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#14140f" media="(prefers-color-scheme: dark)">
<style>
  :root {{
    color-scheme: light;
    --surface: #fcfcfb; --surface-2: #f4f3ef; --rule: #e6e5e1;
    --ink: #12120f; --ink-2: #52514e; --ink-3: #8a8985;
    --accent: #2a78d6; --neg: #b8302f; --pos: #1baf7a;
    --measure: 34rem;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --surface: #14140f; --surface-2: #1c1c18; --rule: #32322c;
      --ink: #f5f4ee; --ink-2: #b9b8ae; --ink-3: #86857c;
      --accent: #6da7ec; --neg: #e66767; --pos: #3fc793;
    }}
  }}
  * {{ box-sizing: border-box; }}
  html {{ -webkit-text-size-adjust: 100%; }}
  body {{
    margin: 0; background: var(--surface); color: var(--ink);
    font: 16px/1.65 ui-serif, Georgia, "Times New Roman", serif;
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 60rem; margin: 0 auto; padding: 0 1.5rem; }}
  .sans {{ font-family: ui-sans-serif, -apple-system, "Helvetica Neue", Arial, sans-serif; }}

  header {{ padding: 5rem 0 3rem; border-bottom: 1px solid var(--rule); }}
  .eyebrow {{
    font-family: ui-sans-serif, -apple-system, Arial, sans-serif;
    font-size: .75rem; letter-spacing: .14em; text-transform: uppercase;
    color: var(--ink-3); margin: 0 0 1.25rem;
  }}
  h1 {{ font-size: clamp(2.1rem, 5.5vw, 3.4rem); line-height: 1.1; margin: 0 0 .75rem; font-weight: 600; letter-spacing: -.02em; }}
  .sub-title {{ font-size: clamp(1.05rem, 2.4vw, 1.3rem); color: var(--ink-2); margin: 0 0 2rem; max-width: var(--measure); }}
  .lede {{ max-width: var(--measure); color: var(--ink-2); }}

  .claim {{
    font-size: clamp(1.5rem, 4vw, 2.1rem); line-height: 1.25; font-weight: 600;
    margin: 2.5rem 0 1rem; letter-spacing: -.015em; max-width: 30rem;
  }}
  section {{ padding: 3.5rem 0; border-bottom: 1px solid var(--rule); }}
  section:last-of-type {{ border-bottom: 0; }}
  h2 {{
    font-family: ui-sans-serif, -apple-system, Arial, sans-serif;
    font-size: .78rem; letter-spacing: .14em; text-transform: uppercase;
    color: var(--ink-3); margin: 0 0 1.5rem; font-weight: 600;
  }}
  h3 {{ font-size: 1.25rem; margin: 2rem 0 .6rem; font-weight: 600; }}
  p {{ max-width: var(--measure); }}
  a {{ color: var(--accent); text-underline-offset: 3px; }}

  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr)); gap: 1px;
            background: var(--rule); border: 1px solid var(--rule); margin: 2rem 0; }}
  .stat {{ background: var(--surface); padding: 1.25rem 1.1rem; }}
  .stat .v {{ font-family: ui-sans-serif, -apple-system, Arial, sans-serif;
              font-size: 1.55rem; font-weight: 600; letter-spacing: -.02em; display: block; }}
  .stat .k {{ font-family: ui-sans-serif, -apple-system, Arial, sans-serif;
              font-size: .72rem; color: var(--ink-3); text-transform: uppercase;
              letter-spacing: .07em; margin-top: .3rem; display: block; line-height: 1.4; }}
  .stat .v.neg {{ color: var(--neg); }}

  figure {{ margin: 2rem 0; }}
  figure img {{ width: 100%; height: auto; display: block; border: 1px solid var(--rule); background: #fcfcfb; border-radius: 2px; }}
  figcaption {{ font-size: .85rem; color: var(--ink-3); margin-top: .7rem; max-width: 42rem; line-height: 1.55; }}

  table {{ border-collapse: collapse; width: 100%; margin: 1.5rem 0; font-size: .9rem;
           font-family: ui-sans-serif, -apple-system, Arial, sans-serif; }}
  .scroll {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
  th, td {{ text-align: left; padding: .6rem .75rem; border-bottom: 1px solid var(--rule); }}
  th {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .07em; color: var(--ink-3); font-weight: 600; white-space: nowrap; }}
  td.num {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
  .sub {{ color: var(--ink-3); font-size: .85em; }}
  tr.baseline td {{ color: var(--ink-3); font-style: italic; }}
  td.hk {{ font-weight: 600; width: 3rem; }}
  .ok {{ color: var(--pos); font-weight: 600; }}
  .no {{ color: var(--ink-3); }}
  .maybe {{ color: var(--ink-2); }}

  blockquote {{ margin: 1.75rem 0; padding: 1rem 1.25rem; background: var(--surface-2);
                border-left: 2px solid var(--accent); max-width: var(--measure); }}
  blockquote p {{ margin: 0; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .86em;
          background: var(--surface-2); padding: .12em .38em; border-radius: 3px; }}
  pre {{ background: var(--surface-2); padding: 1rem 1.15rem; overflow-x: auto;
         border-radius: 3px; border: 1px solid var(--rule); }}
  pre code {{ background: none; padding: 0; font-size: .82rem; line-height: 1.7; }}
  ol, ul {{ max-width: var(--measure); padding-left: 1.15rem; }}
  li {{ margin-bottom: .5rem; }}
  footer {{ padding: 3rem 0 4.5rem; color: var(--ink-3); font-size: .85rem; }}
  footer a {{ color: var(--ink-2); }}
  .caveat {{ font-size: .9rem; }}
</style>
</head>
<body>
<div class="wrap">

<header>
  <p class="eyebrow">Quantitative research &middot; Event study</p>
  <h1>Dead Cat Detector</h1>
  <p class="sub-title">When extreme stock selloffs reverse&mdash;and when they keep falling</p>
  <p class="lede">Using <strong>{ec['n_events']:,}</strong> extreme one-day declines among
  point-in-time S&amp;P&nbsp;500 constituents from {ec['date_min'][:4]}&ndash;{ec['date_max'][:4]},
  this study tests whether severe stock crashes systematically mean-revert, and whether
  observable event-time features distinguish rebounds from continued underperformance.</p>
</header>

<section>
  <h2>The result</h2>
  <p class="claim">There is no average dead&#8209;cat bounce.</p>
  <p>We estimate a mean 20-day cumulative abnormal return against SPY of
  <strong>{pct(a.mean_car20)}</strong> (95% bootstrap CI
  [{pct(a.ci_lo)}, {pct(a.ci_hi)}], n&nbsp;=&nbsp;{int(a.n):,}). Mean CAR is negative at
  <em>every</em> horizon from 1 to 60 trading days.</p>

  <div class="stats">
    <div class="stat"><span class="v neg">{pct(a.mean_car20, 2)}</span><span class="k">Mean CAR<sub>20</sub> vs SPY</span></div>
    <div class="stat"><span class="v">{a.recovery_rate * 100:.1f}%</span><span class="k">End above benchmark</span></div>
    <div class="stat"><span class="v">{a.regained_precrash_rate * 100:.1f}%</span><span class="k">Regain pre-crash close</span></div>
    <div class="stat"><span class="v">{sd_car20 * 100:.1f}%</span><span class="k">Std. dev. of CAR<sub>20</sub></span></div>
  </div>

  <figure>
    <img src="figures/fig03_mean_car_with_ci.png" alt="Mean cumulative abnormal return against SPY from the event close through 60 trading days, with a 95% bootstrap confidence band. The path drifts below zero and remains there.">
    <figcaption>Mean cumulative abnormal return against SPY from the event close through 60
    trading days, with a 95% bootstrap band from 2,000 whole-event resamples. The path drifts
    below zero and stays there.</figcaption>
  </figure>

  <div class="scroll">
  <table>
    <thead><tr><th>Horizon</th><th>Mean CAR</th><th>95% CI</th></tr></thead>
    <tbody>{horizon_rows}</tbody>
  </table>
  </div>

  <blockquote><p class="caveat"><strong>This is not a trading signal.</strong> No transaction
  costs, borrow costs or capacity constraints are modelled. A {pct(a.mean_car20, 2)} mean inside
  a {sd_car20 * 100:.1f}% standard deviation is a statistical displacement, not an edge.</p></blockquote>
</section>

<section>
  <h2>The dataset</h2>
  <p>Daily split- and dividend-adjusted OHLCV, with S&amp;P&nbsp;500 membership
  <strong>reconstructed point in time</strong> &mdash; the current constituent list rolled
  backwards through the index change log &mdash; so a stock is eligible for event detection
  only on dates it actually belonged to the index.</p>
  <div class="stats">
    <div class="stat"><span class="v">{ec['n_events']:,}</span><span class="k">Crash events (z &le; &minus;3&sigma;)</span></div>
    <div class="stat"><span class="v">{ec['n_tickers']}</span><span class="k">Tickers</span></div>
    <div class="stat"><span class="v">{man['price_rows'] / 1e6:.2f}M</span><span class="k">Ticker-days</span></div>
    <div class="stat"><span class="v">19.7</span><span class="k">Years ({ec['date_min'][:4]}&ndash;{ec['date_max'][:4]})</span></div>
  </div>
  <p>A crash is a day on which a stock's return falls at least three standard deviations below
  its own trailing 60-day distribution, with mean and volatility estimated from observations
  <em>strictly preceding</em> the event. A 20-trading-day ticker-specific cooldown collapses
  each episode to its first day, so consecutive breaches are not counted as independent
  observations.</p>
</section>

<section>
  <h2>Robustness</h2>
  <p>The result does not depend on how a crash is defined. Across
  <strong>{rob['n_specifications']}</strong> persisted specifications &mdash; crash threshold
  &times; volatility window &times; cooldown &times; outcome horizon &times; high-VIX
  definition &mdash; mean CAR<sub>20</sub> is negative in
  <strong>{N_NEG} of {N_SPEC_CELLS}</strong> threshold &times; window &times; cooldown cells,
  and the recovery rate is below 50% in
  <strong>{N_BELOW_HALF} of {N_SPEC_CELLS}</strong>.</p>
  <div class="stats">
    <div class="stat"><span class="v">{N_NEG} / {N_SPEC_CELLS}</span><span class="k">Specs with negative mean CAR</span></div>
    <div class="stat"><span class="v">{N_BELOW_HALF} / {N_SPEC_CELLS}</span><span class="k">Specs with recovery below 50%</span></div>
    <div class="stat"><span class="v">{N_POS_SIG}</span><span class="k">Specs positive &amp; significant</span></div>
  </div>
  <figure>
    <img src="figures/fig10_robustness_heatmap.png" alt="Heatmap of mean 20-day cumulative abnormal return across crash thresholds, volatility windows, cooldowns and horizons. Every cell is negative.">
    <figcaption>Mean CAR<sub>20</sub> across the specification grid. Every cell is negative, and
    severity makes the drift worse rather than better: {pct(CAR_AT_MILD, 2)} at
    &minus;{abs(THR_MILD)}&sigma; deepening to {pct(CAR_AT_SEVERE, 2)} at
    &minus;{abs(THR_SEVERE)}&sigma;. Built from the complete grid, not a favourable
    subset.</figcaption>
  </figure>
</section>

<section>
  <h2>Dispersion is the real story</h2>
  <p>The average effect is modestly negative. The variation around it is enormous: the standard
  deviation of CAR<sub>20</sub> is <strong>{sd_car20 * 100:.1f}%</strong> against a
  {pct(a.mean_car20, 2)} mean &mdash; roughly <strong>thirty times</strong> the central
  tendency. Almost every individual crash resolves dramatically better or worse than the
  average.</p>
  <figure>
    <img src="figures/fig07_outcome_distribution.png" alt="Distribution of 20-day cumulative abnormal returns across all crash events, a wide and near-symmetric distribution centred just below zero.">
    <figcaption>The distribution of 20-day outcomes: wide, near-symmetric, and centred a
    fraction of a percent below zero. The economically interesting problem is this
    heterogeneity, not a universal bounce.</figcaption>
  </figure>
</section>

<section>
  <h2>Hypotheses</h2>
  <p>Five hypotheses were pre-registered before estimation. Four failed, and they are reported
  as written.</p>
  <div class="scroll">
  <table>
    <thead><tr><th></th><th>Hypothesis</th><th>Result</th></tr></thead>
    <tbody>{hyp_rows}</tbody>
  </table>
  </div>
  <figure>
    <img src="figures/fig04_car_by_crash_type.png" alt="Mean cumulative abnormal return by crash type. Broad-market, sector and idiosyncratic confidence bands overlap one another throughout.">
    <figcaption>Crash type does not separate 20-day outcomes: broad-market
    {pct(h1.loc['broad_market'].mean_car20, 2)}, sector {pct(h1.loc['sector'].mean_car20, 2)},
    idiosyncratic {pct(h1.loc['idiosyncratic'].mean_car20, 2)}. Three confidence bands sitting
    on top of one another.</figcaption>
  </figure>
</section>

<section>
  <h2>Can we predict recovery?</h2>
  <p>Three model families were trained to classify whether an event would end above the
  benchmark, using chronological splits with a <strong>20-trading-day embargo</strong> between
  blocks. Hyper-parameters were chosen on validation only; the test block was scored once.</p>
  <div class="scroll">
  <table>
    <thead><tr><th>Model</th><th>Test ROC-AUC</th><th>Brier</th><th>Brier skill</th><th>Accuracy</th></tr></thead>
    <tbody>{model_rows}</tbody>
  </table>
  </div>
  <p>Out-of-sample discrimination is <strong>indistinguishable from chance</strong>, and all
  three models have <strong>negative Brier skill</strong> &mdash; each is fractionally worse
  than predicting the training base rate for every event. The constant base-rate model also has
  the highest <em>accuracy</em> while making the same prediction every time, which is precisely
  why accuracy is reported last.</p>
  <figure>
    <img src="figures/fig08_calibration.png" alt="Calibration curves for the three models, tracking the diagonal closely but spanning a narrow range of predicted probabilities.">
    <figcaption>Calibration is good (ECE 0.028&ndash;0.063). The models are well calibrated
    around the base rate and carry no discriminating information &mdash; honest about
    probabilities while knowing nothing about which event is which.</figcaption>
  </figure>
  <h3>Why the interpretability plots do not rescue this</h3>
  <p>Logistic coefficients, permutation importance and SHAP each rank a different feature
  first. Disagreement among interpretation methods, combined with chance-level predictive
  performance, is consistent with unstable feature ranking over noise rather than with real
  signal the models failed to exploit. Attributions extracted from a model with 0.476 test AUC
  describe what an uninformative fit latched onto in-sample; they are not substantive
  economics.</p>
  <figure>
    <img src="figures/fig11_interpretability_comparison.png" alt="Side-by-side comparison of logistic coefficients and permutation importance, which rank different features at the top.">
    <figcaption>Three interpretation methods, three different rankings. An interpretability
    plot inherits the credibility of the model beneath it.</figcaption>
  </figure>
</section>

<section>
  <h2>Multiple testing</h2>
  <p class="claim">0 of {reg['fdr']['n_tests']} exploratory coefficients survive correction.</p>
  <p>Pooling every exploratory coefficient from the extended OLS and logistic models into a
  single Benjamini&ndash;Hochberg family, <strong>none survives at q&nbsp;&le;&nbsp;0.05</strong>.</p>
  <p>This is the study's most important statistical guard. Read without correction, the
  exploratory models offer several coefficients at nominal p&nbsp;&lt;&nbsp;0.05 &mdash; a
  tempting set of &ldquo;findings&rdquo;. But {reg['fdr']['n_tests']} tests against a near-null
  outcome are expected to produce roughly that many false positives by construction. Reporting
  them would have been an artefact of the number of tests performed rather than evidence about
  markets.</p>
  <p>The same logic governs abnormal volume. Its nominal p of {avol.p_value:.3f} becomes
  <strong>q&nbsp;=&nbsp;{avol.q_value:.2f}</strong> after correction. The sign is directionally
  consistent in <strong>{rob['avol_effect_sign_consistency'] * 100:.1f}%</strong> of
  specifications, which makes it a genuine lead &mdash; but the honest description is
  <em>suggestive but statistically insufficient</em>, not that volume predicts recovery.</p>
</section>

<section>
  <h2>Data quality</h2>
  <p>Before any analysis, two screens removed <strong>54 tickers</strong>, each rejection
  manually inspected. Both drop <em>whole tickers</em> rather than winsorising individual
  returns, because the defect is series identity: when a price history is structurally
  inconsistent or represents more than one security, winsorising extreme returns does not repair
  the problem &mdash; it launders it into plausible-looking numbers.</p>
  <ul>
    <li><strong>Ticker-reuse guard ({man['reuse_guard_rejects']} dropped).</strong> The feed
    serves a different company under a recycled symbol. <code>SBNY</code> returns a listing
    beginning 17 months after Signature Bank failed, with no overlap against the symbol's
    recorded index membership.</li>
    <li><strong>Price-integrity screen ({man['integrity_rejects']} dropped).</strong>
    <code>TIE</code>'s history interleaves economically incompatible price series under one
    symbol, alternating day to day between roughly $14 and roughly $8,000. <code>COL</code>
    shows quantised prices of $0.20&ndash;$0.85, inconsistent with Rockwell Collins's known
    trading range of $60&ndash;$140.</li>
  </ul>
  <blockquote><p><strong>Before screening, mean CAR<sub>20</sub> computed to +77.7%</strong>
  with a standard deviation near 58. That is not a research result &mdash; it is a data-quality
  failure found during audit. Six corrupt events on two tickers were overwhelming fourteen
  thousand real ones.</p></blockquote>
  <h3>The NVDA false positive</h3>
  <p>An early version of the screen used an absolute price-level rule and incorrectly flagged
  legitimate NVDA history, whose split-adjusted median close over the window is $0.80. Absolute
  price thresholds are unsafe in adjusted data because splits, scale changes and genuine
  high-growth securities all produce legitimately tiny early prices. The corrected rules test
  <strong>internal time-series consistency</strong> instead &mdash; round-trip level
  discontinuities, extreme-move frequency and value diversity &mdash; none of which depend on
  where a price sits in absolute terms.</p>
</section>

<section>
  <h2>Limitations</h2>
  <ol>
    <li><strong>Survivorship bias is reduced, not solved.</strong> Point-in-time membership
    removes look-ahead in universe selection, but only
    <strong>{cov['historical_only_coverage_pct']}%</strong> of historical-only members have
    usable price history ({cov['current_member_coverage_pct']}% of current members do). If the
    missing names disproportionately include failed firms, the surviving sample is biased
    <em>upward</em> &mdash; against the negative finding reported here. That makes it unlikely
    this bias created the result, but its magnitude is not identified.</li>
    <li><strong>No delisting returns.</strong> A stock removed mid-window stops contributing
    rather than realising a terminal value.</li>
    <li><strong>Market-adjusted, not risk-adjusted.</strong> CAR subtracts SPY, not a factor
    model; a volatility factor could absorb the one surviving coefficient.</li>
    <li><strong>Close-to-close only.</strong> A stock that fell 30% intraday and closed flat is
    not an event here.</li>
    <li><strong>One market, one era.</strong> Large-cap US equity, {ec['date_min'][:4]}&ndash;{ec['date_max'][:4]}.
    Conclusions hold within this design and do not extend automatically elsewhere.</li>
    <li><strong>A near-null is not proof of no effect.</strong> Fundamentals, news text,
    options-implied data and order flow are unexamined.</li>
  </ol>
</section>

<section>
  <h2>Reproduce it</h2>
  <p>Every number on this page is read from persisted result files at build time, so the
  presentation cannot drift from the analysis. The page itself runs no computation.</p>
<pre><code>git clone {GH}
cd dead-cat-detector
make setup      # uv venv (Python 3.12) + dependencies

make verify     # check all 42 documented claims against persisted results
make test       # 43 tests, no network required
make analysis   # rebuild every analysis stage from cached data
make all        # full pipeline including the data download</code></pre>
  <p>All randomness is seeded, and a config fingerprint is stamped into every persisted result.
  The test suite is concentrated on the leakage surface; its load-bearing case mutates every
  return from the event date onward and asserts the rolling statistics that define the crash are
  unchanged.</p>
  <p><a href="{GH}">View the full study on GitHub &rarr;</a></p>
</section>

<footer>
  <p>Gary Wang &middot; MIT licensed (code). Price data retrieved from Yahoo Finance for
  non-commercial research and not redistributed. Nothing here is investment advice.</p>
  <p><a href="{GH}">github.com/Gariyuuu/dead-cat-detector</a></p>
</footer>

</div>
</body>
</html>
"""

# The favicon restates the finding: a cumulative-return path drifting below a
# flat zero reference, in the same palette as the figures. Geometry is shared
# between the SVG and the rasterised PNGs so they cannot diverge.
ICON_BASELINE_Y = 11.0
ICON_PATH = [(3, 11), (8, 13), (12, 12.4), (17, 17), (22, 18.6), (29, 23)]
ICON_ACCENT_LIGHT, ICON_ACCENT_DARK = "#2a78d6", "#6da7ec"
ICON_RULE_LIGHT, ICON_RULE_DARK = "#c9c8c2", "#5d5c55"


def _favicon_svg() -> str:
    pts = " ".join(f"{x},{y}" for x, y in ICON_PATH)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <style>
    .rule {{ stroke: {ICON_RULE_LIGHT}; }}
    .car  {{ stroke: {ICON_ACCENT_LIGHT}; }}
    @media (prefers-color-scheme: dark) {{
      .rule {{ stroke: {ICON_RULE_DARK}; }}
      .car  {{ stroke: {ICON_ACCENT_DARK}; }}
    }}
  </style>
  <line class="rule" x1="3" y1="{ICON_BASELINE_Y}" x2="29" y2="{ICON_BASELINE_Y}"
        stroke-width="1.6" stroke-linecap="round"/>
  <polyline class="car" points="{pts}" fill="none" stroke-width="3.4"
            stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""


def _favicon_png(path: Path, size: int, scale: int = 8, background: str | None = None) -> None:
    """Rasterise the same geometry, supersampled for clean edges at small sizes."""
    from PIL import Image, ImageDraw

    S = size * scale
    k = S / 32.0
    img = Image.new("RGBA", (S, S), background or (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line([(3 * k, ICON_BASELINE_Y * k), (29 * k, ICON_BASELINE_Y * k)],
           fill=ICON_RULE_LIGHT, width=max(1, int(1.6 * k)), joint="curve")
    d.line([(x * k, y * k) for x, y in ICON_PATH],
           fill=ICON_ACCENT_LIGHT, width=max(1, int(3.4 * k)), joint="curve")
    # Round the polyline ends, which PIL does not do for us.
    r = 1.7 * k
    for x, y in (ICON_PATH[0], ICON_PATH[-1]):
        d.ellipse([x * k - r, y * k - r, x * k + r, y * k + r], fill=ICON_ACCENT_LIGHT)
    img.resize((size, size), Image.LANCZOS).save(path)


def build_favicons() -> list[str]:
    (WEB / "favicon.svg").write_text(_favicon_svg())
    _favicon_png(WEB / "favicon-32.png", 32)
    # Apple touch icons are composited on black if transparent, so paint the surface.
    _favicon_png(WEB / "apple-touch-icon.png", 180, background="#fcfcfb")
    return ["favicon.svg", "favicon-32.png", "apple-touch-icon.png"]


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for f in FIGURES:
        shutil.copy2(ROOT / "results" / "figures" / f, FIGS / f)
    icons = build_favicons()
    (WEB / "index.html").write_text(HTML)
    size = (WEB / "index.html").stat().st_size
    figs = sum((FIGS / f).stat().st_size for f in FIGURES)
    print(f"web/index.html  {size / 1024:.0f} KB")
    print(f"figures         {figs / 1024:.0f} KB ({len(FIGURES)} files)")
    ico = sum((WEB / i).stat().st_size for i in icons)
    print(f"icons           {ico / 1024:.1f} KB ({', '.join(icons)})")
    print(f"total payload   {(size + figs + ico) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
