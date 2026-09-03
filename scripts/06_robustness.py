"""Robustness grid: every specification, every estimate, persisted.

The grid crosses crash threshold x volatility window x cooldown x outcome
horizon x high-VIX definition. Every cell is written to disk whether or not it
agrees with the primary specification - the heatmap is built from the complete
grid, not a favourable subset.
"""
from __future__ import annotations

import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deadcat import statistics as S  # noqa: E402
from deadcat.config import PATHS, load_config  # noqa: E402
from deadcat.pipeline import Panel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("robustness")

PRIMARY_X = ["crash_z", "avol", "mom_20", "rv_20", "spy_ret_event", "vix"]


def main() -> None:
    cfg = load_config()
    PATHS.ensure()
    rb = cfg.robustness
    panel = Panel(cfg)

    zs = list(rb.z_thresholds)
    vws = list(rb.vol_windows)
    cds = list(rb.cooldowns)
    hzs = list(rb.horizons)
    vqs = list(rb.vix_high_quantiles)
    log.info("grid: %d event builds -> %d specifications",
             len(zs) * len(vws) * len(cds), len(zs) * len(vws) * len(cds) * len(hzs) * len(vqs))

    rows = []
    for z, vw, cd in itertools.product(zs, vws, cds):
        ev = panel.build(z_threshold=z, vol_window=vw, cooldown=cd)
        if ev.empty:
            continue
        ev["event_date"] = pd.to_datetime(ev["event_date"])
        log.info("z=%.1f vol_window=%d cooldown=%d -> %d events", z, vw, cd, len(ev))

        for h in hzs:
            car = f"car_{h}"
            if car not in ev.columns:
                continue
            sub = ev[ev[car].notna()]
            if len(sub) < 100:
                continue
            base = S.clustered_mean_se(sub[car], sub["event_date"])
            rec = float((sub[car] > 0).mean())

            # Conditional splits that do not depend on the VIX definition.
            broad = sub[sub.crash_type == "broad_market"][car]
            idio = sub[sub.crash_type == "idiosyncratic"][car]
            sect = sub[sub.crash_type == "sector"][car]
            q4 = sub[sub.avol_quartile == "Q4"][car]
            q1 = sub[sub.avol_quartile == "Q1"][car]

            # OLS on this specification (primary right-hand side).
            try:
                o = S.ols_hc3(sub.assign(_y=sub[car]), "_y", PRIMARY_X, cluster="event_date")
                ot = o["table"].set_index("term")
                adj_r2 = o["adj_r2"]
                b_z = float(ot.loc["crash_z", "coef"]); p_z = float(ot.loc["crash_z", "p_value"])
                b_a = float(ot.loc["avol", "coef"]); p_a = float(ot.loc["avol", "p_value"])
                b_rv = float(ot.loc["rv_20", "coef"]); p_rv = float(ot.loc["rv_20", "p_value"])
            except Exception as exc:  # pragma: no cover
                log.warning("OLS failed for z=%s vw=%s cd=%s h=%s: %s", z, vw, cd, h, exc)
                adj_r2 = b_z = p_z = b_a = p_a = b_rv = p_rv = np.nan

            for vq in vqs:
                cut = sub["vix"].quantile(vq)
                hi = sub[sub["vix"] >= cut][car]
                lo = sub[sub["vix"] < cut][car]
                rows.append({
                    "z_threshold": z, "vol_window": vw, "cooldown": cd,
                    "horizon": h, "vix_high_quantile": vq,
                    "n_events": len(sub), "n_dates": int(sub.event_date.nunique()),
                    "mean_car": base["mean"], "se_clustered": base["se"],
                    "ci_lo": base["ci_lo"], "ci_hi": base["ci_hi"],
                    "p_value": base["p_value"], "recovery_rate": rec,
                    "median_car": float(sub[car].median()),
                    "mean_car_broad": float(broad.mean()) if len(broad) else np.nan,
                    "mean_car_sector": float(sect.mean()) if len(sect) else np.nan,
                    "mean_car_idio": float(idio.mean()) if len(idio) else np.nan,
                    "diff_broad_minus_idio": (float(broad.mean() - idio.mean())
                                              if len(broad) and len(idio) else np.nan),
                    "mean_car_avol_q4": float(q4.mean()) if len(q4) else np.nan,
                    "mean_car_avol_q1": float(q1.mean()) if len(q1) else np.nan,
                    "diff_avol_q4_minus_q1": (float(q4.mean() - q1.mean())
                                              if len(q4) and len(q1) else np.nan),
                    "mean_car_high_vix": float(hi.mean()) if len(hi) else np.nan,
                    "mean_car_low_vix": float(lo.mean()) if len(lo) else np.nan,
                    "diff_high_minus_low_vix": (float(hi.mean() - lo.mean())
                                                if len(hi) and len(lo) else np.nan),
                    "ols_adj_r2": adj_r2,
                    "beta_crash_z": b_z, "p_crash_z": p_z,
                    "beta_avol": b_a, "p_avol": p_a,
                    "beta_rv20": b_rv, "p_rv20": p_rv,
                })

    grid = pd.DataFrame(rows)
    grid.to_csv(PATHS.tables / "robustness_grid.csv", index=False)
    log.info("persisted %d specifications", len(grid))

    prim = grid[(grid.z_threshold == cfg.events.z_threshold)
                & (grid.vol_window == cfg.events.vol_window)
                & (grid.cooldown == cfg.events.cooldown_days)
                & (grid.horizon == cfg.outcomes.primary_horizon)
                & (grid.vix_high_quantile == cfg.regimes.vix_high_quantile)]

    at_h = grid[grid.horizon == cfg.outcomes.primary_horizon].drop_duplicates(
        ["z_threshold", "vol_window", "cooldown"])
    summary = {
        "config_fingerprint": cfg.fingerprint,
        "n_specifications": int(len(grid)),
        "primary_cell": prim.iloc[0].to_dict() if len(prim) else None,
        "car_h20_across_specs": {
            "n": int(len(at_h)),
            "min": float(at_h.mean_car.min()), "max": float(at_h.mean_car.max()),
            "median": float(at_h.mean_car.median()),
            "share_negative": float((at_h.mean_car < 0).mean()),
            "share_negative_and_significant": float(
                ((at_h.mean_car < 0) & (at_h.p_value < 0.05)).mean()),
            "share_positive_and_significant": float(
                ((at_h.mean_car > 0) & (at_h.p_value < 0.05)).mean()),
        },
        "recovery_rate_across_specs": {
            "min": float(at_h.recovery_rate.min()), "max": float(at_h.recovery_rate.max()),
            "median": float(at_h.recovery_rate.median()),
            "share_below_half": float((at_h.recovery_rate < 0.5).mean()),
        },
        "all_horizons": grid.groupby("horizon").agg(
            mean_car=("mean_car", "median"),
            share_negative=("mean_car", lambda s: float((s < 0).mean())),
            share_sig_negative=("mean_car", "size"),
        ).to_dict(),
        "avol_effect_sign_consistency": float((grid.diff_avol_q4_minus_q1 > 0).mean()),
        "broad_vs_idio_sign_consistency": float((grid.diff_broad_minus_idio < 0).mean()),
    }
    (PATHS.metrics / "robustness.json").write_text(json.dumps(summary, indent=2, default=str))
    log.info("summary:\n%s", json.dumps(summary["car_h20_across_specs"], indent=2))
    log.info("recovery:\n%s", json.dumps(summary["recovery_rate_across_specs"], indent=2))


if __name__ == "__main__":
    main()
