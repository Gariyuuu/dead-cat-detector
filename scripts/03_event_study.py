"""Event study: mean CAR in event time with bootstrap bands, by conditioning group.

Tests H1 (do extreme declines mean-revert at all?), H2 (crash type), H3
(abnormal volume) and H5 (VIX regime) on the primary specification.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deadcat import data as D, events as E, statistics as S  # noqa: E402
from deadcat.config import PATHS, load_config  # noqa: E402
from deadcat.pipeline import Panel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("event_study")

MAX_TAU = 60


def main() -> None:
    cfg = load_config()
    PATHS.ensure()
    seed = int(cfg.study.seed)
    n_boot = int(cfg.bootstrap.n_boot)
    ci = float(cfg.bootstrap.ci)
    H = int(cfg.outcomes.primary_horizon)

    panel = Panel(cfg)
    ev = D.load_processed("events.parquet")
    log.info("events=%d", len(ev))

    paths = E.car_path(ev, panel.close, panel.spy, max_tau=MAX_TAU)
    complete = np.isfinite(paths).all(axis=1)
    log.info("events with a complete %d-day path: %d", MAX_TAU, complete.sum())

    groups: dict[str, np.ndarray] = {"all_events": np.ones(len(ev), bool)}
    for t in ("broad_market", "sector", "idiosyncratic"):
        groups[t] = (ev["crash_type"] == t).to_numpy()
    groups["avol_q4_highest"] = (ev["avol_quartile"] == "Q4").to_numpy()
    groups["avol_q1_lowest"] = (ev["avol_quartile"] == "Q1").to_numpy()
    groups["high_vix"] = (ev["vix_regime"] == "high_vix").to_numpy()
    groups["low_vix"] = (ev["vix_regime"] == "low_vix").to_numpy()

    frames = []
    for name, mask in groups.items():
        sub = paths[mask]
        if np.isfinite(sub).all(axis=1).sum() < 30:
            log.warning("skipping %s (too few complete paths)", name)
            continue
        tbl = S.bootstrap_mean_path(sub, n_boot=n_boot, ci=ci, seed=seed)
        tbl.insert(0, "group", name)
        frames.append(tbl)
        log.info("%-18s n=%6d  CAR20=%+.4f  CAR60=%+.4f",
                 name, tbl.n_events.iloc[0], tbl.mean_car.iloc[H], tbl.mean_car.iloc[MAX_TAU])
    path_tbl = pd.concat(frames, ignore_index=True)
    path_tbl.to_csv(PATHS.tables / "event_study_paths.csv", index=False)

    # ---- H1: is mean CAR_20 different from zero, overall and per group? ----
    h1_rows = []
    for name, mask in groups.items():
        x = ev.loc[mask, f"car_{H}"].to_numpy()
        r = S.bootstrap_mean(x, n_boot=n_boot, ci=ci, seed=seed)
        med = float(np.nanmedian(x))
        rec = float(np.nanmean(ev.loc[mask, f"recovered_{H}d"]))
        h1_rows.append({"group": name, "mean_car20": r["mean"], "median_car20": med,
                        "ci_lo": r["lo"], "ci_hi": r["hi"], "p_value": r["p_two_sided"],
                        "n": r["n"], "recovery_rate": rec,
                        "regained_precrash_rate": float(np.nanmean(ev.loc[mask, f"regained_precrash_{H}"]))})
    h1 = pd.DataFrame(h1_rows)
    fdr = S.benjamini_hochberg(h1["p_value"].values)
    h1["q_value"], h1["reject_fdr"] = fdr["q_value"], fdr["reject_fdr"]
    h1.to_csv(PATHS.tables / "h1_mean_car_by_group.csv", index=False)

    # ---- H2/H3/H5: pairwise contrasts ----
    contrasts = [
        ("H2", "broad_market", "idiosyncratic"),
        ("H2", "sector", "idiosyncratic"),
        ("H2", "broad_market", "sector"),
        ("H3", "avol_q4_highest", "avol_q1_lowest"),
        ("H5", "high_vix", "low_vix"),
    ]
    rows = []
    for hyp, a, b in contrasts:
        ra = ev.loc[groups[a], f"car_{H}"].to_numpy()
        rb = ev.loc[groups[b], f"car_{H}"].to_numpy()
        d = S.bootstrap_diff(ra, rb, n_boot=n_boot, ci=ci, seed=seed)
        pa = float(np.nanmean(ev.loc[groups[a], f"recovered_{H}d"]))
        pb = float(np.nanmean(ev.loc[groups[b], f"recovered_{H}d"]))
        rows.append({"hypothesis": hyp, "group_a": a, "group_b": b,
                     "mean_car20_a": float(np.nanmean(ra)), "mean_car20_b": float(np.nanmean(rb)),
                     "diff": d["diff"], "ci_lo": d["lo"], "ci_hi": d["hi"],
                     "p_value": d["p_two_sided"], "n_a": d["n_a"], "n_b": d["n_b"],
                     "recovery_rate_a": pa, "recovery_rate_b": pb,
                     "recovery_rate_diff": pa - pb})
    con = pd.DataFrame(rows)
    fdr = S.benjamini_hochberg(con["p_value"].values)
    con["q_value"], con["reject_fdr"] = fdr["q_value"], fdr["reject_fdr"]
    con.to_csv(PATHS.tables / "group_contrasts.csv", index=False)

    # ---- Horizon term structure for the full sample ----
    hz = []
    for h in list(cfg.outcomes.horizons):
        r = S.bootstrap_mean(ev[f"car_{h}"].to_numpy(), n_boot=n_boot, ci=ci, seed=seed)
        hz.append({"horizon": h, **r})
    pd.DataFrame(hz).to_csv(PATHS.tables / "car_by_horizon.csv", index=False)

    summary = {
        "config_fingerprint": cfg.fingerprint,
        "n_events": int(len(ev)),
        "n_complete_paths": int(complete.sum()),
        "primary_horizon": H,
        "n_bootstrap": n_boot,
        "seed": seed,
        "h1_all_events": h1[h1.group == "all_events"].iloc[0].to_dict(),
        "car_by_horizon": hz,
        "contrasts": con.to_dict("records"),
    }
    (PATHS.metrics / "event_study.json").write_text(json.dumps(summary, indent=2, default=str))
    np.save(PATHS.data_processed / "car_paths.npy", paths)

    log.info("\n%s", h1.round(5).to_string(index=False))
    log.info("\n%s", con.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
