"""Is the weather/events improvement real? Block bootstrap over test days.

The ablation reports a relative WAPE improvement of the conditions model over the
history-only baseline. A point estimate alone cannot say whether that gap would
survive a different draw of test days — demand errors are strongly correlated
*within* a day (one storm, one parade), so a naive per-row bootstrap would be
wildly overconfident. The resampling unit here is therefore the **test day**:
days are drawn with replacement, WAPE is recomputed for both models on each
resample, and the distribution of the relative delta gives the confidence
interval and a one-sided p-value for "the improvement is > 0".

Everything runs in pandas/numpy from the exported artifacts — the conditions
model's JSON coefficients are applied directly (the ablation proved this
arithmetic matches Spark's predictions to ~1e-13), and the recomputed WAPEs are
validated against the stored metrics before any bootstrap is trusted.

WAPE decomposes over days (sum of |error| over sum of actuals), so each
resample is two vector sums — 10,000 replicates run in milliseconds.

The trips-only reference is ``hist_avg_demand``: on this data the fitted
time-only OLS converges to it (identical MAE to 3 decimals), so "conditions
model vs hist_avg" and "full vs time-only feature set" are the same comparison,
which the validation below re-checks rather than assumes.

    python -m src.batch.significance
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

N_BOOT = 10_000
SIGNIFICANCE_CSV = config.REPORTS_DIR / "significance.csv"
RAIN_THRESHOLD_MM = 1.0  # same threshold as ablation.py


def apply_conditions_model(frame: pd.DataFrame, model: dict) -> pd.Series:
    """The exported linear model, re-applied column-wise (clamped at zero)."""
    coef = model["coefficients"]
    mean_temp = float(model["train_mean_temp_c"])

    two_pi = 2.0 * math.pi
    hist = frame["hist_avg_demand"]
    columns = {
        "hist_avg_demand": hist,
        "hour_sin": frame["hour_sin"], "hour_cos": frame["hour_cos"],
        "dow_sin": frame["dow_sin"], "dow_cos": frame["dow_cos"],
        "weekend_d": frame["is_weekend"].astype(float),
        "temp_c": frame["temp_c"], "precip_mm": frame["precip_mm"],
        "temp_dev_x_hist": (frame["temp_c"] - mean_temp) * hist,
        "precip_x_hist": frame["precip_mm"] * hist,
        "holiday_d": frame["is_holiday"].astype(float),
        "fedhol_d": frame["is_federal_holiday"].astype(float),
        "event_d": frame["is_event"].astype(float),
        "fedhol_x_hist": frame["is_federal_holiday"].astype(float) * hist,
        "event_x_hist": frame["is_event"].astype(float) * hist,
    }
    for k in range(1, config.FOURIER_TERMS + 1):
        angle = two_pi * k * frame["hour_of_week"] / config.HOURS_PER_WEEK
        columns[f"how_sin_{k}"] = np.sin(angle)
        columns[f"how_cos_{k}"] = np.cos(angle)

    raw = float(model["intercept"]) + sum(
        coef[name] * values for name, values in columns.items()
    )
    return raw.clip(lower=0.0)


def day_sums(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-day components of WAPE for both models: sum|err| and sum(actual)."""
    parts = pd.DataFrame({
        "date_local": frame["date_local"],
        "abs_err_full": (frame["pred_full"] - frame["trip_count"]).abs(),
        "abs_err_hist": (frame["hist_avg_demand"] - frame["trip_count"]).abs(),
        "actual": frame["trip_count"],
    })
    return parts.groupby("date_local").sum()


def bootstrap(day_table: pd.DataFrame, rng: np.random.Generator) -> dict:
    """CI and p-value for the relative WAPE improvement, resampling days."""
    full = day_table["abs_err_full"].to_numpy()
    hist = day_table["abs_err_hist"].to_numpy()
    actual = day_table["actual"].to_numpy()
    n_days = len(day_table)

    point = 100.0 * (hist.sum() - full.sum()) / hist.sum()

    draws = rng.integers(0, n_days, size=(N_BOOT, n_days))
    full_sum = full[draws].sum(axis=1)
    hist_sum = hist[draws].sum(axis=1)
    deltas = 100.0 * (hist_sum - full_sum) / hist_sum

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    # One-sided: how often would resampled days show no improvement at all?
    p_value = float((deltas <= 0.0).mean())
    return {
        "n_days": n_days,
        "wape_hist_pct": 100.0 * hist.sum() / actual.sum(),
        "wape_full_pct": 100.0 * full.sum() / actual.sum(),
        "delta_rel_pct": point,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "p_one_sided": p_value,
    }


def main() -> int:
    for path in (config.FEATURES_PARQUET, config.CONDITIONS_MODEL_JSON):
        if not Path(path).exists():
            print(f"ERROR: {path} missing — run the batch pipeline + ablation first",
                  file=sys.stderr)
            return 2

    model = json.loads(config.CONDITIONS_MODEL_JSON.read_text())
    frame = pd.read_parquet(config.FEATURES_PARQUET)
    test = frame[~frame["is_train"]].copy()
    test["pred_full"] = apply_conditions_model(test, model)

    print("=" * 78)
    print("Block bootstrap — is the weather/events improvement real?")
    print("=" * 78)
    print(f"  test rows       : {len(test):,}")
    print(f"  resampling unit : test day (errors correlate within a day)")
    print(f"  replicates      : {N_BOOT:,}   seed {config.SEED}")

    # --- validation before trusting anything ----------------------------------
    print("\nVALIDATION")
    stored = model["test_metrics"]
    recomputed_wape = float(
        (test["pred_full"] - test["trip_count"]).abs().sum() / test["trip_count"].sum()
    )
    drift = abs(recomputed_wape - stored["this_model"]["wape"])
    print(f"  recomputed full-model WAPE : {100 * recomputed_wape:.3f}% "
          f"(stored {100 * stored['this_model']['wape']:.3f}%, "
          f"|diff| {drift:.2e})")
    ok = drift < 1e-9
    hist_wape = float(
        (test["hist_avg_demand"] - test["trip_count"]).abs().sum()
        / test["trip_count"].sum()
    )
    hist_drift = abs(hist_wape - stored["hist_avg_baseline"]["wape"])
    print(f"  recomputed hist_avg WAPE   : {100 * hist_wape:.3f}% "
          f"(stored {100 * stored['hist_avg_baseline']['wape']:.3f}%, "
          f"|diff| {hist_drift:.2e})")
    ok = ok and hist_drift < 1e-9
    time_only_gap = abs(stored["time_only"]["wape"] - stored["hist_avg_baseline"]["wape"])
    print(f"  time-only OLS vs hist_avg  : |WAPE diff| {100 * time_only_gap:.4f} pp "
          "-> hist_avg is a faithful trips-only reference")
    ok = ok and time_only_gap < 5e-3
    print(f"  -> {'PASS' if ok else 'FAIL — do not trust the bootstrap below'}")

    # --- the bootstraps --------------------------------------------------------
    rng = np.random.default_rng(config.SEED)
    subsets = {
        "all": test,
        "special_days": test[test["is_special_day"]],
        "rain_hours": test[test["precip_mm"] > RAIN_THRESHOLD_MM],
    }

    rows = []
    print("\nRESULTS — relative WAPE improvement, conditions model vs history-only")
    print(f"  {'subset':<14}{'days':>6}{'WAPE hist':>11}{'WAPE full':>11}"
          f"{'delta':>9}{'95% CI':>19}{'p (one-sided)':>15}")
    print("  " + "-" * 84)
    for name, subset in subsets.items():
        result = bootstrap(day_sums(subset), rng)
        rows.append({"subset": name, **result})
        print(f"  {name:<14}{result['n_days']:>6}"
              f"{result['wape_hist_pct']:>10.2f}%{result['wape_full_pct']:>10.2f}%"
              f"{result['delta_rel_pct']:>8.2f}%"
              f"{'[' + format(result['ci95_lo'], '.2f') + ', ' + format(result['ci95_hi'], '.2f') + ']':>19}"
              f"{result['p_one_sided']:>15.4f}")

    print("\n  Reading: a CI excluding 0 means the improvement survives resampling the")
    print("  test days; p is the fraction of resamples showing NO improvement. The")
    print("  special-days subset has few days — its wide CI is the honest cost of a")
    print("  time-based split, reported rather than hidden.")

    with SIGNIFICANCE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  wrote : {SIGNIFICANCE_CSV}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
