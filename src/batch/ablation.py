"""Feature ablation: do weather & events beat trips-only? (proposal open question #2)

This is the supervised half of the modelling story — the K-Means clustering answers
*where and when* demand concentrates; this script answers whether the **external
signals** the proposal promised to fuse (Open-Meteo weather, NYC events/holidays)
carry predictive information beyond what the demand history already knows.

**Design.** Four nested feature sets, each fitted by two learners on the identical
train split and scored on the identical held-out test rows with the same
MAE/RMSE/WAPE code ``evaluate.py`` uses (imported, not re-implemented):

  feature set             adds
  ----------------------  -----------------------------------------------------------
  time only               ``hist_avg_demand`` + cyclical hour/dow + weekly Fourier
  time + weather          ``temp_c``, ``precip_mm``, and their x-hist interactions
  time + events           holiday/event flags and their x-hist interactions
  time + weather + events all of the above

  learner                 role
  ----------------------  -----------------------------------------------------------
  linear (OLS)            interpretable coefficients; **portable** — exported to JSON
                          and re-applied in pandas by ``predict_live.py``
  gradient-boosted trees  capacity ceiling — finds non-linear weather interactions
                          the linear model cannot, at the cost of a Spark-only model

The weather/event columns are interacted with ``hist_avg_demand`` because their
effect is proportional, not additive: a rainy Friday adds many trips in Midtown and
a fraction of one trip in a quiet zone. A global additive rain coefficient would be
mis-specified for 225 zones spanning three orders of magnitude in volume.

**The verdict is whatever the numbers say.** The deltas of ``+ weather + events``
over ``time only`` are reported overall, on special days (holiday or event), and on
rain hours (precip > 1 mm) — the subsets where the signals should earn their keep.
The convention printed with the verdict: a relative WAPE improvement >= 2% overall
or >= 5% on a target subset counts as "meaningful"; anything smaller is reported as
marginal, not spun. The README's caveat applies when reading the weather result:
Open-Meteo's ~11 km grid collapses 225 zones onto 27 distinct weather series, so
weather here is close to a citywide temporal signal, not a per-zone one.

**No leakage.** ``hist_avg_demand`` is already train-only (features.py). Both
learners fit on ``is_train`` rows only. Weather and event flags are *observed
exogenous inputs* — knowing the rain at prediction time is the operating assumption
of the deployed system (a weather forecast), not target leakage.

**Artifacts.**
  * ``reports/ablation_metrics.csv``   every (learner, feature set, subset) scored
  * ``models/conditions_model.json``   the full linear model: intercept,
    coefficients, temperature normals — everything predict_live.py needs to serve
    weather/event-aware forecasts instantly in pandas (verified here against
    Spark's own predictions before it is written)
  * ``models/hist_avg.parquet``        the (zone, hour, dow) -> hist_avg table the
    portable model takes as its base input

    python -m src.batch.ablation             # both learners (GBT takes minutes)
    python -m src.batch.ablation --skip-gbt  # linear only, fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.batch.evaluate import metrics_for, print_table  # noqa: E402
from src.spark_session import describe, get_spark  # noqa: E402

# Base temporal features. is_special_day is deliberately excluded everywhere: it is
# the OR of the other flags, and perfect collinearity would make the linear
# coefficients unreadable without adding information.
TIME_FEATURES = [
    "hist_avg_demand",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "how_sin_1", "how_cos_1", "how_sin_2", "how_cos_2", "how_sin_3", "how_cos_3",
    "weekend_d",
]
WEATHER_FEATURES = ["temp_c", "precip_mm", "temp_dev_x_hist", "precip_x_hist"]
EVENT_FEATURES = ["holiday_d", "fedhol_d", "event_d", "fedhol_x_hist", "event_x_hist"]

FEATURE_SETS: list[tuple[str, list[str]]] = [
    ("time only", TIME_FEATURES),
    ("time + weather", TIME_FEATURES + WEATHER_FEATURES),
    ("time + events", TIME_FEATURES + EVENT_FEATURES),
    ("time + weather + events", TIME_FEATURES + WEATHER_FEATURES + EVENT_FEATURES),
]

LEARNERS = ["linear", "gbt"]

# The convention the verdict is judged by — printed with the result, never implicit.
MEANINGFUL_OVERALL_PCT = 2.0   # relative WAPE improvement, full test grid
MEANINGFUL_SUBSET_PCT = 5.0    # relative WAPE improvement, special days / rain hours

RAIN_THRESHOLD_MM = 1.0


def prepare(frame: DataFrame) -> tuple[DataFrame, float]:
    """Cast flags to doubles and build the x-hist interaction columns.

    Returns the augmented frame and the train-split mean temperature (the centre
    for ``temp_dev``, so the temperature interaction reads as "per degree away
    from typical", not "per degree above zero").
    """
    train_mean_temp = float(
        frame.filter(F.col("is_train")).agg(F.avg("temp_c")).first()[0]
    )
    return (
        frame.withColumn("weekend_d", F.col("is_weekend").cast("double"))
        .withColumn("holiday_d", F.col("is_holiday").cast("double"))
        .withColumn("fedhol_d", F.col("is_federal_holiday").cast("double"))
        .withColumn("event_d", F.col("is_event").cast("double"))
        .withColumn(
            "temp_dev_x_hist",
            (F.col("temp_c") - F.lit(train_mean_temp)) * F.col("hist_avg_demand"),
        )
        .withColumn("precip_x_hist", F.col("precip_mm") * F.col("hist_avg_demand"))
        .withColumn("fedhol_x_hist", F.col("fedhol_d") * F.col("hist_avg_demand"))
        .withColumn("event_x_hist", F.col("event_d") * F.col("hist_avg_demand"))
    ), train_mean_temp


def fit_predict(
    frame: DataFrame, learner: str, features: list[str], out_col: str
) -> tuple[DataFrame, object]:
    """Fit one (learner, feature set) on train rows, predict every row.

    Demand cannot be negative, so predictions are clamped at zero — the linear
    model would otherwise forecast small negative counts in quiet cells.
    """
    assembler = VectorAssembler(inputCols=features, outputCol="_features")
    assembled = assembler.transform(frame)

    if learner == "linear":
        # OLS via the normal-equation solver: exact, fast at this width, and free
        # of regularisation so the exported coefficients are plain least squares.
        estimator = LinearRegression(
            featuresCol="_features", labelCol="trip_count",
            regParam=0.0, elasticNetParam=0.0, solver="normal",
        )
    elif learner == "gbt":
        estimator = GBTRegressor(
            featuresCol="_features", labelCol="trip_count",
            maxDepth=6, maxIter=30, stepSize=0.1, subsamplingRate=0.8,
            seed=config.SEED,
        )
    else:
        raise ValueError(learner)

    model = estimator.fit(assembled.filter(F.col("is_train")))
    predicted = (
        model.transform(assembled)
        .withColumn(out_col, F.greatest(F.lit(0.0), F.col("prediction")))
        .drop("_features", "prediction")
    )
    return predicted, model


def validate(test: DataFrame, columns: list[str], n_expected: int) -> bool:
    """Every variant must score the identical test rows, with no nulls."""
    ok = True
    print("\n" + "=" * 92)
    print("VALIDATION")
    print("=" * 92)
    print(f"  test rows                   : {n_expected:,}")
    for column in columns:
        scored = test.filter(F.col(column).isNotNull()).count()
        negative = test.filter(F.col(column) < 0).count()
        flag = "" if scored == n_expected and negative == 0 else "   <-- PROBLEM"
        print(f"    {column:<28} scored {scored:>9,}   negative {negative}{flag}")
        ok = ok and scored == n_expected and negative == 0
    return ok


def check_portable_model(
    test: DataFrame, model, features: list[str], intercept: float,
) -> tuple[bool, float]:
    """Prove the exported (intercept, coefficients) reproduce Spark's predictions.

    predict_live.py will apply the JSON in pandas; if plain arithmetic on the
    exported numbers did not equal ``model.transform``, the live answer would
    silently drift from the evaluated one. Checked on the 20 busiest test rows —
    the cells where an error would matter most.
    """
    coefficients = list(model.coefficients)
    rows = (
        test.orderBy(F.desc("trip_count"))
        .select(*features, "pred_linear_full")
        .limit(20)
        .collect()
    )
    worst = 0.0
    for row in rows:
        manual = intercept + sum(c * float(row[f]) for c, f in zip(coefficients, features))
        manual = max(0.0, manual)
        worst = max(worst, abs(manual - float(row["pred_linear_full"])))
    ok = worst < 1e-6
    print(f"\n  portable-model parity (JSON arithmetic vs Spark, 20 busiest rows):")
    print(f"    max |difference| = {worst:.2e}  ->  {'PASS' if ok else 'FAIL'}")
    return ok, worst


def verdict(results: dict[tuple[str, str, str], dict], learners: list[str]) -> None:
    """The answer to open question #2, stated from the numbers."""
    print("\n" + "=" * 92)
    print("VERDICT — do weather & events meaningfully beat trips-only?")
    print("=" * 92)
    print(f"  Convention: 'meaningful' = relative WAPE improvement of "
          f">= {MEANINGFUL_OVERALL_PCT:.0f}% overall")
    print(f"  or >= {MEANINGFUL_SUBSET_PCT:.0f}% on a target subset "
          "(special days, rain hours). Stated up front so the")
    print("  threshold is a decision, not a post-hoc rationalisation.")

    for learner in learners:
        base = results[(learner, "time only", "all")]
        full = results[(learner, "time + weather + events", "all")]
        print(f"\n  {learner.upper()}")
        for subset, label in (
            ("all", "full test grid"),
            ("special", "special days (holiday or event)"),
            ("rain", f"rain hours (precip > {RAIN_THRESHOLD_MM:g} mm)"),
        ):
            b = results[(learner, "time only", subset)]
            f_ = results[(learner, "time + weather + events", subset)]
            if b["wape"] is None or f_["wape"] is None:
                continue
            rel = 100.0 * (b["wape"] - f_["wape"]) / b["wape"]
            print(f"    {label:<38} WAPE {100 * b['wape']:6.2f}% -> "
                  f"{100 * f_['wape']:6.2f}%   ({rel:+.2f}% relative)")

        rel_all = 100.0 * (base["wape"] - full["wape"]) / base["wape"]
        rel_special = None
        b_s = results[(learner, "time only", "special")]
        f_s = results[(learner, "time + weather + events", "special")]
        if b_s["wape"] is not None and f_s["wape"] is not None:
            rel_special = 100.0 * (b_s["wape"] - f_s["wape"]) / b_s["wape"]

        meaningful = rel_all >= MEANINGFUL_OVERALL_PCT or (
            rel_special is not None and rel_special >= MEANINGFUL_SUBSET_PCT
        )
        if meaningful:
            print(f"    => MEANINGFUL for {learner}: the external signals clear the "
                  "stated threshold.")
        else:
            print(f"    => MARGINAL for {learner}: the gain does not clear the stated "
                  "threshold. Two honest")
            print("       reasons to expect exactly this: hist_avg already absorbs the "
                  "seasonal-average part")
            print("       of weather, and the ~11 km Open-Meteo grid gives 27 distinct "
                  "series for 225 zones,")
            print("       so weather cannot explain *spatial* differences at all.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-gbt", action="store_true",
                        help="linear learner only (fast run)")
    args = parser.parse_args()
    learners = ["linear"] if args.skip_gbt else LEARNERS

    if not Path(config.FEATURES_PARQUET).exists():
        print(f"ERROR: {config.FEATURES_PARQUET} missing — run: "
              "python -m src.batch.features", file=sys.stderr)
        return 2

    spark = get_spark("ablation")
    print("=" * 92)
    print("Feature ablation — weather & events vs trips-only (open question #2)")
    print("=" * 92)
    describe(spark)

    frame = spark.read.parquet(str(config.FEATURES_PARQUET))
    frame, train_mean_temp = prepare(frame)
    frame = frame.withColumn("pred_hist_avg", F.col("hist_avg_demand"))
    print(f"  train mean temperature : {train_mean_temp:.2f} C (centre for temp_dev)")

    # --- fit every (learner, feature set) ------------------------------------
    pred_columns: list[tuple[str, str, str]] = []  # (column, learner, set name)
    linear_full_model = None
    linear_full_features: list[str] = []
    for learner in learners:
        for set_name, features in FEATURE_SETS:
            slug = set_name.replace(" ", "").replace("+", "_")
            out_col = f"pred_{learner}_{slug}"
            if learner == "linear" and set_name == "time + weather + events":
                out_col = "pred_linear_full"  # referenced by the export below
            print(f"\n  fitting {learner:<7} [{set_name}] "
                  f"({len(features)} features) -> {out_col}")
            frame, model = fit_predict(frame, learner, features, out_col)
            pred_columns.append((out_col, learner, set_name))
            if out_col == "pred_linear_full":
                linear_full_model = model
                linear_full_features = features

    frame.cache()
    test = frame.filter(~F.col("is_train")).cache()
    n_test = test.count()

    ok = validate(test, [c for c, _, _ in pred_columns], n_test)

    # --- score ----------------------------------------------------------------
    subsets = {
        "all": test,
        "special": test.filter(F.col("is_special_day")),
        "rain": test.filter(F.col("precip_mm") > RAIN_THRESHOLD_MM),
    }
    results: dict[tuple[str, str, str], dict] = {}
    for subset_name, subset_frame in subsets.items():
        for column, learner, set_name in pred_columns:
            results[(learner, set_name, subset_name)] = metrics_for(
                subset_frame, column
            )
        results[("hist_avg", "baseline", subset_name)] = metrics_for(
            subset_frame, "pred_hist_avg"
        )

    for subset_name, subset_label in (
        ("all", f"FULL TEST GRID — N = {n_test:,}"),
        ("special", "SPECIAL DAYS ONLY (holiday or event)"),
        ("rain", f"RAIN HOURS ONLY (precip > {RAIN_THRESHOLD_MM:g} mm)"),
    ):
        rows = [("Historical avg (baseline, for context)",
                 results[("hist_avg", "baseline", subset_name)])]
        for learner in learners:
            for set_name, _ in FEATURE_SETS:
                rows.append(
                    (f"{learner}: {set_name}", results[(learner, set_name, subset_name)])
                )
        n_sub = rows[0][1]["n"]
        print_table(f"{subset_label}" + (f" — N = {n_sub:,}"
                                         if subset_name != "all" else ""), rows)

    # --- interpret the linear coefficients ------------------------------------
    if linear_full_model is not None:
        coef = dict(zip(linear_full_features, linear_full_model.coefficients))
        print("\n" + "=" * 92)
        print("LINEAR COEFFICIENTS — the external signals, in plain terms")
        print("=" * 92)
        print(f"  (interactions are per unit of hist_avg, i.e. fractional change "
              "of a cell's usual demand)")
        print(f"  precip_x_hist   {coef['precip_x_hist']:+.5f}  ->  each mm of rain "
              f"shifts demand by {100 * coef['precip_x_hist']:+.2f}% of usual")
        print(f"  temp_dev_x_hist {coef['temp_dev_x_hist']:+.5f}  ->  each degree C "
              f"from typical shifts it by {100 * coef['temp_dev_x_hist']:+.2f}%")
        print(f"  event_x_hist    {coef['event_x_hist']:+.5f}  ->  an event day "
              f"shifts demand by {100 * coef['event_x_hist']:+.2f}% of usual")
        print(f"  fedhol_x_hist   {coef['fedhol_x_hist']:+.5f}  ->  a federal holiday "
              f"shifts it by {100 * coef['fedhol_x_hist']:+.2f}%")
        print(f"  hist_avg_demand {coef['hist_avg_demand']:+.5f}  (the base ladder rung "
              "the model is built on)")

    verdict(results, learners)

    # --- export the portable conditions model ---------------------------------
    export_ok = True
    if linear_full_model is not None:
        intercept = float(linear_full_model.intercept)
        export_ok, worst = check_portable_model(
            test, linear_full_model, linear_full_features, intercept
        )

        # (zone, hour, dow) -> hist_avg — the portable model's base input.
        hist = (
            frame.select("zone_id", "hour_local", "dow", "hist_avg_demand")
            .distinct()
        )
        hist.coalesce(1).write.mode("overwrite").parquet(str(config.HIST_AVG_PARQUET))
        n_hist = hist.count()
        n_zones = frame.select("zone_id").distinct().count()
        hist_ok = n_hist == n_zones * config.HOURS_PER_WEEK
        print(f"  hist_avg export             : {n_hist:,} rows "
              f"(expected {n_zones} zones x {config.HOURS_PER_WEEK} = "
              f"{n_zones * config.HOURS_PER_WEEK:,})  "
              f"{'PASS' if hist_ok else 'FAIL'}")
        export_ok = export_ok and hist_ok

        # Monthly temperature normals: the default temperature for a query that
        # supplies none. Computed from the weather columns over the whole range —
        # weather is exogenous, so this is climatology, not target leakage.
        normals = {
            int(r["month"]): float(r["t"])
            for r in frame.groupBy(F.month("ts_local").alias("month"))
            .agg(F.avg("temp_c").alias("t"))
            .collect()
        }

        dates = frame.agg(
            F.min("date_local").alias("lo"), F.max("date_local").alias("hi")
        ).first()
        cutoff = (
            frame.filter(~F.col("is_train")).agg(F.min("date_local")).first()[0]
        )
        payload = {
            "learner": "linear regression (OLS, normal solver), Spark MLlib",
            "feature_set": "time + weather + events",
            "target": "trip_count per (zone, hour)",
            "clamp_at_zero": True,
            "intercept": intercept,
            "coefficients": {
                name: float(c)
                for name, c in zip(linear_full_features, linear_full_model.coefficients)
            },
            "train_mean_temp_c": train_mean_temp,
            "monthly_temp_normals_c": {str(m): normals[m] for m in sorted(normals)},
            "default_precip_mm": 0.0,
            "fitted_on": {
                "start": dates["lo"], "end": dates["hi"], "test_from": cutoff,
                "train_rows": int(frame.filter(F.col("is_train")).count()),
            },
            "test_metrics": {
                "this_model": results[("linear", "time + weather + events", "all")],
                "time_only": results[("linear", "time only", "all")],
                "hist_avg_baseline": results[("hist_avg", "baseline", "all")],
            },
            "spark_version": spark.version,
            "seed": config.SEED,
            "portable_parity_max_abs_diff": worst,
        }
        config.CONDITIONS_MODEL_JSON.write_text(json.dumps(payload, indent=2))
        print(f"  wrote : {config.CONDITIONS_MODEL_JSON}")
        print(f"  wrote : {config.HIST_AVG_PARQUET}")

    # --- csv -------------------------------------------------------------------
    with config.ABLATION_METRICS_CSV.open("w", encoding="utf-8") as handle:
        handle.write("learner,feature_set,subset,n,mae,rmse,wape,mape_nonzero,mean_pred\n")
        for (learner, set_name, subset_name), m in sorted(results.items()):
            wape = "" if m["wape"] is None else f"{m['wape']:.6f}"
            mape = "" if m["mape_nonzero"] is None else f"{m['mape_nonzero']:.6f}"
            handle.write(
                f"{learner},\"{set_name}\",{subset_name},{m['n']},{m['mae']:.6f},"
                f"{m['rmse']:.6f},{wape},{mape},{m['mean_pred']:.6f}\n"
            )
    print(f"  wrote : {config.ABLATION_METRICS_CSV}")

    ok = ok and export_ok
    print("\n" + "=" * 92)
    print("DONE")
    print(f"  RESULT : {'PASS' if ok else 'FAIL'}")
    print("=" * 92)

    spark.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
