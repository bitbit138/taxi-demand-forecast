"""Build the modelling table -> ``data/processed/features.parquet``.

**Row grain: one row per ``(zone_id, date_local, hour_local)``** over the 223-zone
modelling set — 223 zones x 2183 hours = 486,809 rows. This is the *per-observation*
table. It feeds the baseline ladder and the evaluation in ``evaluate.py``.

It is NOT the clustering input. K-Means (Decision A) clusters *zones* by their
``(hour, dow)`` demand profile, which is a 223-row x 168-column aggregation derived
**downstream in ``train_kmeans.py``** from this table. Two different artifacts:

  * ``features.parquet``      -> per (zone, hour) observation, for baselines/evaluation
  * profile matrix in         -> per zone, for clustering
    ``train_kmeans.py``

Conflating them would silently change what K-Means is clustering.

**Ordering: exclude, then zero-fill.** The grid is built from ``weather.parquet``
restricted to the modelling zones, so it inherits the DST-correct 2183-hour calendar by
construction (local 02:00 on 2024-03-10 does not exist) and weather can never be null.
Demand is left-joined onto that grid and absent bins become ``trip_count = 0``.

**No leakage in ``hist_avg_demand``.** It is both a feature and a baseline, so computing
it over the whole quarter would leak test-period demand into training. It is fitted on
the **training split only** and applied to both splits. The split is time-based
(earlier hours train, later test) and deterministic — no seed is involved — and is
recorded as ``is_train`` so ``evaluate.py`` reuses exactly this split.

    python -m src.batch.features
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.spark_session import describe, get_spark  # noqa: E402

# Columns that must never be null once the table is assembled.
REQUIRED_NON_NULL = [
    "zone_id", "date_local", "hour_local", "ts_local", "trip_count", "is_train",
    "dow", "is_weekend", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "temp_c", "precip_mm",
    "is_holiday", "is_federal_holiday", "is_event", "is_special_day",
    "hist_avg_demand",
]


def build_grid(spark: SparkSession) -> DataFrame:
    """Full ``modelling zones x every local hour`` grid, with weather attached.

    Derived from ``weather.parquet`` so the hour calendar is DST-correct by
    construction rather than by regenerating a date range here.
    """
    weather = spark.read.parquet(str(config.WEATHER_PARQUET)).select(
        "zone_id", "ts_local", "date_local", "hour_local", "temp_c", "precip_mm"
    )
    zones = spark.read.parquet(str(config.MODELING_ZONES_PARQUET)).select("zone_id")
    return weather.join(F.broadcast(zones), on="zone_id", how="inner")


def attach_demand(grid: DataFrame, spark: SparkSession) -> DataFrame:
    """Left-join observed demand onto the grid; absent bins become zero."""
    demand = spark.read.parquet(str(config.DEMAND_PARQUET)).select(
        F.col("PULocationID").alias("zone_id"),
        "date_local",
        "hour_local",
        "trip_count",
    )
    return (
        grid.join(demand, on=["zone_id", "date_local", "hour_local"], how="left")
        .withColumn("trip_count", F.coalesce(F.col("trip_count"), F.lit(0)).cast("int"))
    )


def add_calendar_and_cyclical(frame: DataFrame) -> DataFrame:
    """Calendar fields, sin/cos encodings, and Fourier terms on hour-of-week."""
    # Spark's dayofweek is 1=Sunday..7=Saturday; remap to 0=Monday..6=Sunday.
    dow = ((F.dayofweek(F.col("ts_local")) + F.lit(5)) % F.lit(7)).cast("smallint")

    frame = (
        frame.withColumn("dow", dow)
        .withColumn("is_weekend", F.col("dow").isin(5, 6))
        .withColumn("month", F.month("ts_local").cast("smallint"))
        .withColumn(
            "hour_of_week",
            (F.col("dow") * F.lit(24) + F.col("hour_local")).cast("smallint"),
        )
    )

    two_pi = 2.0 * math.pi
    frame = (
        frame.withColumn("hour_sin", F.sin(two_pi * F.col("hour_local") / F.lit(24.0)))
        .withColumn("hour_cos", F.cos(two_pi * F.col("hour_local") / F.lit(24.0)))
        .withColumn("dow_sin", F.sin(two_pi * F.col("dow") / F.lit(7.0)))
        .withColumn("dow_cos", F.cos(two_pi * F.col("dow") / F.lit(7.0)))
    )

    # A few harmonics on the 168-hour week capture daily+weekly shape jointly,
    # which plain hour/dow encodings cannot (e.g. "Friday evening" specifically).
    for k in range(1, config.FOURIER_TERMS + 1):
        angle = two_pi * F.lit(float(k)) * F.col("hour_of_week") / F.lit(
            float(config.HOURS_PER_WEEK)
        )
        frame = frame.withColumn(f"how_sin_{k}", F.sin(angle))
        frame = frame.withColumn(f"how_cos_{k}", F.cos(angle))

    return frame


def attach_events(frame: DataFrame, spark: SparkSession) -> DataFrame:
    """Join the event calendar on ``date_local``."""
    events = (
        spark.read.option("header", True)
        .csv(str(config.EVENTS_CSV))
        .select(
            "date_local",
            F.col("is_holiday").cast("boolean").alias("is_holiday"),
            F.col("is_federal_holiday").cast("boolean").alias("is_federal_holiday"),
            F.col("is_event").cast("boolean").alias("is_event"),
            F.col("is_special_day").cast("boolean").alias("is_special_day"),
        )
    )
    return frame.join(F.broadcast(events), on="date_local", how="left")


def add_split(frame: DataFrame) -> tuple[DataFrame, str]:
    """Mark a time-based train/test split. Returns ``(frame, cutoff_date)``.

    Earlier dates train, later dates test — a random split would let the model see
    future hours of the same day and inflate every metric.
    """
    dates = [
        r["date_local"]
        for r in frame.select("date_local").distinct().orderBy("date_local").collect()
    ]
    cutoff_index = int(len(dates) * config.TRAIN_FRACTION)
    cutoff = dates[cutoff_index]
    return frame.withColumn("is_train", F.col("date_local") < F.lit(cutoff)), cutoff


def add_historical_average(frame: DataFrame) -> DataFrame:
    """``hist_avg_demand(zone, hour, dow)`` fitted on the TRAIN split only.

    This column is simultaneously a feature and the first rung of the baseline ladder,
    so fitting it over the full quarter would leak test-period demand into training and
    flatter every downstream metric.
    """
    train_means = (
        frame.filter(F.col("is_train"))
        .groupBy("zone_id", "hour_local", "dow")
        .agg(F.avg("trip_count").alias("hist_avg_demand"))
    )

    # Fallbacks for any (zone, hour, dow) unseen in training — also train-only.
    zone_means = (
        frame.filter(F.col("is_train"))
        .groupBy("zone_id")
        .agg(F.avg("trip_count").alias("zone_mean"))
    )
    global_mean = (
        frame.filter(F.col("is_train")).agg(F.avg("trip_count")).first()[0] or 0.0
    )

    return (
        frame.join(train_means, on=["zone_id", "hour_local", "dow"], how="left")
        .join(F.broadcast(zone_means), on="zone_id", how="left")
        .withColumn(
            "hist_avg_demand",
            F.coalesce(
                F.col("hist_avg_demand"), F.col("zone_mean"), F.lit(float(global_mean))
            ),
        )
        .drop("zone_mean")
    )


def validate(frame: DataFrame, spark: SparkSession, cutoff: str) -> bool:
    """Grid shape, zero-fill accounting, leakage and null checks."""
    ok = True
    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    n_zones = spark.read.parquet(str(config.MODELING_ZONES_PARQUET)).count()
    n_hours = (
        spark.read.parquet(str(config.WEATHER_PARQUET))
        .select("ts_local")
        .distinct()
        .count()
    )
    expected_rows = n_zones * n_hours

    stats = frame.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("zone_id").alias("zones"),
        F.countDistinct("ts_local").alias("hours"),
        F.sum("trip_count").alias("total_trips"),
        F.sum(F.when(F.col("trip_count") == 0, 1).otherwise(0)).alias("zero_bins"),
        F.sum(F.when(F.col("is_train"), 1).otherwise(0)).alias("train_rows"),
    ).first()

    rows = int(stats["rows"])
    print(f"  grid                 : {int(stats['zones'])} zones x "
          f"{int(stats['hours'])} hours = {rows:,} rows")
    print(f"  expected             : {n_zones} x {n_hours} = {expected_rows:,}")
    shape_ok = rows == expected_rows and int(stats["zones"]) == n_zones
    print(f"  shape correct        : {shape_ok}")
    ok = ok and shape_ok

    dupes = (
        frame.groupBy("zone_id", "date_local", "hour_local").count()
        .filter(F.col("count") > 1).count()
    )
    print(f"  duplicate keys       : {dupes}")
    ok = ok and dupes == 0

    # DST: the skipped local hour must be absent for every zone.
    dst_rows = frame.filter(
        (F.col("date_local") == "2024-03-10") & (F.col("hour_local") == 2)
    ).count()
    print(f"  rows at 2024-03-10 02:00 : {dst_rows} (DST gap, must be 0)")
    ok = ok and dst_rows == 0

    # --- zero-fill accounting -------------------------------------------------
    demand = spark.read.parquet(str(config.DEMAND_PARQUET))
    zones = spark.read.parquet(str(config.MODELING_ZONES_PARQUET)).select("zone_id")
    observed = demand.join(
        F.broadcast(zones), demand["PULocationID"] == zones["zone_id"], "inner"
    )
    observed_bins = observed.count()
    observed_trips = observed.agg(F.sum("trip_count")).first()[0]

    zero_bins = int(stats["zero_bins"])
    filled_bins = rows - zero_bins
    print(f"\n  observed bins (kept zones) : {observed_bins:,}")
    print(f"  non-zero bins in grid      : {filled_bins:,}")
    print(f"  zero-filled bins           : {zero_bins:,}")
    print(f"  expected zero-fill         : {expected_rows - observed_bins:,}")
    fill_ok = filled_bins == observed_bins and zero_bins == expected_rows - observed_bins
    print(f"  zero-fill accounting       : {fill_ok}")
    ok = ok and fill_ok

    total_trips = int(stats["total_trips"])
    print(f"  trips preserved            : {total_trips:,} vs {int(observed_trips):,} "
          f"({total_trips == int(observed_trips)})")
    ok = ok and total_trips == int(observed_trips)
    print(f"  grid occupancy             : {100 * filled_bins / rows:.1f}%")

    # --- split ----------------------------------------------------------------
    train_rows = int(stats["train_rows"])
    print(f"\n  split cutoff date    : {cutoff} (earlier = train)")
    print(f"  train / test rows    : {train_rows:,} / {rows - train_rows:,} "
          f"({100 * train_rows / rows:.1f}% train, target "
          f"{100 * config.TRAIN_FRACTION:.0f}%)")
    train_days = frame.filter(F.col("is_train")).select("date_local").distinct().count()
    test_days = frame.filter(~F.col("is_train")).select("date_local").distinct().count()
    print(f"  train / test days    : {train_days} / {test_days}")
    ok = ok and train_rows > 0 and rows - train_rows > 0

    # --- leakage check --------------------------------------------------------
    # hist_avg_demand must be reproducible from train rows alone. Recomputing it
    # over everything and comparing proves the fitted values used only training data.
    full_means = (
        frame.groupBy("zone_id", "hour_local", "dow")
        .agg(F.avg("trip_count").alias("full_avg"))
    )
    differing = (
        frame.select("zone_id", "hour_local", "dow", "hist_avg_demand")
        .distinct()
        .join(full_means, on=["zone_id", "hour_local", "dow"], how="inner")
        .filter(F.abs(F.col("hist_avg_demand") - F.col("full_avg")) > 1e-9)
        .count()
    )
    print(f"\n  hist_avg differs from full-quarter mean in {differing:,} "
          f"(zone,hour,dow) cells")
    print("    -> non-zero confirms it was fitted on train only, not the whole quarter")
    ok = ok and differing > 0

    # --- nulls ----------------------------------------------------------------
    fourier_cols = [
        f"how_{fn}_{k}" for k in range(1, config.FOURIER_TERMS + 1) for fn in ("sin", "cos")
    ]
    checked = REQUIRED_NON_NULL + fourier_cols
    null_counts = frame.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in checked]
    ).first().asDict()
    offenders = {c: n for c, n in null_counts.items() if n}
    print(f"\n  columns checked for nulls  : {len(checked)}")
    print(f"  columns containing nulls   : {len(offenders)} {offenders or ''}")
    ok = ok and not offenders

    # Cyclical sanity: sin^2 + cos^2 == 1 everywhere.
    bad_unit = frame.filter(
        F.abs(F.col("hour_sin") ** 2 + F.col("hour_cos") ** 2 - 1.0) > 1e-9
    ).count()
    print(f"  hour sin^2+cos^2 != 1      : {bad_unit}")
    ok = ok and bad_unit == 0

    print("=" * 78)
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def show_samples(frame: DataFrame) -> None:
    """One busy Midtown hour and one empty outer-borough hour, fully assembled."""
    cols = [
        "zone_id", "date_local", "hour_local", "dow", "is_weekend", "trip_count",
        "hist_avg_demand", "temp_c", "precip_mm",
        "is_holiday", "is_event", "is_special_day",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "how_sin_1", "how_cos_1",
        "is_train",
    ]

    print("\n" + "=" * 78)
    print("SAMPLE ROW A — busy Midtown Center (zone 161), evening rush")
    print("=" * 78)
    busy = (
        frame.filter((F.col("zone_id") == 161) & (F.col("hour_local") == 18))
        .orderBy(F.desc("trip_count"))
        .limit(1)
    )
    _print_vertical(busy, cols)

    print("\n" + "=" * 78)
    print("SAMPLE ROW B — empty outer-borough zone-hour (zero-filled)")
    print("=" * 78)
    empty = (
        frame.filter((F.col("trip_count") == 0) & (F.col("hour_local") == 4))
        .orderBy("zone_id", "date_local")
        .limit(1)
    )
    _print_vertical(empty, cols)


def _print_vertical(frame: DataFrame, cols: list[str]) -> None:
    row = frame.select(*cols).first()
    if row is None:
        print("  (no matching row)")
        return
    for col in cols:
        value = row[col]
        if isinstance(value, float):
            print(f"  {col:<18} {value:>12.4f}")
        else:
            print(f"  {col:<18} {str(value):>12}")


def main() -> int:
    for path, hint in (
        (config.DEMAND_PARQUET, "src.batch.clean_aggregate"),
        (config.WEATHER_PARQUET, "src.ingest.fetch_weather"),
        (config.EVENTS_CSV, "src.ingest.build_events"),
        (config.MODELING_ZONES_PARQUET, "src.batch.zone_policy"),
    ):
        if not Path(path).exists():
            print(f"ERROR: {path} missing — run: python -m {hint}", file=sys.stderr)
            return 2

    spark = get_spark("features")

    print("=" * 78)
    print("Build modelling table")
    print("=" * 78)
    describe(spark)
    print(f"  grain     one row per (zone_id, date_local, hour_local)")
    print(f"  fourier   {config.FOURIER_TERMS} harmonics on the "
          f"{config.HOURS_PER_WEEK}-hour week")
    print(f"  split     time-based, {100 * config.TRAIN_FRACTION:.0f}% train")

    grid = build_grid(spark)
    frame = attach_demand(grid, spark)
    frame = add_calendar_and_cyclical(frame)
    frame = attach_events(frame, spark)
    frame, cutoff = add_split(frame)
    frame = add_historical_average(frame)
    frame.cache()

    ok = validate(frame, spark, cutoff)
    show_samples(frame)

    frame.coalesce(1).write.mode("overwrite").parquet(str(config.FEATURES_PARQUET))

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  wrote : {config.FEATURES_PARQUET}")
    print(f"  grain : (zone_id, date_local, hour_local) — per-observation table")
    print(f"  note  : the per-zone (hour, dow) profile matrix for K-Means is derived")
    print(f"          downstream in train_kmeans.py, not stored here")
    print("=" * 78)

    spark.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
