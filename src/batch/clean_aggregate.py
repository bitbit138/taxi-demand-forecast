"""Clean raw TLC trips and aggregate to hourly demand per zone.

``data/raw/*.parquet`` -> ``data/processed/demand.parquet``, keyed
``(PULocationID, date_local, hour_local) -> trip_count``.

**Timezone.** TLC parquet stores ``tpep_pickup_datetime`` as
``Timestamp(isAdjustedToUTC=false)`` — naive local NYC wall-clock, no zone attached.
The session timezone is UTC (see ``src/spark_session.py``), so those values read back
byte-for-byte unshifted and ``date()``/``hour()`` yield the NY-local calendar date and
hour directly. No conversion is applied because none is needed; applying one would
shift every trip by five hours. ``assert_local_wallclock()`` checks this empirically
against the known NYC demand curve rather than trusting the reasoning.

**Empty bins are NOT zero-filled here.** A ``(zone, hour)`` with no trips is simply
absent from the output — this file reports observed demand only. Zero-fill happens in
``features.py``, per PROJECT_PLAN.md, where the full zone x hour grid is constructed
before feature building. Keeping it there means one place owns the grid.

**Schema drift** is handled with an explicit column select + cast (never
``mergeSchema``): 2025+ files add ``cbd_congestion_fee`` and casing varies by year.

    python -m src.batch.clean_aggregate
    python -m src.batch.clean_aggregate --force   # write despite a funnel warning
"""

from __future__ import annotations

import argparse
import sys
from functools import reduce
from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.spark_session import describe, get_spark  # noqa: E402

# Explicit cast targets — the raw files vary (passenger_count is int64 in 2024,
# double in other years), so every column is pinned rather than inferred.
COLUMN_TYPES: dict[str, T.DataType] = {
    "tpep_pickup_datetime": T.TimestampType(),
    "tpep_dropoff_datetime": T.TimestampType(),
    "PULocationID": T.IntegerType(),
    "DOLocationID": T.IntegerType(),
    "passenger_count": T.IntegerType(),
    "trip_distance": T.DoubleType(),
    "fare_amount": T.DoubleType(),
    "total_amount": T.DoubleType(),
}


def read_raw(spark: SparkSession) -> DataFrame:
    """Read the configured monthly files with an explicit column select + cast."""
    paths = [str(config.RAW_DIR / f"yellow_tripdata_{m}.parquet") for m in config.MONTHS]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files (run: python -m src.ingest.download_tlc):\n  "
            + "\n  ".join(missing)
        )

    raw = spark.read.parquet(*paths)

    absent = [c for c in config.TLC_COLUMNS if c not in raw.columns]
    if absent:
        raise ValueError(
            f"Columns absent from the raw files: {absent}\n"
            f"Available: {sorted(raw.columns)}"
        )

    return raw.select(
        *[F.col(c).cast(COLUMN_TYPES[c]).alias(c) for c in config.TLC_COLUMNS]
    )


def build_filters() -> list[tuple[str, Column]]:
    """Ordered cleaning predicates. Each is applied on top of the previous ones."""
    pickup = F.col("tpep_pickup_datetime")
    dropoff = F.col("tpep_dropoff_datetime")
    duration_min = (dropoff.cast("long") - pickup.cast("long")) / 60.0

    # Inclusive local-date bounds for the configured range.
    start = F.lit(f"{config.START_DATE} 00:00:00").cast("timestamp")
    end = F.lit(f"{config.END_DATE} 23:59:59").cast("timestamp")

    return [
        ("timestamps present", pickup.isNotNull() & dropoff.isNotNull()),
        ("pickup zone present", F.col("PULocationID").isNotNull()),
        (
            f"zone id in {config.VALID_ZONE_MIN}..{config.VALID_ZONE_MAX + 2}",
            F.col("PULocationID").between(config.VALID_ZONE_MIN, config.VALID_ZONE_MAX + 2),
        ),
        (
            f"drop unknown zones {config.EXCLUDED_ZONE_IDS}",
            ~F.col("PULocationID").isin(config.EXCLUDED_ZONE_IDS),
        ),
        ("pickup < dropoff", pickup < dropoff),
        (
            f"duration <= {config.MAX_TRIP_DURATION_MINUTES / 60:.0f}h",
            duration_min <= config.MAX_TRIP_DURATION_MINUTES,
        ),
        (
            f"distance in ({config.MIN_TRIP_DISTANCE}, {config.MAX_TRIP_DISTANCE}]",
            (F.col("trip_distance") > config.MIN_TRIP_DISTANCE)
            & (F.col("trip_distance") <= config.MAX_TRIP_DISTANCE),
        ),
        (
            f"fare in ({config.MIN_FARE_AMOUNT}, {config.MAX_FARE_AMOUNT}]",
            (F.col("fare_amount") > config.MIN_FARE_AMOUNT)
            & (F.col("fare_amount") <= config.MAX_FARE_AMOUNT),
        ),
        (
            # Null passenger_count is common and says nothing about whether the trip
            # happened. The target is a trip *count*, so a missing passenger figure is
            # not a reason to discard real demand — only an out-of-range value is.
            f"passenger_count null or {config.MIN_PASSENGER_COUNT}"
            f"..{config.MAX_PASSENGER_COUNT}",
            F.col("passenger_count").isNull()
            | F.col("passenger_count").between(
                config.MIN_PASSENGER_COUNT, config.MAX_PASSENGER_COUNT
            ),
        ),
        (
            f"pickup within {config.START_DATE}..{config.END_DATE}",
            pickup.between(start, end),
        ),
    ]


def funnel(trips: DataFrame, filters: list[tuple[str, Column]]) -> tuple[list[dict], int, int]:
    """Rows surviving each cumulative filter stage, computed in a single pass.

    Returns ``(stages, rows_in, rows_out)``. Nothing is collected except this one
    aggregate row — the trip data itself never leaves the cluster.
    """
    cumulative: list[Column] = []
    running: Column | None = None
    for _, predicate in filters:
        running = predicate if running is None else running & predicate
        cumulative.append(running)

    row = trips.agg(
        F.count(F.lit(1)).alias("rows_in"),
        *[
            F.sum(F.when(cond, 1).otherwise(0)).alias(f"stage_{i}")
            for i, cond in enumerate(cumulative)
        ],
    ).first()

    rows_in = int(row["rows_in"])
    stages, previous = [], rows_in
    for i, (name, _) in enumerate(filters):
        remaining = int(row[f"stage_{i}"])
        dropped = previous - remaining
        stages.append(
            {
                "name": name,
                "dropped": dropped,
                "remaining": remaining,
                "pct_of_input": dropped / rows_in if rows_in else 0.0,
            }
        )
        previous = remaining
    return stages, rows_in, previous


def print_funnel(stages: list[dict], rows_in: int, rows_out: int) -> list[dict]:
    """Print the funnel table. Returns stages breaching the alert threshold."""
    print("\n" + "=" * 78)
    print("CLEANING FUNNEL")
    print("=" * 78)
    print(f"{'filter':<44}{'dropped':>12}{'% in':>8}{'remaining':>14}")
    print("-" * 78)
    print(f"{'rows in':<44}{'':>12}{'':>8}{rows_in:>14,}")
    for stage in stages:
        flag = "  <-- HIGH" if stage["pct_of_input"] > config.FUNNEL_ALERT_FRACTION else ""
        print(
            f"{stage['name']:<44}{stage['dropped']:>12,}"
            f"{100 * stage['pct_of_input']:>7.2f}%{stage['remaining']:>14,}{flag}"
        )
    print("-" * 78)
    total_dropped = rows_in - rows_out
    print(f"{'rows out':<44}{total_dropped:>12,}"
          f"{100 * total_dropped / rows_in:>7.2f}%{rows_out:>14,}")
    print("=" * 78)
    return [s for s in stages if s["pct_of_input"] > config.FUNNEL_ALERT_FRACTION]


def assert_local_wallclock(demand: DataFrame) -> None:
    """Empirically confirm hours are NY-local, not UTC-shifted.

    NYC yellow-taxi demand peaks in the evening rush (roughly 17:00-19:00 local) and
    bottoms out pre-dawn (03:00-05:00). Read as UTC the whole curve would slide five
    hours and the peak would land near midnight, so this is a real check on the
    timezone assumption rather than a restatement of it.
    """
    by_hour = (
        demand.groupBy("hour_local")
        .agg(F.sum("trip_count").alias("trips"))
        .orderBy(F.desc("trips"))
        .limit(1)
        .first()
    )
    trough = (
        demand.groupBy("hour_local")
        .agg(F.sum("trip_count").alias("trips"))
        .orderBy("trips")
        .limit(1)
        .first()
    )
    peak_hour, quiet_hour = int(by_hour["hour_local"]), int(trough["hour_local"])
    print(f"  busiest hour of day  : {peak_hour:02d}:00 local "
          f"({int(by_hour['trips']):,} trips)")
    print(f"  quietest hour of day : {quiet_hour:02d}:00 local "
          f"({int(trough['trips']):,} trips)")

    if not (15 <= peak_hour <= 20 and 2 <= quiet_hour <= 6):
        raise ValueError(
            f"Demand curve looks shifted: peak {peak_hour:02d}:00, "
            f"trough {quiet_hour:02d}:00. Expected an evening peak (15-20) and a "
            "pre-dawn trough (02-06). Check the session timezone."
        )
    print("  -> curve matches NY-local expectations (evening peak, pre-dawn trough)")


def local_date_hour(timestamp: Column) -> tuple[Column, Column]:
    """``(date_local, hour_local)`` from a naive NY-local wall-clock column.

    Shared with ``src/stream/spark_stream.py`` so the streaming windows land on
    exactly the same calendar cells as the batch aggregation. Naive wall-clock in,
    NY-local date and hour out, with no conversion — see the module docstring.
    """
    return (
        F.date_format(timestamp, "yyyy-MM-dd").alias("date_local"),
        F.hour(timestamp).cast("smallint").alias("hour_local"),
    )


def aggregate(trips: DataFrame) -> DataFrame:
    """Aggregate surviving trips to hourly demand per pickup zone."""
    date_local, hour_local = local_date_hour(F.col("tpep_pickup_datetime"))
    return (
        trips.select(F.col("PULocationID"), date_local, hour_local)
        .groupBy("PULocationID", "date_local", "hour_local")
        .agg(F.count(F.lit(1)).cast("int").alias("trip_count"))
    )


def validate(demand: DataFrame, rows_out: int) -> bool:
    """Conservation, zone coverage and shape checks."""
    ok = True
    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    totals = demand.agg(
        F.sum("trip_count").alias("total"),
        F.count(F.lit(1)).alias("bins"),
        F.countDistinct("PULocationID").alias("zones"),
        F.countDistinct("date_local").alias("days"),
        F.min("trip_count").alias("min_count"),
        F.max("trip_count").alias("max_count"),
    ).first()

    total = int(totals["total"])
    print(f"  surviving trips      : {rows_out:,}")
    print(f"  sum(trip_count)      : {total:,}")
    conserved = total == rows_out
    print(f"  conserved            : {conserved}  "
          f"({'no trips lost or double-counted' if conserved else 'MISMATCH'})")
    ok = ok and conserved

    print(f"  non-empty zone-hours : {int(totals['bins']):,}")
    print(f"  distinct zones       : {int(totals['zones'])}")
    print(f"  distinct days        : {int(totals['days'])}")
    print(f"  trip_count range     : {int(totals['min_count'])} .. {int(totals['max_count'])}")

    if int(totals["min_count"]) < 1:
        ok = False
        print("    ERROR: a bin has a non-positive count")

    excluded = demand.filter(F.col("PULocationID").isin(config.EXCLUDED_ZONE_IDS)).count()
    print(f"  rows in zones {config.EXCLUDED_ZONE_IDS} : {excluded}")
    ok = ok and excluded == 0

    dupes = (
        demand.groupBy("PULocationID", "date_local", "hour_local")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    print(f"  duplicate keys       : {dupes}")
    ok = ok and dupes == 0

    print()
    assert_local_wallclock(demand)

    # Empty-bin accounting: how sparse is the observed grid?
    expected_bins = int(totals["zones"]) * int(totals["days"]) * 24
    filled = int(totals["bins"])
    print(f"\n  grid occupancy       : {filled:,} / {expected_bins:,} "
          f"({100 * filled / expected_bins:.1f}%) — the remaining "
          f"{expected_bins - filled:,} empty bins are zero-filled in features.py, "
          "not here")

    print("=" * 78)
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def show_extremes(demand: DataFrame) -> None:
    """Busiest and quietest zone-hours, for eyeballing the shape."""
    lookup_exists = config.ZONE_LOOKUP_CSV.exists()
    print("\nBusiest zone-hours:")
    demand.orderBy(F.desc("trip_count")).limit(5).show(truncate=False)

    print("Quietest zone-hours (ties broken by zone then time):")
    (
        demand.orderBy("trip_count", "PULocationID", "date_local", "hour_local")
        .limit(5)
        .show(truncate=False)
    )

    print("Busiest zones overall:")
    top = (
        demand.groupBy("PULocationID")
        .agg(F.sum("trip_count").alias("total_trips"))
        .orderBy(F.desc("total_trips"))
        .limit(5)
    )
    if lookup_exists:
        spark = demand.sparkSession
        names = (
            spark.read.option("header", True)
            .csv(str(config.ZONE_LOOKUP_CSV))
            .select(
                F.col("LocationID").cast("int").alias("PULocationID"),
                F.col("Zone").alias("zone_name"),
                F.col("Borough").alias("borough"),
            )
        )
        top = top.join(names, on="PULocationID", how="left").orderBy(F.desc("total_trips"))
    top.show(truncate=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true",
        help="write output even if a filter exceeds the alert threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spark = get_spark("clean-aggregate")

    print("=" * 78)
    print("Clean + aggregate TLC trips")
    print("=" * 78)
    describe(spark)
    print(f"  months  {', '.join(config.MONTHS)}")
    print(f"  columns {', '.join(config.TLC_COLUMNS)}")

    trips = read_raw(spark)
    # Read once, reuse for the funnel pass and the aggregation pass.
    trips.cache()

    filters = build_filters()
    stages, rows_in, rows_out = funnel(trips, filters)
    breaches = print_funnel(stages, rows_in, rows_out)

    if breaches and not args.force:
        print("\n" + "!" * 78)
        print("STOPPED — a filter dropped more than "
              f"{100 * config.FUNNEL_ALERT_FRACTION:.0f}% of rows:")
        for stage in breaches:
            print(f"  {stage['name']}: {stage['dropped']:,} rows "
                  f"({100 * stage['pct_of_input']:.2f}%)")
        print("\nNothing was written. Review the filter, then re-run with --force")
        print("to accept it as-is.")
        print("!" * 78)
        spark.stop()
        return 3

    clean = trips.filter(reduce(lambda a, b: a & b, (cond for _, cond in filters)))
    demand = aggregate(clean)
    demand.cache()

    ok = validate(demand, rows_out)
    show_extremes(demand)

    demand.coalesce(1).write.mode("overwrite").parquet(str(config.DEMAND_PARQUET))

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  wrote : {config.DEMAND_PARQUET}")
    print(f"  key   : (PULocationID, date_local, hour_local) -> trip_count")
    print(f"  note  : observed demand only; empty bins zero-filled in features.py")
    print("=" * 78)

    spark.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
