"""Owns the zero-fill and zone-exclusion policy for the modelling set.

One module decides which zones are modelled so that ``features.py``,
``train_kmeans.py``, ``evaluate.py`` and ``make_maps.py`` cannot drift apart.

**Zero-fill** (``config.ZERO_FILL_DEMAND``): ``demand.parquet`` holds observed demand
only — a ``(zone, hour)`` with no trips is absent, not zero. ``features.py`` builds the
full ``kept zones x every hour`` grid and fills the gaps with ``trip_count = 0``, so each
zone's series is continuous before lag/rolling features are computed. The zero-fill is
applied *after* exclusion, so dropped zones never enter the grid.

**Exclusion** (``config.MIN_ZONE_TRIPS_PER_DAY``): a zone averaging under one trip per
day is ~99% zeros across the quarter. Its feature vector is essentially the zero vector,
so K-Means would spend a cluster telling "no demand" apart from everything else and the
silhouette score would be flattered by a trivially separable group. The floor is
expressed per day so it scales unchanged to the full year.

The two rules are deliberately paired: zero-filling *without* an exclusion floor would
manufacture tens of thousands of all-zero rows for zones that have no taxi service at
all (Governor's Island has no road access), which is exactly the degenerate cluster the
floor exists to prevent.

    python -m src.batch.zone_policy        # print the decision and write the zone list
"""

from __future__ import annotations

import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.spark_session import get_spark  # noqa: E402


def all_valid_zones(spark: SparkSession) -> DataFrame:
    """Every real taxi zone (1..263) with its name and borough."""
    return (
        spark.read.option("header", True)
        .csv(str(config.ZONE_LOOKUP_CSV))
        .select(
            F.col("LocationID").cast("int").alias("zone_id"),
            F.col("Zone").alias("zone_name"),
            F.col("Borough").alias("borough"),
        )
        .filter(
            F.col("zone_id").between(config.VALID_ZONE_MIN, config.VALID_ZONE_MAX)
            & ~F.col("zone_id").isin(config.EXCLUDED_ZONE_IDS)
        )
    )


def zone_totals(spark: SparkSession) -> tuple[DataFrame, int]:
    """Per-zone trip totals over the range. Returns ``(frame, n_days)``.

    Zones with no trips at all are absent from ``demand.parquet``, so the totals are
    right-joined onto the full zone list and filled with zero — otherwise they would
    silently escape the floor by not existing.
    """
    demand = spark.read.parquet(str(config.DEMAND_PARQUET))
    n_days = demand.select(F.countDistinct("date_local")).first()[0]

    observed = demand.groupBy("PULocationID").agg(
        F.sum("trip_count").cast("long").alias("total_trips"),
        F.countDistinct("date_local", "hour_local").alias("non_empty_bins"),
    )

    totals = (
        all_valid_zones(spark)
        .join(observed, F.col("zone_id") == observed["PULocationID"], "left")
        .drop("PULocationID")
        .fillna({"total_trips": 0, "non_empty_bins": 0})
        .withColumn("trips_per_day", F.col("total_trips") / F.lit(n_days))
    )
    return totals, n_days


def split_zones(spark: SparkSession) -> tuple[DataFrame, DataFrame, int]:
    """Apply the floor. Returns ``(kept, excluded, n_days)``."""
    totals, n_days = zone_totals(spark)
    keep = F.col("trips_per_day") >= F.lit(config.MIN_ZONE_TRIPS_PER_DAY)
    return totals.filter(keep), totals.filter(~keep), n_days


def modeling_zone_ids(spark: SparkSession) -> list[int]:
    """Zone ids in the modelling set, read from the persisted list when available."""
    if config.MODELING_ZONES_PARQUET.exists():
        frame = spark.read.parquet(str(config.MODELING_ZONES_PARQUET))
    else:
        frame, _, _ = split_zones(spark)
    return [int(r["zone_id"]) for r in frame.select("zone_id").orderBy("zone_id").collect()]


def report(kept: DataFrame, excluded: DataFrame, n_days: int) -> None:
    """Print the exclusion decision in full — it must never be implicit."""
    floor = config.MIN_ZONE_TRIPS_PER_DAY * n_days
    n_kept, n_excluded = kept.count(), excluded.count()

    kept_trips = kept.agg(F.sum("total_trips")).first()[0] or 0
    lost_trips = excluded.agg(F.sum("total_trips")).first()[0] or 0
    all_trips = kept_trips + lost_trips

    print("\n" + "=" * 78)
    print("ZONE POLICY")
    print("=" * 78)
    print(f"  zero-fill              : {config.ZERO_FILL_DEMAND} "
          f"(applied in features.py, after exclusion)")
    print(f"  exclusion floor        : {config.MIN_ZONE_TRIPS_PER_DAY} trips/day "
          f"x {n_days} days = {floor:.0f} trips over the range")
    print(f"  candidate zones        : {n_kept + n_excluded} (valid ids "
          f"{config.VALID_ZONE_MIN}..{config.VALID_ZONE_MAX}, "
          f"{config.EXCLUDED_ZONE_IDS} already dropped upstream)")

    print(f"\n  EXCLUDED               : {n_excluded} zones, "
          f"{lost_trips:,} trips ({100 * lost_trips / all_trips:.4f}% of demand)")
    print("-" * 78)
    print(f"  {'zone':>5}  {'trips':>7}  {'/day':>6}  {'borough':<14} name")
    print("-" * 78)
    for row in excluded.orderBy("total_trips", "zone_id").collect():
        print(f"  {row['zone_id']:>5}  {row['total_trips']:>7,}  "
              f"{row['trips_per_day']:>6.2f}  {row['borough']:<14} {row['zone_name']}")
    print("-" * 78)

    by_borough = (
        excluded.groupBy("borough")
        .agg(F.count(F.lit(1)).alias("n"))
        .orderBy(F.desc("n"))
        .collect()
    )
    print("  excluded by borough    : "
          + ", ".join(f"{r['borough']} {r['n']}" for r in by_borough))

    print(f"\n  MODELLING SET          : {n_kept} zones, {kept_trips:,} trips "
          f"({100 * kept_trips / all_trips:.4f}% of demand retained)")
    smallest = kept.orderBy("total_trips").limit(3).collect()
    print("  smallest kept zones    : "
          + "; ".join(
              f"{r['zone_id']} {r['zone_name']} ({r['total_trips']:,})" for r in smallest
          ))
    print("=" * 78)


def main() -> int:
    spark = get_spark("zone-policy")

    if not config.DEMAND_PARQUET.exists():
        print(f"ERROR: {config.DEMAND_PARQUET} not found — run "
              "python -m src.batch.clean_aggregate", file=sys.stderr)
        spark.stop()
        return 2

    kept, excluded, n_days = split_zones(spark)
    kept.cache()
    excluded.cache()
    report(kept, excluded, n_days)

    kept.coalesce(1).write.mode("overwrite").parquet(str(config.MODELING_ZONES_PARQUET))
    print(f"\nWrote modelling zone list -> {config.MODELING_ZONES_PARQUET}")

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
