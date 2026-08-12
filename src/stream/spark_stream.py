"""Structured Streaming: Kafka -> windowed demand -> predicted vs actual.

Consumes ``taxi-trips``, reapplies the **batch** cleaning filters, aggregates to
tumbling 1-hour windows on event time, and serves the saved K-Means forecaster
(whatever K ``kmeans_metadata.json`` records). It never trains: the cluster shapes
and zone levels are loaded from ``models/`` exactly as ``train_kmeans.py`` wrote them.

**Nothing is reimplemented.** The cleaning predicates come from
``clean_aggregate.build_filters()`` and the NY-local calendar derivation from
``clean_aggregate.local_date_hour()``. The modelling-zone set is applied by joining
``modeling_zones.parquet``. Streamed per-window demand is therefore comparable to
``demand.parquet`` cell for cell, which ``--validate`` checks.

**Event time** is ``tpep_pickup_datetime`` from the payload — naive NY-local wall-clock,
read unshifted under the UTC session timezone, the same convention as everywhere else.
Windows are driven by the data, never by processing time, so a replay is reproducible.

**Watermark: 2 hours** (``config.WATERMARK_DELAY``). In a real feed a trip is only
recorded at dropoff, so its lateness relative to the pickup event time is roughly the
trip duration. Measured on 2024-01: p99.9 of duration is 1.909 h and only 0.092% of
trips exceed 2 h — beyond which the tail is almost flat (0.066% at 3 h, 0.059% at 6 h),
i.e. data-quality junk that the cleaning filters reject anyway. Two hours therefore
covers ~99.9% of genuine lateness without holding windows open indefinitely.

**Output mode: append.** A window is emitted once, after the watermark guarantees no
further rows can land in it, so every emitted value is final and directly comparable to
the batch aggregate. ``update`` mode would re-emit each window as it fills — livelier on
screen but the same window appears repeatedly with growing counts, and nothing marks
which emission is final, so batch/stream agreement could not be checked. Append is also
the only mode the parquet file sink supports. Use ``--console-mode update`` for a demo
where watching counts climb matters more than comparability.

Launch (first run downloads the connector from Maven into .ivy2/)::

    python -m src.stream.spark_stream --run-seconds 180 --validate

Equivalent spark-submit form::

    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3 \
      src/stream/spark_stream.py
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import reduce
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.batch.clean_aggregate import build_filters, local_date_hour  # noqa: E402
from src.spark_session import describe, get_spark  # noqa: E402

SPARK_TYPES = {"string": T.StringType(), "int": T.IntegerType(), "double": T.DoubleType()}


def message_schema() -> T.StructType:
    """Payload schema, built from the same list producer.py serialises from."""
    return T.StructType(
        [
            T.StructField(name, SPARK_TYPES[kind], True)
            for name, kind in config.KAFKA_MESSAGE_FIELDS
        ]
    )


def read_stream(spark: SparkSession, starting_offsets: str) -> DataFrame:
    """Kafka source -> parsed trip columns with a real timestamp for event time."""
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", config.KAFKA_TOPIC)
        .option("startingOffsets", starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = raw.select(
        F.from_json(F.col("value").cast("string"), message_schema()).alias("trip")
    ).select("trip.*")

    # The payload carries timestamps as strings; cast to the same types the batch
    # filters expect so build_filters() applies unchanged.
    return parsed.select(
        F.to_timestamp(
            F.col(config.KAFKA_EVENT_TIME_FIELD), config.KAFKA_EVENT_TIME_FORMAT
        ).alias("tpep_pickup_datetime"),
        F.to_timestamp(
            F.col("tpep_dropoff_datetime"), config.KAFKA_EVENT_TIME_FORMAT
        ).alias("tpep_dropoff_datetime"),
        F.col("PULocationID").cast("int").alias("PULocationID"),
        F.col("DOLocationID").cast("int").alias("DOLocationID"),
        F.col("passenger_count").cast("int").alias("passenger_count"),
        F.col("trip_distance").cast("double").alias("trip_distance"),
        F.col("fare_amount").cast("double").alias("fare_amount"),
        F.col("total_amount").cast("double").alias("total_amount"),
    )


def clean(trips: DataFrame) -> DataFrame:
    """Apply the batch cleaning predicates, unmodified."""
    predicates = [predicate for _, predicate in build_filters()]
    return trips.filter(reduce(lambda a, b: a & b, predicates))


def windowed_demand(trips: DataFrame, zones: DataFrame) -> DataFrame:
    """Tumbling 1-hour event-time windows, restricted to the modelling zones."""
    windowed = (
        trips.withWatermark("tpep_pickup_datetime", config.WATERMARK_DELAY)
        .groupBy(
            F.window(F.col("tpep_pickup_datetime"), config.WINDOW_DURATION).alias("w"),
            F.col("PULocationID"),
        )
        .agg(F.count(F.lit(1)).cast("int").alias("actual_demand"))
    )

    # Same calendar derivation as the batch job, applied to the window start.
    date_local, hour_local = local_date_hour(F.col("w.start"))
    windowed = windowed.select(
        F.col("PULocationID").alias("zone_id"),
        F.col("w.start").alias("window_start"),
        F.col("w.end").alias("window_end"),
        date_local,
        hour_local,
        "actual_demand",
    )

    # Restrict to the modelling-zone set so streamed zones match the model.
    return windowed.join(F.broadcast(zones), on="zone_id", how="inner")


def load_forecaster(spark: SparkSession) -> tuple[DataFrame, DataFrame]:
    """Load the saved artifacts. Returns ``(zones, cluster_shape)``.

    The live rule is cluster **shape** x the zone's own level — the same rule
    evaluate.py scored at MAE 7.587 on the full-year held-out split, not the raw
    pooled mean (20.223) and deliberately not hist_avg, which has no compact live
    representation. (Both figures come from reports/baseline_metrics.csv; earlier
    revisions of this docstring quoted the Q1 numbers.)
    """
    zones = spark.read.parquet(str(config.ZONE_CLUSTERS_PARQUET)).select(
        "zone_id", "cluster", "zone_mean_demand"
    )
    shape = spark.read.parquet(str(config.CLUSTER_PROFILE_PARQUET)).select(
        "cluster", "hour_of_week", "cluster_share"
    )
    return zones.cache(), shape.cache()


def add_prediction(windows: DataFrame, shape: DataFrame) -> DataFrame:
    """Attach the forecast: zone level x cluster share x 168, plus the error."""
    # dayofweek is 1=Sunday..7=Saturday; remap to 0=Monday..6=Sunday, as in features.py.
    dow = ((F.dayofweek(F.col("window_start")) + F.lit(5)) % F.lit(7)).cast("smallint")
    windows = windows.withColumn(
        "hour_of_week", (dow * F.lit(24) + F.col("hour_local")).cast("smallint")
    )

    joined = windows.join(F.broadcast(shape), on=["cluster", "hour_of_week"], how="left")
    return joined.withColumn(
        "predicted_demand",
        F.round(
            F.col("zone_mean_demand")
            * F.col("cluster_share")
            * F.lit(float(config.HOURS_PER_WEEK)),
            2,
        ),
    ).withColumn(
        "error", F.round(F.col("predicted_demand") - F.col("actual_demand"), 2)
    )


OUTPUT_COLUMNS = [
    "zone_id", "window_start", "window_end", "date_local", "hour_local",
    "cluster", "actual_demand", "predicted_demand", "error",
]


def make_sink(console_rows: int):
    """foreachBatch sink: append to parquet and print a sample of each batch."""

    def write_batch(batch: DataFrame, batch_id: int) -> None:
        if batch.rdd.isEmpty():
            return
        batch = batch.select(*OUTPUT_COLUMNS).cache()
        n = batch.count()

        stats = batch.agg(
            F.min("window_start").alias("lo"),
            F.max("window_start").alias("hi"),
            F.sum("actual_demand").alias("actual"),
            F.sum("predicted_demand").alias("predicted"),
            F.avg(F.abs("error")).alias("mae"),
        ).first()

        print(f"\n--- batch {batch_id}: {n:,} closed (zone, window) cells | "
              f"windows {stats['lo']} .. {stats['hi']} ---")
        print(f"    actual {int(stats['actual']):,} | "
              f"predicted {stats['predicted']:,.0f} | live MAE {stats['mae']:.2f}")
        batch.orderBy(F.desc("actual_demand")).limit(console_rows).show(truncate=False)

        batch.write.mode("append").parquet(str(config.STREAM_OUTPUT_DIR))
        batch.unpersist()

    return write_batch


def validate(spark: SparkSession) -> bool:
    """Streamed per-window demand must equal demand.parquet for the same cells."""
    print("\n" + "=" * 92)
    print("VALIDATION — streamed windows vs batch demand.parquet")
    print("=" * 92)

    if not Path(config.STREAM_OUTPUT_DIR).exists():
        print("  no streamed output written — nothing to compare")
        return False

    streamed = (
        spark.read.parquet(str(config.STREAM_OUTPUT_DIR))
        .groupBy("zone_id", "date_local", "hour_local")
        .agg(F.sum("actual_demand").cast("int").alias("streamed"))
    )
    # demand.parquet covers every zone that saw a trip; the stream serves only the
    # modelling-zone set. Restrict the batch side to the same set or the comparison
    # reports excluded zones as phantom mismatches.
    modeling = spark.read.parquet(str(config.MODELING_ZONES_PARQUET)).select("zone_id")
    batch = (
        spark.read.parquet(str(config.DEMAND_PARQUET))
        .select(
            F.col("PULocationID").alias("zone_id"), "date_local", "hour_local",
            F.col("trip_count").alias("batch"),
        )
        .join(F.broadcast(modeling), on="zone_id", how="inner")
    )

    windows = [
        r["window"]
        for r in streamed.select(
            F.concat_ws(" ", "date_local", F.lpad("hour_local", 2, "0")).alias("window")
        ).distinct().orderBy("window").collect()
    ]
    print(f"  closed windows streamed : {len(windows)}")
    if not windows:
        print("  none — the watermark never advanced past a window boundary.")
        print("  Replay a longer slice: with a 2h watermark the last ~2h never close.")
        return False
    print(f"  window hours            : {', '.join(windows[:8])}"
          f"{' ...' if len(windows) > 8 else ''}")

    compared = streamed.join(batch, on=["zone_id", "date_local", "hour_local"], how="full")
    compared = compared.fillna({"streamed": 0, "batch": 0})

    # Only compare cells inside the windows the stream actually closed; batch covers
    # the whole quarter, so anything outside is simply not part of this replay.
    in_scope = compared.filter(
        F.concat_ws(" ", "date_local", F.lpad("hour_local", 2, "0")).isin(windows)
    ).cache()

    total = in_scope.count()
    mismatched = in_scope.filter(F.col("streamed") != F.col("batch"))
    n_bad = mismatched.count()

    sums = in_scope.agg(
        F.sum("streamed").alias("s"), F.sum("batch").alias("b")
    ).first()

    print(f"\n  (zone, hour) cells in those windows : {total:,}")
    print(f"  streamed total trips                : {int(sums['s']):,}")
    print(f"  batch total trips                   : {int(sums['b']):,}")
    print(f"  cells disagreeing                   : {n_bad:,}")

    if n_bad:
        print("\n  MISMATCH — first 20 disagreeing cells:")
        mismatched.orderBy(
            F.desc(F.abs(F.col("streamed") - F.col("batch")))
        ).limit(20).show(truncate=False)
        print("  Streaming and batch disagree. Stopping here rather than proceeding.")
        return False

    print("\n  Every streamed cell matches the batch aggregate exactly.")

    print("\n  Per-window totals:")
    (
        in_scope.groupBy("date_local", "hour_local")
        .agg(
            F.sum("streamed").alias("streamed_trips"),
            F.sum("batch").alias("batch_trips"),
            F.countDistinct("zone_id").alias("zones"),
        )
        .orderBy("date_local", "hour_local")
        .show(truncate=False)
    )
    return True


def show_predictions(spark: SparkSession) -> None:
    """A few predicted-vs-actual rows from the live forecaster."""
    if not Path(config.STREAM_OUTPUT_DIR).exists():
        return
    output = spark.read.parquet(str(config.STREAM_OUTPUT_DIR)).cache()

    print("\n" + "=" * 92)
    print("LIVE FORECASTER — predicted vs actual (cluster shape x zone level)")
    print("=" * 92)

    stats = output.agg(
        F.count(F.lit(1)).alias("n"),
        F.avg(F.abs("error")).alias("mae"),
        F.sqrt(F.avg(F.pow("error", F.lit(2)))).alias("rmse"),
        F.sum(F.abs("error")).alias("abs_err"),
        F.sum("actual_demand").alias("actual"),
    ).first()
    wape = float(stats["abs_err"]) / max(float(stats["actual"]), 1.0)
    print(f"  cells {int(stats['n']):,} | MAE {stats['mae']:.3f} | "
          f"RMSE {stats['rmse']:.3f} | WAPE {100 * wape:.2f}%")
    print("  (in-sample here: the replayed slice is January, which is in the training")
    print("   split — this demonstrates the live path, not held-out accuracy.)")

    print("\n  Busiest cells:")
    output.orderBy(F.desc("actual_demand")).select(*OUTPUT_COLUMNS).show(
        8, truncate=False
    )
    print("  Quietest cells:")
    output.orderBy("actual_demand", "zone_id").select(*OUTPUT_COLUMNS).show(
        5, truncate=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-seconds", type=int, default=0,
        help="stop after N seconds (0 = run until interrupted)",
    )
    parser.add_argument(
        "--starting-offsets", default="earliest", choices=["earliest", "latest"],
        help="Kafka start position (default earliest)",
    )
    parser.add_argument(
        "--console-mode", default="append", choices=["append", "update"],
        help="append emits each window once when final; update re-emits as it fills",
    )
    parser.add_argument(
        "--trigger-seconds", type=int, default=5, help="micro-batch interval",
    )
    parser.add_argument(
        "--console-rows", type=int, default=5, help="rows to print per batch",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="clear the parquet sink and checkpoint before starting",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="after stopping, compare streamed windows against demand.parquet",
    )
    parser.add_argument(
        "--ready-file",
        help="write this file once the query is running, so a launcher can wait for "
             "the consumer before starting the producer",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for path, hint in (
        (config.ZONE_CLUSTERS_PARQUET, "src.batch.train_kmeans"),
        (config.CLUSTER_PROFILE_PARQUET, "src.batch.train_kmeans"),
        (config.MODELING_ZONES_PARQUET, "src.batch.zone_policy"),
    ):
        if not Path(path).exists():
            print(f"ERROR: {path} missing — run: python -m {hint}", file=sys.stderr)
            return 2

    if args.fresh:
        import shutil  # noqa: PLC0415

        for path in (config.STREAM_OUTPUT_DIR, config.STREAM_CHECKPOINT_DIR):
            if Path(path).exists():
                shutil.rmtree(path)
                print(f"cleared {path}")

    spark = get_spark("spark-stream", packages=config.SPARK_PACKAGES_KAFKA_ONLY)

    print("=" * 92)
    print("Structured Streaming — Kafka -> windowed demand -> predicted vs actual")
    print("=" * 92)
    describe(spark)
    print(f"  connector    {config.KAFKA_PACKAGE}")
    print(f"  topic        {config.KAFKA_TOPIC} @ {config.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"  offsets      {args.starting_offsets}")
    print(f"  window       tumbling {config.WINDOW_DURATION} on "
          f"{config.KAFKA_EVENT_TIME_FIELD} (event time)")
    print(f"  watermark    {config.WATERMARK_DELAY}  "
          "(covers p99.9 of trip duration = 1.91h)")
    print(f"  output mode  {args.console_mode}")
    print(f"  model        {config.KMEANS_MODEL_DIR} — loaded, never retrained")

    zones, shape = load_forecaster(spark)
    modeling = spark.read.parquet(str(config.MODELING_ZONES_PARQUET)).select("zone_id")
    # The inner join is the intended restriction, but it used to be silent: a model
    # fitted on a different range than the zone list on disk simply served fewer
    # zones than it knew about, with nothing in the output saying so.
    n_model_zones = zones.count()
    zones = zones.join(F.broadcast(modeling), on="zone_id", how="inner").cache()
    n_served = zones.count()
    print(f"  zones served {n_served} (modelling set)")

    range_warning = config.model_range_warning()
    if range_warning or n_served != n_model_zones:
        print("  " + "!" * 74)
        if range_warning:
            print(f"  WARNING: {range_warning}")
        if n_served != n_model_zones:
            print(f"  WARNING: the saved model covers {n_model_zones} zones but only "
                  f"{n_served} are in {config.MODELING_ZONES_PARQUET.name} — "
                  f"{n_model_zones - n_served} are not served.")
        print("  Re-run src.batch.zone_policy and src.batch.train_kmeans on the same")
        print("  TAXI_MONTHS to align them. Streaming continues on the served set.")
        print("  " + "!" * 74)

    trips = clean(read_stream(spark, args.starting_offsets))
    windows = windowed_demand(trips, zones)
    predicted = add_prediction(windows, shape)

    query = (
        predicted.writeStream.outputMode(args.console_mode)
        .foreachBatch(make_sink(args.console_rows))
        .option("checkpointLocation", str(config.STREAM_CHECKPOINT_DIR))
        .trigger(processingTime=f"{args.trigger_seconds} seconds")
        .start()
    )

    print(f"\nquery started (id {query.id}); "
          f"{'running ' + str(args.run_seconds) + 's' if args.run_seconds else 'Ctrl-C to stop'}")

    if args.ready_file:
        ready = Path(args.ready_file)
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.write_text(str(query.id), encoding="utf-8")
        print(f"readiness marker written -> {ready}")

    try:
        if args.run_seconds:
            deadline = time.monotonic() + args.run_seconds
            while time.monotonic() < deadline and query.isActive:
                query.awaitTermination(timeout=2)
        else:
            query.awaitTermination()
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if query.isActive:
            query.stop()
        progress = query.lastProgress
        if progress:
            print(f"\nlast batch: {progress.get('numInputRows', 0)} input rows, "
                  f"watermark {progress.get('eventTime', {}).get('watermark', 'n/a')}")

    ok = True
    if args.validate:
        ok = validate(spark)
        if ok:
            show_predictions(spark)

    print("\n" + "=" * 92)
    print("DONE")
    print(f"  stream output : {config.STREAM_OUTPUT_DIR}")
    print(f"  checkpoint    : {config.STREAM_CHECKPOINT_DIR}")
    print("=" * 92)

    spark.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
