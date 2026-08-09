"""Scaling benchmark: wall-clock of the core batch path vs input size.

The proposal's Big-Data claim is that the pipeline is *distributed by design* —
the same code runs unchanged whether it is handed one month or a year. This
script measures that claim instead of asserting it: it runs the canonical heavy
path (read monthly TLC parquet -> apply the **exact** cleaning predicates from
``clean_aggregate.build_filters()`` -> aggregate to ``(zone, date, hour)`` ->
materialise) over growing month windows and records rows-in, rows-out and
wall-clock seconds.

What near-linear timings demonstrate: throughput is bounded by partition scans,
not by any single-machine data structure — Spark's execution graph is identical
on a cluster, so the measured curve is the single-node floor of a horizontally
scalable job. The figure caption states the machine, because wall-clock numbers
are machine-specific; the *shape* of the curve is the result.

Timing methodology: each window is timed around a full action (``count`` on the
aggregated frame after a fresh read), with a cold JVM warm-up window run first
and discarded so JIT/COMPILE time does not pollute the smallest window.

Run with ``TAXI_MONTHS=full`` — the cleaning predicates clip to the configured
date range, so the sample config would silently truncate every window to Q1::

    TAXI_MONTHS=full python -m src.batch.benchmark_scale
"""

from __future__ import annotations

import csv
import os
import platform
import sys
import time
from pathlib import Path

from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.batch.clean_aggregate import (  # noqa: E402
    COLUMN_TYPES, build_filters, local_date_hour,
)
from src.spark_session import describe, get_spark  # noqa: E402

WINDOWS = [1, 3, 6, 12]  # leading N months of config.FULL_YEAR_MONTHS
BENCHMARK_CSV = config.REPORTS_DIR / "scale_benchmark.csv"


def month_paths(n_months: int) -> list[str]:
    paths = []
    for month in config.FULL_YEAR_MONTHS[:n_months]:
        path = config.RAW_DIR / f"yellow_tripdata_{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run: python -m src.ingest.download_tlc --yes "
                "(with TAXI_MONTHS=full)"
            )
        paths.append(str(path))
    return paths


def run_window(spark, n_months: int) -> dict:
    """One timed pass of read -> clean -> aggregate -> materialise."""
    started = time.perf_counter()

    # Same explicit select + cast as clean_aggregate.read_raw() — 2024 files store
    # timestamps as TIMESTAMP_NTZ, which the duration filter cannot cast directly.
    raw = spark.read.parquet(*month_paths(n_months)).select(
        *[F.col(c).cast(COLUMN_TYPES[c]).alias(c) for c in config.TLC_COLUMNS]
    )
    rows_in = raw.count()

    cleaned = raw
    for _name, predicate in build_filters():
        cleaned = cleaned.filter(predicate)
    date_local, hour_local = local_date_hour(F.col("tpep_pickup_datetime"))
    demand = (
        cleaned.select(F.col("PULocationID"), date_local, hour_local)
        .groupBy("PULocationID", "date_local", "hour_local")
        .agg(F.count(F.lit(1)).alias("trip_count"))
    )
    rows_out = demand.count()

    elapsed = time.perf_counter() - started
    return {
        "months": n_months,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "seconds": round(elapsed, 2),
        "rows_per_second": int(rows_in / elapsed),
    }


def main() -> int:
    if not config.USE_FULL_YEAR:
        print("ERROR: run with TAXI_MONTHS=full — the cleaning predicates clip to "
              "the configured range, so the sample config would truncate every "
              "window to Q1 and the curve would be meaningless.", file=sys.stderr)
        return 2

    spark = get_spark("benchmark-scale")
    print("=" * 78)
    print("Scaling benchmark — core batch path, growing month windows")
    print("=" * 78)
    describe(spark)
    machine = f"{platform.machine()}, {os.cpu_count()} cores, local[*]"
    print(f"  machine   {machine}")
    print(f"  windows   {WINDOWS} months (leading months of 2024)")

    print("\n  warm-up (1 month, discarded — JVM/JIT cost stays out of the curve)")
    run_window(spark, 1)

    results = []
    for n_months in WINDOWS:
        result = run_window(spark, n_months)
        results.append(result)
        print(f"  {result['months']:>2} months : {result['rows_in']:>11,} rows in "
              f"-> {result['rows_out']:>9,} cells   {result['seconds']:>7.2f} s   "
              f"{result['rows_per_second']:>9,} rows/s")

    # --- validation: the claim is near-linearity, so measure it ---------------
    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)
    base = results[0]
    ok = True
    print(f"  {'window':>8} {'rows ratio':>11} {'time ratio':>11} "
          f"{'time/rows (1.0 = linear)':>26}")
    worst = 1.0
    for result in results[1:]:
        rows_ratio = result["rows_in"] / base["rows_in"]
        time_ratio = result["seconds"] / base["seconds"]
        ratio = time_ratio / rows_ratio
        worst = max(worst, ratio)
        print(f"  {result['months']:>7}m {rows_ratio:>11.2f} {time_ratio:>11.2f} "
              f"{ratio:>26.2f}")
    # Sub-linear is fine (fixed per-job overhead amortises); the failure mode the
    # claim rules out is super-linear growth. 1.5x head-room over strictly linear.
    linear_enough = worst <= 1.5
    print(f"\n  worst time-vs-rows ratio : {worst:.2f}  "
          f"(<= 1.5 required)  {'PASS' if linear_enough else 'FAIL'}")
    ok = ok and linear_enough

    with BENCHMARK_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["months", "rows_in", "rows_out", "seconds",
                        "rows_per_second", "machine"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({**result, "machine": machine})
    print(f"\n  wrote : {BENCHMARK_CSV}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    spark.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
