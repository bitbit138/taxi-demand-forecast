"""Association rules over origin->destination flows with Spark MLlib FP-Growth.

The course pairs clustering with **association rules**; this is the project's use
of the second technique, applied where it means something for the demand problem:
a taxi that drops off is supply appearing where demand comes next, so
high-confidence ``pickup-zone -> dropoff-zone`` rules are exactly the
repositioning structure a dispatcher exploits between forecast windows.

**Design.** One transaction per cleaned trip, two items:
``PU:<zone_id>`` and ``DO:<zone_id>``. FP-Growth mines frequent itemsets and
rules; we keep the directed ``PU -> DO`` rules and read three numbers per rule:

  * **support**    — share of all trips on this exact flow
  * **confidence** — P(dropoff zone | pickup zone): where does this origin send you
  * **lift**       — confidence over the destination's base rate; > 1 means the
    origin *specifically* feeds that destination rather than it being popular
    everywhere

Cleaning is the batch pipeline's own (``build_filters()`` imported, not
re-implemented), so the transaction set is the same 39M-trip population every
other result uses. Run with ``TAXI_MONTHS=full``.

    TAXI_MONTHS=full python -m src.batch.association_rules
"""

from __future__ import annotations

import sys
from pathlib import Path

from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.batch.clean_aggregate import COLUMN_TYPES, build_filters  # noqa: E402
from src.spark_session import describe, get_spark  # noqa: E402

MIN_SUPPORT = 0.001      # >= ~39k trips on the flow — structural, not anecdotal
# 5%: NYC destinations are dispersed, so per-origin confidence is structurally low —
# at 10% only adjacent-neighbourhood flows survive; 5% also admits the airport
# corridors while still requiring 1-in-20 of the origin's trips on the flow.
MIN_CONFIDENCE = 0.05
TOP_N = 15
RULES_CSV = config.REPORTS_DIR / "association_rules.csv"


def main() -> int:
    if not config.USE_FULL_YEAR:
        print("ERROR: run with TAXI_MONTHS=full — the cleaning predicates clip to "
              "the configured range.", file=sys.stderr)
        return 2

    spark = get_spark("association-rules")
    print("=" * 92)
    print("FP-Growth — pickup -> dropoff association rules (full-year 2024)")
    print("=" * 92)
    describe(spark)
    print(f"  minSupport    {MIN_SUPPORT}  (~{MIN_SUPPORT:.1%} of all trips)")
    print(f"  minConfidence {MIN_CONFIDENCE}")

    # Same explicit select + cast as clean_aggregate.read_raw() — 2024 files store
    # timestamps as TIMESTAMP_NTZ, which the duration filter cannot cast directly.
    raw = spark.read.parquet(
        *[str(config.RAW_DIR / f"yellow_tripdata_{m}.parquet")
          for m in config.MONTHS]
    ).select(*[F.col(c).cast(COLUMN_TYPES[c]).alias(c) for c in config.TLC_COLUMNS])
    cleaned = raw
    for _name, predicate in build_filters():
        cleaned = cleaned.filter(predicate)

    transactions = cleaned.select(
        F.array(
            F.concat(F.lit("PU:"), F.col("PULocationID").cast("string")),
            F.concat(F.lit("DO:"), F.col("DOLocationID").cast("string")),
        ).alias("items")
    )
    n_trips = transactions.count()
    print(f"  transactions  {n_trips:,} cleaned trips, 2 items each (PU, DO)")

    model = FPGrowth(
        itemsCol="items", minSupport=MIN_SUPPORT, minConfidence=MIN_CONFIDENCE
    ).fit(transactions)

    # Directed PU -> DO rules only; the reverse direction answers a different
    # question ("where did arrivals come from") and would double the table.
    rules = (
        model.associationRules
        .filter(
            (F.size("antecedent") == 1) & (F.size("consequent") == 1)
            & F.element_at("antecedent", 1).startswith("PU:")
            & F.element_at("consequent", 1).startswith("DO:")
        )
        .withColumn("pu_zone", F.regexp_extract(F.element_at("antecedent", 1), r"(\d+)", 1).cast("int"))
        .withColumn("do_zone", F.regexp_extract(F.element_at("consequent", 1), r"(\d+)", 1).cast("int"))
        .select("pu_zone", "do_zone", "support", "confidence", "lift")
    )

    lookup = (
        spark.read.option("header", True).csv(str(config.ZONE_LOOKUP_CSV))
        .select(F.col("LocationID").cast("int").alias("zone_id"),
                F.col("Zone").alias("zone_name"))
    )
    named = (
        rules
        .join(lookup.withColumnRenamed("zone_id", "pu_zone")
              .withColumnRenamed("zone_name", "pu_name"), on="pu_zone")
        .join(lookup.withColumnRenamed("zone_id", "do_zone")
              .withColumnRenamed("zone_name", "do_name"), on="do_zone")
        .orderBy(F.desc("lift"))
        .cache()
    )
    n_rules = named.count()

    print(f"\n  {n_rules} directed PU->DO rules clear the thresholds")
    print(f"\n  TOP {TOP_N} BY LIFT — origins that specifically feed a destination")
    print(f"  {'pickup':<28}{'dropoff':<28}{'support':>9}{'conf':>7}{'lift':>7}")
    print("  " + "-" * 78)
    top = named.limit(TOP_N).collect()
    for row in top:
        print(f"  {row['pu_name']:<28.27}{row['do_name']:<28.27}"
              f"{100 * row['support']:>8.3f}%{row['confidence']:>7.2f}{row['lift']:>7.1f}")

    # --- validation -----------------------------------------------------------
    print("\n" + "=" * 92)
    print("VALIDATION")
    print("=" * 92)
    ok = n_rules > 0
    print(f"  rules found                   : {n_rules} (> 0: {ok})")

    bounds = named.agg(
        F.min("support").alias("min_s"), F.max("confidence").alias("max_c"),
        F.min("confidence").alias("min_c"),
    ).first()
    support_ok = float(bounds["min_s"]) >= MIN_SUPPORT
    conf_ok = MIN_CONFIDENCE <= float(bounds["min_c"]) and float(bounds["max_c"]) <= 1.0
    print(f"  all supports >= minSupport    : {support_ok}")
    print(f"  confidences within [{MIN_CONFIDENCE}, 1]   : {conf_ok}")
    ok = ok and support_ok and conf_ok

    # Lift sanity on a known structural flow: an airport's top rule should beat
    # independence clearly (airports are the canonical directed flow in NYC).
    airport = named.filter(F.col("pu_name").contains("Airport"))
    airport_top = airport.first()
    if airport_top is not None:
        lift_ok = float(airport_top["lift"]) > 1.5
        print(f"  top airport rule lift > 1.5   : {lift_ok} "
              f"({airport_top['pu_name']} -> {airport_top['do_name']}, "
              f"lift {airport_top['lift']:.1f})")
        ok = ok and lift_ok

    named.toPandas().to_csv(RULES_CSV, index=False)
    print(f"\n  wrote : {RULES_CSV}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")

    spark.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
