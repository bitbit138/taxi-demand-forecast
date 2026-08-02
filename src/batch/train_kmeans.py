"""Cluster zones by weekly demand shape with Spark MLlib K-Means.

Builds the **223 x 168 zone-by-hour-of-week profile matrix** from
``features.parquet`` — the aggregation deliberately kept out of ``features.py`` so it
is obvious what K-Means actually consumes — sweeps K, and saves the fitted pipeline.

**Train rows only.** The profile is built from ``is_train`` rows exclusively, the same
discipline as ``hist_avg_demand``. Clustering over the full quarter would let test-period
demand shape the centroids that are then scored against that same test period.
``assert_train_only()`` proves the profile's date range lies inside the training split
rather than trusting the filter.

**Clustering on pattern, not volume.** Raw profiles are dominated by magnitude: Midtown
does ~400k trips a quarter and East Flushing ~100, so Euclidean distance on raw counts
sorts zones into volume tiers and says nothing about *when* demand happens. Each zone's
profile is therefore **L1 row-normalised** into a distribution over the 168 hours of the
week — every zone sums to 1.0 and each value reads directly as "share of this zone's
weekly demand occurring in this hour". L1 was chosen over per-zone z-scoring because it
keeps that interpretation (a z-scored profile has no natural units and its centroids are
not readable as demand shares), and because a demand profile is naturally a
non-negative distribution. ``StandardScaler`` then runs *across* zones per hour column
so no single hour dominates the distance.

Both variants are fitted and reported so the effect of normalisation is visible:

  * ``raw``        assemble -> scale -> KMeans          (volume still present)
  * ``normalized`` assemble -> L1 -> scale -> KMeans    (shape only; this is saved)

**Elbow vs silhouette.** Both curves are printed for K=2..12 and the disagreement, if
any, is surfaced rather than resolved silently. Override with ``--k``.

    python -m src.batch.train_kmeans
    python -m src.batch.train_kmeans --k 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Normalizer, StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.spark_session import describe, get_spark  # noqa: E402

PROFILE_COLS = [f"how_{i}" for i in range(config.HOURS_PER_WEEK)]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def how_label(hour_of_week: int) -> str:
    """168-hour index -> e.g. 'Fri 23:00'."""
    return f"{DAY_NAMES[hour_of_week // 24]} {hour_of_week % 24:02d}:00"


def assert_train_only(features: DataFrame) -> None:
    """Prove the profile source contains no test dates."""
    train = features.filter(F.col("is_train"))
    bounds = train.agg(
        F.min("date_local").alias("lo"), F.max("date_local").alias("hi")
    ).first()
    test_lo = features.filter(~F.col("is_train")).agg(F.min("date_local")).first()[0]

    leaked = train.filter(F.col("date_local") >= F.lit(test_lo)).count()
    print(f"  profile source dates : {bounds['lo']} .. {bounds['hi']}")
    print(f"  test split begins    : {test_lo}")
    print(f"  train rows on/after test start : {leaked}")
    if leaked or bounds["hi"] >= test_lo:
        raise ValueError("Profile matrix would include test dates — leakage.")
    print("  -> profile is strictly within the training split")


def build_profiles(features: DataFrame) -> DataFrame:
    """223 x 168 matrix: mean demand per zone per hour-of-week, train rows only."""
    train = features.filter(F.col("is_train"))

    long = (
        train.groupBy("zone_id", "hour_of_week")
        .agg(F.avg("trip_count").alias("mean_demand"))
    )
    wide = (
        long.groupBy("zone_id")
        .pivot("hour_of_week", list(range(config.HOURS_PER_WEEK)))
        .agg(F.first("mean_demand"))
    )
    wide = wide.toDF("zone_id", *PROFILE_COLS)

    # Any hour-of-week never observed for a zone would break VectorAssembler.
    missing = wide.select(
        F.sum(
            sum((F.col(c).isNull()).cast("int") for c in PROFILE_COLS)
        ).alias("nulls")
    ).first()["nulls"]
    if missing:
        print(f"  WARNING: {missing} empty (zone, hour-of-week) cells filled with 0")
        wide = wide.fillna(0.0, subset=PROFILE_COLS)

    return wide.withColumn(
        "total_demand", sum(F.col(c) for c in PROFILE_COLS)
    )


def make_pipeline(k: int, normalize: bool) -> Pipeline:
    """Assemble -> (L1 normalise) -> standardise -> KMeans, with a fixed seed."""
    stages = [VectorAssembler(inputCols=PROFILE_COLS, outputCol="raw_features")]
    scaler_input = "raw_features"

    if normalize:
        stages.append(
            # p=1: each zone's 168 values sum to 1 -> a demand distribution.
            Normalizer(inputCol="raw_features", outputCol="shape_features", p=1.0)
        )
        scaler_input = "shape_features"

    stages.append(
        StandardScaler(
            inputCol=scaler_input, outputCol="features", withMean=True, withStd=True
        )
    )
    stages.append(
        KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=k,
            seed=config.SEED,
            maxIter=config.KMEANS_MAX_ITER,
            initMode="k-means||",
        )
    )
    return Pipeline(stages=stages)


def sweep(profiles: DataFrame, normalize: bool, label: str) -> list[dict]:
    """Fit K=K_MIN..K_MAX, recording WCSS and full (unsampled) silhouette."""
    evaluator = ClusteringEvaluator(
        featuresCol="features",
        predictionCol="cluster",
        metricName="silhouette",
        distanceMeasure="squaredEuclidean",
    )

    results = []
    print(f"\n  sweeping {label}: K = {config.K_MIN}..{config.K_MAX}")
    for k in range(config.K_MIN, config.K_MAX + 1):
        model = make_pipeline(k, normalize).fit(profiles)
        assigned = model.transform(profiles)
        wcss = model.stages[-1].summary.trainingCost
        # 223 points — the silhouette is exact, nothing is sampled.
        silhouette = evaluator.evaluate(assigned)
        sizes = model.stages[-1].summary.clusterSizes
        results.append(
            {
                "k": k,
                "wcss": float(wcss),
                "silhouette": float(silhouette),
                "min_cluster": int(min(sizes)),
                "max_cluster": int(max(sizes)),
            }
        )
        print(f"    K={k:>2}  WCSS={wcss:>12,.1f}  silhouette={silhouette:+.4f}  "
              f"sizes {min(sizes)}..{max(sizes)}")
    return results


def elbow_k(results: list[dict]) -> int:
    """Knee of the WCSS curve: point furthest from the line joining its endpoints."""
    xs = [r["k"] for r in results]
    ys = [r["wcss"] for r in results]
    x0, y0, x1, y1 = xs[0], ys[0], xs[-1], ys[-1]

    # Normalise both axes so the distance is not dominated by the WCSS magnitude.
    span_x = (x1 - x0) or 1.0
    span_y = (y0 - y1) or 1.0
    best_k, best_d = xs[0], -1.0
    for x, y in zip(xs, ys):
        nx, ny = (x - x0) / span_x, (y - y1) / span_y
        # Line from (0,1) to (1,0) in normalised space -> distance = |nx + ny - 1|/sqrt2
        d = abs(nx + ny - 1.0)
        if d > best_d:
            best_k, best_d = x, d
    return best_k


def print_curves(results: list[dict]) -> None:
    """ASCII elbow and silhouette curves."""
    width = 44
    wcss = [r["wcss"] for r in results]
    sil = [r["silhouette"] for r in results]
    w_lo, w_hi = min(wcss), max(wcss)
    s_lo, s_hi = min(sil), max(sil)

    print(f"\n  {'K':>3}  {'WCSS':>12}  elbow{'':<40}{'silhouette':>12}  quality")
    print("  " + "-" * 96)
    for r in results:
        w_frac = (r["wcss"] - w_lo) / ((w_hi - w_lo) or 1)
        s_frac = (r["silhouette"] - s_lo) / ((s_hi - s_lo) or 1)
        w_bar = "#" * max(1, int(w_frac * width))
        s_bar = "#" * max(1, int(s_frac * width))
        print(f"  {r['k']:>3}  {r['wcss']:>12,.0f}  {w_bar:<45}"
              f"{r['silhouette']:>+12.4f}  {s_bar}")


def cluster_volume_correlation(assigned: DataFrame) -> float:
    """How strongly does the clustering track raw volume? 1.0 = pure volume tiers."""
    return float(
        assigned.select(F.corr("cluster", "total_demand")).first()[0] or 0.0
    )


def describe_clusters(assigned: DataFrame, spark: SparkSession, k: int) -> None:
    """Size, peak-hour signature and representative zones for each cluster."""
    zones = spark.read.parquet(str(config.MODELING_ZONES_PARQUET)).select(
        "zone_id", "zone_name", "borough", "total_trips"
    )
    joined = assigned.join(F.broadcast(zones), on="zone_id", how="left").cache()

    # Cluster mean profile, as shares, to read off the peak hour-of-week.
    share_exprs = [
        F.avg(F.col(c) / F.col("total_demand")).alias(c) for c in PROFILE_COLS
    ]
    means = joined.groupBy("cluster").agg(*share_exprs).collect()
    peaks = {}
    for row in means:
        shares = [(c, row[c]) for c in PROFILE_COLS]
        top = sorted(shares, key=lambda t: t[1], reverse=True)[:3]
        peaks[int(row["cluster"])] = [
            (int(c.split("_")[1]), v) for c, v in top
        ]

    stats = (
        joined.groupBy("cluster")
        .agg(
            F.count(F.lit(1)).alias("n_zones"),
            F.avg("total_trips").alias("avg_zone_trips"),
            F.sum("total_trips").alias("sum_trips"),
        )
        .orderBy("cluster")
        .collect()
    )

    print("\n" + "=" * 78)
    print(f"CLUSTER INTERPRETATION (K={k})")
    print("=" * 78)
    for row in stats:
        cid = int(row["cluster"])
        top = peaks[cid]
        peak_str = ", ".join(f"{how_label(h)} ({100 * v:.2f}%)" for h, v in top)
        print(f"\n  cluster {cid}  |  {int(row['n_zones']):>3} zones  |  "
              f"avg {row['avg_zone_trips']:>9,.0f} trips/zone  |  "
              f"{int(row['sum_trips']):>9,} total")
        print(f"    peak hours-of-week : {peak_str}")

        members = (
            joined.filter(F.col("cluster") == cid)
            .orderBy(F.desc("total_trips"))
            .select("zone_id", "zone_name", "borough", "total_trips")
            .limit(3)
            .collect()
        )
        print("    representative zones (largest):")
        for m in members:
            print(f"      {m['zone_id']:>4}  {m['zone_name'][:44]:<44} "
                  f"{m['borough']:<14} {int(m['total_trips']):>8,}")
    print("=" * 78)


def save_artifacts(
    model: PipelineModel,
    assigned: DataFrame,
    features: DataFrame,
    k: int,
    chosen_by: str,
    sweeps: dict[str, list[dict]],
    selection: dict,
) -> None:
    """Persist the pipeline and the prediction rule for the streaming job."""
    model.write().overwrite().save(str(config.KMEANS_MODEL_DIR))

    zone_clusters = assigned.select("zone_id", "cluster", "total_demand")
    zone_clusters.coalesce(1).write.mode("overwrite").parquet(
        str(config.ZONE_CLUSTERS_PARQUET)
    )

    # Prediction rule: a query's demand is its cluster's mean demand for that
    # hour-of-week. Computed on TRAIN rows only, like everything else.
    cluster_demand = (
        features.filter(F.col("is_train"))
        .join(F.broadcast(zone_clusters.select("zone_id", "cluster")), on="zone_id")
        .groupBy("cluster", "hour_of_week")
        .agg(
            F.avg("trip_count").alias("predicted_demand"),
            F.count(F.lit(1)).alias("n_observations"),
        )
    )
    cluster_demand.coalesce(1).write.mode("overwrite").parquet(
        str(config.CLUSTER_PROFILE_PARQUET)
    )

    metadata = {
        "chosen_k": k,
        "chosen_by": chosen_by,
        # Kept even when K is overridden — the report needs to show that the two
        # criteria disagreed and which one the chosen K came from.
        "k_suggested_by_elbow": selection["elbow"],
        "k_suggested_by_silhouette": selection["silhouette"],
        "criteria_agree": selection["elbow"] == selection["silhouette"],
        "seed": config.SEED,
        "normalization": "L1 row-normalisation (demand share per hour-of-week)",
        "profile_shape": [assigned.count(), config.HOURS_PER_WEEK],
        "train_only": True,
        "spark_version": config.SPARK_VERSION,
        "k_range": [config.K_MIN, config.K_MAX],
        "sweeps": sweeps,
    }
    config.KMEANS_METADATA_JSON.write_text(json.dumps(metadata, indent=2))

    with config.KSWEEP_CSV.open("w", encoding="utf-8") as handle:
        handle.write("variant,k,wcss,silhouette,min_cluster,max_cluster\n")
        for variant, rows in sweeps.items():
            for r in rows:
                handle.write(
                    f"{variant},{r['k']},{r['wcss']:.6f},{r['silhouette']:.6f},"
                    f"{r['min_cluster']},{r['max_cluster']}\n"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--k", type=int, default=None,
        help="override the chosen K (default: resolve elbow vs silhouette)",
    )
    parser.add_argument(
        "--skip-raw", action="store_true",
        help="skip the raw-profile comparison sweep",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not config.FEATURES_PARQUET.exists():
        print(f"ERROR: {config.FEATURES_PARQUET} missing — run "
              "python -m src.batch.features", file=sys.stderr)
        return 2

    spark = get_spark("train-kmeans")
    print("=" * 78)
    print("K-Means on zone weekly demand profiles")
    print("=" * 78)
    describe(spark)
    print(f"  seed      {config.SEED} (fixed)")
    print(f"  K range   {config.K_MIN}..{config.K_MAX}")

    features = spark.read.parquet(str(config.FEATURES_PARQUET)).cache()

    print("\nLeakage check:")
    assert_train_only(features)

    profiles = build_profiles(features).cache()
    n_zones = profiles.count()
    print(f"\n  profile matrix : {n_zones} zones x {config.HOURS_PER_WEEK} "
          "hour-of-week columns")

    sweeps: dict[str, list[dict]] = {}
    if not args.skip_raw:
        sweeps["raw"] = sweep(profiles, normalize=False, label="RAW profiles (volume present)")
    sweeps["normalized"] = sweep(
        profiles, normalize=True, label="NORMALISED profiles (shape only)"
    )

    # --- does normalisation actually change what is being clustered? ----------
    if "raw" in sweeps:
        print("\n" + "=" * 78)
        print("DOES SCALE DOMINATE? (correlation of cluster id with zone volume)")
        print("=" * 78)
        for variant, normalize in (("raw", False), ("normalized", True)):
            probe_k = max(r["silhouette"] for r in sweeps[variant])
            probe_k = next(r["k"] for r in sweeps[variant] if r["silhouette"] == probe_k)
            probe = make_pipeline(probe_k, normalize).fit(profiles).transform(profiles)
            corr = cluster_volume_correlation(probe)
            spread = (
                probe.groupBy("cluster")
                .agg(F.avg("total_demand").alias("avg"))
                .agg(F.max("avg") / F.min("avg"))
                .first()[0]
            )
            print(f"  {variant:<11} best-silhouette K={probe_k:>2}  "
                  f"|corr(cluster, volume)| = {abs(corr):.3f}  "
                  f"max/min cluster mean volume = {spread:,.1f}x")
        print("  A high correlation or a huge volume spread means the clustering is")
        print("  really a volume tiering rather than a temporal-shape grouping.")
        print("=" * 78)

    results = sweeps["normalized"]
    print("\n" + "=" * 78)
    print("K SELECTION (normalised profiles)")
    print("=" * 78)
    print_curves(results)

    k_elbow = elbow_k(results)
    best_sil = max(results, key=lambda r: r["silhouette"])
    k_sil = best_sil["k"]

    print(f"\n  elbow (WCSS knee)     suggests K = {k_elbow}")
    print(f"  silhouette maximum    suggests K = {k_sil} "
          f"({best_sil['silhouette']:+.4f})")

    if args.k is not None:
        chosen, chosen_by = args.k, "manual --k override"
    elif k_elbow == k_sil:
        chosen, chosen_by = k_sil, "elbow and silhouette agree"
    else:
        sil_at_elbow = next(r["silhouette"] for r in results if r["k"] == k_elbow)
        print("\n  " + "!" * 74)
        print("  THE TWO CRITERIA DISAGREE — surfacing rather than picking silently.")
        print(f"    K={k_sil:>2}: silhouette {best_sil['silhouette']:+.4f} (best separation), "
              f"WCSS {next(r['wcss'] for r in results if r['k'] == k_sil):,.0f}")
        print(f"    K={k_elbow:>2}: silhouette {sil_at_elbow:+.4f}, "
              f"WCSS {next(r['wcss'] for r in results if r['k'] == k_elbow):,.0f} "
              "(knee of the elbow curve)")
        print("    Silhouette favours few, well-separated clusters; the elbow favours")
        print("    explaining more variance. More clusters = finer demand patterns but")
        print("    thinner per-cluster evidence for the prediction rule.")
        print(f"    Defaulting to the silhouette's K={k_sil}; re-run with --k "
              f"{k_elbow} to take the elbow's.")
        print("  " + "!" * 74)
        chosen, chosen_by = k_sil, f"silhouette max (elbow suggested {k_elbow})"

    print(f"\n  CHOSEN K = {chosen}  ({chosen_by})")

    final = make_pipeline(chosen, normalize=True).fit(profiles)
    assigned = final.transform(profiles).select(
        "zone_id", "cluster", "total_demand", *PROFILE_COLS
    ).cache()

    describe_clusters(assigned, spark, chosen)
    save_artifacts(
        final, assigned, features, chosen, chosen_by, sweeps,
        {"elbow": k_elbow, "silhouette": k_sil},
    )

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  model            : {config.KMEANS_MODEL_DIR}")
    print(f"  zone -> cluster  : {config.ZONE_CLUSTERS_PARQUET}")
    print(f"  prediction rule  : {config.CLUSTER_PROFILE_PARQUET}")
    print(f"  metadata         : {config.KMEANS_METADATA_JSON}")
    print(f"  sweep curves     : {config.KSWEEP_CSV}")
    print("  the streaming job loads this exact pipeline — it never retrains")
    print("=" * 78)

    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
