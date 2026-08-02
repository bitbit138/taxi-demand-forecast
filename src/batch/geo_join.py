"""Load the taxi-zone shapefile with Apache Sedona -> centroids + polygons.

``data/external/taxi_zones/taxi_zones.shp`` -> ``data/processed/zone_centroids.parquet``
with one row per zone: ``zone_id, zone_name, borough, lat, lon, area_km2, geometry_wkt``.

Trips already carry ``PULocationID``, so trip -> zone needs no spatial join; the genuine
spatial work is **polygon -> centroid -> nearest weather grid point** (feeding
``fetch_weather.py``) and **polygon -> map rendering** (Phase 7 Folium + GeoJSON). This
distinction belongs in the report.

**Projection.** The shapefile is EPSG:2263 (NY State Plane, *feet*). Centroids are
computed in that projected CRS — where a centroid is metrically meaningful — and only
then transformed to EPSG:4326. Areas are likewise computed in 2263 and converted from
square feet to km². This mirrors ``fetch_weather.py`` exactly, and the two are
cross-checked against each other below: Sedona's centroids must agree with the
geopandas ones already used for the weather fetch, which also pins down EPSG:4326
axis order rather than trusting it.

    python -m src.batch.geo_join
"""

from __future__ import annotations

import sys
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sedona.spark import SedonaContext

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.spark_session import describe, get_spark  # noqa: E402

SOURCE_CRS = "EPSG:2263"   # NY State Plane Long Island, US survey feet
TARGET_CRS = "EPSG:4326"   # WGS84 lat/lon
SQFT_PER_SQKM = 10_763_910.4
# Sedona and geopandas should agree to well under a metre; 1e-6 deg is ~0.1 m.
CENTROID_TOLERANCE_DEG = 1e-5


def load_zones(sedona: SparkSession) -> DataFrame:
    """Read the shapefile and derive centroids, areas and WGS84 polygons."""
    zones = sedona.read.format("shapefile").load(str(config.ZONE_SHAPEFILE_DIR))

    # The reader gives the raw geometry plus the .dbf attribute columns.
    zones = zones.select(
        F.col("LocationID").cast("int").alias("zone_id"),
        F.col("zone").alias("zone_name"),
        F.col("borough").alias("borough"),
        F.col("geometry").alias("geom_2263"),
    ).filter(
        F.col("zone_id").between(config.VALID_ZONE_MIN, config.VALID_ZONE_MAX)
        & ~F.col("zone_id").isin(config.EXCLUDED_ZONE_IDS)
    )

    zones = zones.withColumn(
        # Centroid in the projected CRS first — see module docstring.
        "centroid_4326",
        F.expr(
            f"ST_Transform(ST_SetSRID(ST_Centroid(geom_2263), 2263), "
            f"'{SOURCE_CRS}', '{TARGET_CRS}')"
        ),
    ).withColumn(
        "geom_4326",
        F.expr(
            f"ST_Transform(ST_SetSRID(geom_2263, 2263), '{SOURCE_CRS}', '{TARGET_CRS}')"
        ),
    )

    return zones.select(
        "zone_id",
        "zone_name",
        "borough",
        # Sedona orders EPSG:4326 as (longitude, latitude); the cross-check against
        # geopandas below fails loudly if that ever changes.
        F.expr("ST_Y(centroid_4326)").alias("lat"),
        F.expr("ST_X(centroid_4326)").alias("lon"),
        (F.expr("ST_Area(geom_2263)") / F.lit(SQFT_PER_SQKM)).alias("area_km2"),
        F.expr("ST_AsText(geom_4326)").alias("geometry_wkt"),
    )


def check_against_geopandas(zones: DataFrame) -> bool:
    """Sedona centroids must match the geopandas ones used for the weather fetch."""
    from src.ingest.fetch_weather import load_centroids

    reference = load_centroids().rename(columns={"lat": "gpd_lat", "lon": "gpd_lon"})
    spark = zones.sparkSession
    ref = spark.createDataFrame(reference)

    joined = zones.join(ref, on="zone_id", how="inner").select(
        "zone_id",
        F.abs(F.col("lat") - F.col("gpd_lat")).alias("d_lat"),
        F.abs(F.col("lon") - F.col("gpd_lon")).alias("d_lon"),
    )
    worst = joined.agg(
        F.max("d_lat").alias("max_d_lat"),
        F.max("d_lon").alias("max_d_lon"),
        F.count(F.lit(1)).alias("compared"),
    ).first()

    max_delta = max(float(worst["max_d_lat"]), float(worst["max_d_lon"]))
    print(f"  cross-check vs geopandas : {int(worst['compared'])} zones compared, "
          f"max delta {max_delta:.3e} deg (~{max_delta * 111_000:.3f} m)")

    if max_delta > CENTROID_TOLERANCE_DEG:
        print("    MISMATCH — Sedona and geopandas disagree. If lat/lon look "
              "swapped, EPSG:4326 axis order has changed.")
        joined.orderBy(F.desc("d_lat")).show(5, truncate=False)
        return False
    print("    -> agrees with the centroids already used for weather; "
          "lon/lat axis order confirmed")
    return True


def validate(zones: DataFrame, spark: SparkSession) -> bool:
    """Uniqueness, coverage and a clean 1:1 join back to PULocationID."""
    ok = True
    print("\n" + "=" * 78)
    print("VALIDATION")
    print("=" * 78)

    stats = zones.agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct("zone_id").alias("distinct_zones"),
        F.min("lat").alias("min_lat"), F.max("lat").alias("max_lat"),
        F.min("lon").alias("min_lon"), F.max("lon").alias("max_lon"),
        F.sum("area_km2").alias("total_area"),
    ).first()

    n_rows, n_zones = int(stats["rows"]), int(stats["distinct_zones"])
    print(f"  rows / distinct zones    : {n_rows} / {n_zones}")
    unique = n_rows == n_zones
    print(f"  one row per zone         : {unique}")
    ok = ok and unique

    expected = config.VALID_ZONE_MAX - config.VALID_ZONE_MIN + 1
    print(f"  expected zones           : {expected}")
    ok = ok and n_zones == expected

    print(f"  lat range                : {stats['min_lat']:.4f} .. {stats['max_lat']:.4f}")
    print(f"  lon range                : {stats['min_lon']:.4f} .. {stats['max_lon']:.4f}")
    in_nyc = (
        40.4 <= stats["min_lat"] and stats["max_lat"] <= 41.0
        and -74.3 <= stats["min_lon"] and stats["max_lon"] <= -73.6
    )
    print(f"  centroids inside NYC box : {in_nyc}")
    ok = ok and in_nyc
    print(f"  total area               : {stats['total_area']:,.1f} km2 "
          "(NYC is ~1,220 km2 incl. water)")

    nulls = zones.filter(
        F.col("lat").isNull() | F.col("lon").isNull() | F.col("geometry_wkt").isNull()
    ).count()
    print(f"  null geometry/centroid   : {nulls}")
    ok = ok and nulls == 0

    ok = check_against_geopandas(zones) and ok

    # --- the join that matters: demand.PULocationID -> zone_centroids.zone_id ---
    print("\n  join back to demand.parquet (PULocationID -> zone_id):")
    demand = spark.read.parquet(str(config.DEMAND_PARQUET))
    demand_rows = demand.count()
    demand_zones = demand.select("PULocationID").distinct()

    unmatched = demand_zones.join(
        zones, demand_zones["PULocationID"] == zones["zone_id"], "left_anti"
    ).count()
    print(f"    demand zones unmatched : {unmatched}")
    ok = ok and unmatched == 0

    joined_rows = demand.join(
        zones.select("zone_id", "lat", "lon"),
        demand["PULocationID"] == F.col("zone_id"),
        "left",
    ).count()
    no_fanout = joined_rows == demand_rows
    print(f"    rows before / after    : {demand_rows:,} / {joined_rows:,}")
    print(f"    strict 1:1, no fan-out : {no_fanout}")
    ok = ok and no_fanout

    # Every modelling zone must have geometry, or the maps will have holes.
    if config.MODELING_ZONES_PARQUET.exists():
        modeling = spark.read.parquet(str(config.MODELING_ZONES_PARQUET)).select("zone_id")
        n_model = modeling.count()
        missing_geom = modeling.join(zones, on="zone_id", how="left_anti").count()
        print(f"\n  modelling zones          : {n_model}")
        print(f"    without geometry       : {missing_geom}")
        ok = ok and missing_geom == 0

    print("=" * 78)
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def main() -> int:
    if not config.ZONE_SHAPEFILE.exists():
        print(f"ERROR: {config.ZONE_SHAPEFILE} not found — run "
              "python -m src.ingest.download_tlc", file=sys.stderr)
        return 2

    spark = get_spark("geo-join", packages=config.SPARK_PACKAGES_SEDONA_ONLY)
    sedona = SedonaContext.create(spark)

    print("=" * 78)
    print("Sedona geo join — taxi zone centroids + polygons")
    print("=" * 78)
    describe(spark)
    print(f"  sedona    {config.SEDONA_VERSION} (geotools-wrapper "
          f"{config.GEOTOOLS_WRAPPER_VERSION})")
    print(f"  shapefile {config.ZONE_SHAPEFILE}")
    print(f"  transform {SOURCE_CRS} -> {TARGET_CRS}")

    zones = load_zones(sedona)
    zones.cache()

    ok = validate(zones, sedona)

    print("\nLargest zones by area:")
    zones.select("zone_id", "zone_name", "borough", "area_km2", "lat", "lon").orderBy(
        F.desc("area_km2")
    ).limit(5).show(truncate=False)

    zones.coalesce(1).write.mode("overwrite").parquet(str(config.ZONE_CENTROIDS_PARQUET))

    print("=" * 78)
    print("DONE")
    print(f"  wrote : {config.ZONE_CENTROIDS_PARQUET}")
    print("  cols  : zone_id, zone_name, borough, lat, lon, area_km2, geometry_wkt")
    print("  use   : centroids -> weather join; polygons -> Folium + GeoJSON (Phase 7)")
    print("=" * 78)

    sedona.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
