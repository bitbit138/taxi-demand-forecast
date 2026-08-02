"""Single source of truth for paths, dates, versions and tuning constants.

Every script imports from here — no magic numbers or hard-coded paths anywhere else.
Scripts live under ``src/`` and are run from the repo root, e.g.::

    python -m src.ingest.download_tlc
    spark-submit --packages "$(python -c 'import config;print(config.SPARK_PACKAGES)')" src/batch/features.py
"""

from __future__ import annotations

import calendar
import os
import re
import subprocess
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # TLC monthly parquet, as downloaded
EXTERNAL_DIR = DATA_DIR / "external"  # zone lookup, shapefile, weather, events
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"        # plots, GeoJSON, metrics tables
NOTEBOOKS_DIR = ROOT / "notebooks"

# Named artifacts (referenced across phases)
DEMAND_PARQUET = PROCESSED_DIR / "demand.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"
ZONE_CENTROIDS_PARQUET = PROCESSED_DIR / "zone_centroids.parquet"
WEATHER_PARQUET = EXTERNAL_DIR / "weather.parquet"
EVENTS_CSV = EXTERNAL_DIR / "events.csv"
ZONE_LOOKUP_CSV = EXTERNAL_DIR / "taxi_zone_lookup.csv"
ZONE_SHAPEFILE_DIR = EXTERNAL_DIR / "taxi_zones"          # unzipped shapefile
ZONE_SHAPEFILE = ZONE_SHAPEFILE_DIR / "taxi_zones.shp"

KMEANS_MODEL_DIR = MODELS_DIR / "kmeans"                  # saved PipelineModel
CLUSTER_PROFILE_PARQUET = MODELS_DIR / "cluster_demand.parquet"  # cluster -> mean demand

STREAM_OUTPUT_DIR = PROCESSED_DIR / "stream_predictions"
STREAM_CHECKPOINT_DIR = PROCESSED_DIR / "_checkpoints"

for _d in (RAW_DIR, EXTERNAL_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Date range — iterate on the 3-month sample, switch to full-year for finals
# --------------------------------------------------------------------------- #
SAMPLE_MONTHS = ["2024-01", "2024-02", "2024-03"]
FULL_YEAR_MONTHS = [f"2024-{m:02d}" for m in range(1, 13)]

# Flip via env var so no code edit is needed: TAXI_MONTHS=full
USE_FULL_YEAR = os.getenv("TAXI_MONTHS", "sample").lower() == "full"
MONTHS = FULL_YEAR_MONTHS if USE_FULL_YEAR else SAMPLE_MONTHS

_first_year, _first_month = (int(x) for x in MONTHS[0].split("-"))
_last_year, _last_month = (int(x) for x in MONTHS[-1].split("-"))
START_DATE = f"{_first_year:04d}-{_first_month:02d}-01"                     # inclusive
END_DATE = (                                                                # inclusive
    f"{_last_year:04d}-{_last_month:02d}-"
    f"{calendar.monthrange(_last_year, _last_month)[1]:02d}"
)

# Fraction of the (time-ordered) range used for training; the tail is the test set.
TRAIN_FRACTION = 0.8

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Version matrix — Spark 3.5.3 / Scala 2.12 / Java 17 / Sedona 1.9.0
# One mismatch here costs hours; see the gotcha list in PROJECT_PLAN.md.
# --------------------------------------------------------------------------- #
SPARK_VERSION = "3.5.3"
SCALA_VERSION = "2.12"
SEDONA_VERSION = "1.9.0"
GEOTOOLS_WRAPPER_VERSION = "1.9.0-33.5"  # exact pairing from Sedona's maven-coordinates page

KAFKA_PACKAGE = f"org.apache.spark:spark-sql-kafka-0-10_{SCALA_VERSION}:{SPARK_VERSION}"
SEDONA_PACKAGE = f"org.apache.sedona:sedona-spark-shaded-3.5_{SCALA_VERSION}:{SEDONA_VERSION}"
GEOTOOLS_PACKAGE = f"org.datasyslab:geotools-wrapper:{GEOTOOLS_WRAPPER_VERSION}"

# Comma-separated string for --packages / spark.jars.packages
SPARK_PACKAGES = ",".join([KAFKA_PACKAGE, SEDONA_PACKAGE, GEOTOOLS_PACKAGE])
SPARK_PACKAGES_KAFKA_ONLY = KAFKA_PACKAGE
SPARK_PACKAGES_SEDONA_ONLY = ",".join([SEDONA_PACKAGE, GEOTOOLS_PACKAGE])

# Local Spark defaults (laptop-sized).
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
SPARK_SHUFFLE_PARTITIONS = int(os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
# Ivy cache: keep resolved jars in-repo so the first --packages run is the only download.
SPARK_IVY_DIR = ROOT / ".ivy2"

SUPPORTED_JAVA_MAJORS = (8, 11, 17)  # Spark 3.5.x runs on these only

# Windows only: Hadoop's RawLocalFileSystem shells out to winutils.exe to set file
# permissions, so any local write fails with "HADOOP_HOME and hadoop.home.dir are
# unset" without it. Binaries are the community cdarlint/winutils build for
# Hadoop 3.3.6, compatible with the 3.3.4 client jars PySpark 3.5.3 ships.
HADOOP_HOME = ROOT / "hadoop"
WINUTILS_EXE = HADOOP_HOME / "bin" / "winutils.exe"


def configure_hadoop_home() -> None:
    """Point HADOOP_HOME at the bundled winutils (no-op off Windows)."""
    if os.name != "nt":
        return
    if not WINUTILS_EXE.exists():
        raise RuntimeError(
            f"{WINUTILS_EXE} is missing — Spark cannot write local files on Windows.\n"
            "Fetch winutils.exe + hadoop.dll (hadoop-3.3.6) into hadoop/bin; see README."
        )
    os.environ.setdefault("HADOOP_HOME", str(HADOOP_HOME))
    os.environ.setdefault("hadoop.home.dir", str(HADOOP_HOME))
    # hadoop.dll must be loadable from PATH for the native IO calls.
    bin_dir = str(HADOOP_HOME / "bin")
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def require_supported_java() -> str:
    """Fail fast if JAVA_HOME is missing or points at an unsupported JDK.

    Spark 3.5.3 does not run on Java 21/25 — without this check the failure
    surfaces as an opaque Py4J gateway error. Call this before building a
    SparkSession. Returns the resolved JAVA_HOME.
    """
    java_home = os.environ.get("JAVA_HOME", "").strip().rstrip("\\/")
    if not java_home:
        raise RuntimeError(
            "JAVA_HOME is not set. Spark 3.5.3 needs a Java "
            f"{'/'.join(map(str, SUPPORTED_JAVA_MAJORS))} JDK (Temurin 17 recommended)."
        )

    java_exe = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
    if not java_exe.exists():
        raise RuntimeError(f"JAVA_HOME={java_home!r} has no bin/java — check the path.")

    # `java -version` prints to stderr; "17.0.20" -> 17, legacy "1.8.0_x" -> 8.
    out = subprocess.run(
        [str(java_exe), "-version"], capture_output=True, text=True, check=True
    ).stderr
    match = re.search(r'version "(\d+)(?:\.(\d+))?', out)
    if not match:
        raise RuntimeError(f"Could not parse Java version from:\n{out}")
    major = int(match.group(2) or 0) if match.group(1) == "1" else int(match.group(1))

    if major not in SUPPORTED_JAVA_MAJORS:
        raise RuntimeError(
            f"Java {major} at {java_home!r} is not supported by Spark {SPARK_VERSION}. "
            f"Point JAVA_HOME at a Java {'/'.join(map(str, SUPPORTED_JAVA_MAJORS))} JDK."
        )
    return java_home


# --------------------------------------------------------------------------- #
# Kafka
# --------------------------------------------------------------------------- #
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = "taxi-trips"
KAFKA_NUM_PARTITIONS = 3
KAFKA_REPLICATION_FACTOR = 1

# Producer replay: how much faster than wall-clock the historical stream is replayed.
REPLAY_SPEEDUP = 3600.0        # 1 simulated hour per real second
REPLAY_MAX_MESSAGES = 200_000  # safety cap for demos; None = replay everything

# --------------------------------------------------------------------------- #
# Streaming windows
# --------------------------------------------------------------------------- #
WINDOW_DURATION = "1 hour"     # tumbling window == the batch aggregation grain
WATERMARK_DELAY = "2 hours"    # tolerated event-time lateness

# --------------------------------------------------------------------------- #
# Data source URLs
# --------------------------------------------------------------------------- #
TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
TLC_PARQUET_URL = TLC_BASE_URL + "/yellow_tripdata_{month}.parquet"  # month = YYYY-MM
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_SHAPEFILE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY_VARS = ["temperature_2m", "precipitation"]
OPEN_METEO_SLEEP_SECONDS = 1.0   # politeness delay between calls
OPEN_METEO_BATCH_SIZE = 20       # centroids per burst before a longer pause

# --------------------------------------------------------------------------- #
# Timezones — TLC timestamps are local NYC, Open-Meteo returns UTC
# --------------------------------------------------------------------------- #
LOCAL_TZ = "America/New_York"
UTC_TZ = "UTC"

# --------------------------------------------------------------------------- #
# Cleaning thresholds (clean_aggregate.py)
# --------------------------------------------------------------------------- #
EXCLUDED_ZONE_IDS = [264, 265]   # "Unknown" / "N/A"
VALID_ZONE_MIN, VALID_ZONE_MAX = 1, 263
MIN_TRIP_DISTANCE = 0.0          # exclusive: distance must be > 0
MAX_TRIP_DISTANCE = 100.0        # miles
MIN_FARE_AMOUNT = 0.0            # exclusive
MAX_FARE_AMOUNT = 1000.0
MIN_PASSENGER_COUNT, MAX_PASSENGER_COUNT = 1, 8
MAX_TRIP_DURATION_MINUTES = 24 * 60

# A single cleaning filter removing more than this fraction of rows is treated as
# suspicious: it more likely signals a wrong assumption than dirty data, and would
# silently distort demand. clean_aggregate.py refuses to write in that case.
FUNNEL_ALERT_FRACTION = 0.10

# Explicit read schema — never use mergeSchema (schema drifts across months).
TLC_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
]

# --------------------------------------------------------------------------- #
# Feature engineering (features.py)
# --------------------------------------------------------------------------- #
FOURIER_TERMS = 3                # harmonics on the 168-hour week signal
HOURS_PER_WEEK = 168

# --------------------------------------------------------------------------- #
# Zone policy — owned by src/batch/zone_policy.py, applied by features.py
# --------------------------------------------------------------------------- #
# Zero-fill: every kept zone gets a row for EVERY hour of the range, with absent
# (zone, hour) bins set to trip_count = 0, so each zone's series is continuous.
ZERO_FILL_DEMAND = True

# Exclusion floor: a zone averaging fewer than this many trips per day over the
# range is dropped from the modelling set. Below one trip a day a zone's hourly
# profile is ~99% zeros — effectively the zero vector — so K-Means would spend a
# cluster separating "no demand" from everything else and the silhouette would be
# flattered by that trivially separable group. Expressed per day so it scales
# unchanged from the 3-month sample to the full year.
MIN_ZONE_TRIPS_PER_DAY = 1.0
MODELING_ZONES_PARQUET = PROCESSED_DIR / "modeling_zones.parquet"

# --------------------------------------------------------------------------- #
# Modelling (train_kmeans.py / evaluate.py)
# --------------------------------------------------------------------------- #
K_MIN, K_MAX = 2, 12             # elbow + silhouette sweep range
KMEANS_MAX_ITER = 50
MOVING_AVG_WINDOW = 24 * 7       # hours in the moving-average baselines
EWMA_ALPHA = 0.3
MAPE_EPSILON = 1.0               # guard against divide-by-zero on empty bins

# --------------------------------------------------------------------------- #
# Curated NYC event dates (build_events.py adds holidays on top)
# --------------------------------------------------------------------------- #
NYC_EVENT_DATES = {
    "2024-01-01": "New Year's Day celebrations",
    "2024-02-11": "Super Bowl LVIII",
    # The parade moves to the preceding Saturday when 17 March is a Sunday (as in
    # 2024) so it does not clash with Mass. The parade is the demand driver; the day
    # itself still generates evening bar traffic, so both are listed.
    "2024-03-16": "St. Patrick's Day Parade",
    "2024-03-17": "St. Patrick's Day",
    "2024-04-08": "Solar eclipse",
    "2024-06-30": "NYC Pride March",
    "2024-07-04": "Independence Day fireworks",
    "2024-08-26": "US Open begins",
    "2024-09-24": "UN General Assembly high-level week",
    "2024-11-03": "TCS NYC Marathon",
    "2024-11-28": "Macy's Thanksgiving Day Parade",
    "2024-12-31": "New Year's Eve, Times Square",
}
