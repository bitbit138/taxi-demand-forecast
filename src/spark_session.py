"""Shared SparkSession builder — every Spark script in this project goes through here.

Keeps the version matrix, jar coordinates, Ivy cache and timezone policy in one place
so batch and streaming jobs are configured identically.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def get_spark(
    app_name: str,
    packages: str | None = None,
    shuffle_partitions: int | None = None,
) -> SparkSession:
    """Build (or fetch) the local SparkSession.

    Args:
        app_name: shown in the Spark UI.
        packages: Maven coordinates for ``spark.jars.packages`` — pass
            ``config.SPARK_PACKAGES_KAFKA_ONLY`` / ``_SEDONA_ONLY`` so a job only
            resolves the jars it needs. None means no extra jars.
        shuffle_partitions: override ``config.SPARK_SHUFFLE_PARTITIONS``.

    Raises:
        RuntimeError: if JAVA_HOME points at a JDK Spark 3.5 cannot run on.
    """
    config.require_supported_java()
    config.configure_hadoop_home()

    # Spark builds RPC URLs from the machine hostname. An underscore (e.g. "Lee_laptop")
    # is illegal in a URI authority, so the driver dies with
    # "Invalid Spark URL: spark://HeartbeatReceiver@...". Pinning the local hostname
    # sidesteps it and is correct anyway for a local[*] master.
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    # Executors spawn their Python worker from PATH, which here is a different
    # interpreter than the venv driver -> "Python worker failed to connect back".
    # Both ends must be the same interpreter.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    # This repo path contains spaces ("Year 3", "Semester B"), which trips PySpark's
    # SPARK_HOME auto-detection and prints a spurious "Missing Python executable"
    # warning. Setting it from the installed package removes the guesswork.
    os.environ.setdefault("SPARK_HOME", str(Path(pyspark.__file__).resolve().parent))

    builder = (
        SparkSession.builder.appName(app_name)
        .master(config.SPARK_MASTER)
        .config("spark.driver.memory", config.SPARK_DRIVER_MEMORY)
        .config("spark.driver.host", "localhost")
        .config(
            "spark.sql.shuffle.partitions",
            shuffle_partitions or config.SPARK_SHUFFLE_PARTITIONS,
        )
        # Keep resolved jars in-repo: the first --packages run is the only download.
        .config("spark.jars.ivy", str(config.SPARK_IVY_DIR))
        # TLC parquet stores NYC wall-clock times. A UTC session timezone reads those
        # values back unshifted; the real NY<->UTC alignment against Open-Meteo is done
        # explicitly in features.py rather than implicitly by the session timezone.
        .config("spark.sql.session.timeZone", config.UTC_TZ)
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
        .config("spark.sql.adaptive.enabled", "true")
    )

    if packages:
        builder = builder.config("spark.jars.packages", packages)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def describe(spark: SparkSession) -> None:
    """Print the running versions — cheap proof the version matrix is intact."""
    jvm_version = spark.sparkContext._jvm.System.getProperty("java.version")
    print(f"  Spark   {spark.version}")
    print(f"  Java    {jvm_version}")
    print(f"  Python  {sys.version.split()[0]}")
    print(f"  Master  {spark.sparkContext.master}")
