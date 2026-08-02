# Transportation Demand Forecast — NYC Taxi

Course 10351 (ניתוח נתוני עתק / Big Data Analytics), Afeka.
Team: Lee Rosenblit, Tom Bitran.

Forecast NYC yellow-taxi **demand per `zone × hour`** by fusing TLC trip records with weather
(Open-Meteo) and event/holiday flags. **Kafka** replays historical parquet as a simulated live
stream; **Spark Structured Streaming** consumes it. **K-Means (Spark MLlib)** discovers
demand-pattern clusters and a query's demand is predicted from its cluster mean, scored against a
ladder of baselines with **MAE / RMSE / MAPE**.

The model is trained **once in batch, saved, and reloaded unchanged by the streaming job** — the
live path never retrains, so batch and live predictions agree by construction. Re-running
`train_kmeans.py` is the way to refresh clusters.

---

## Version matrix — do not mix

| Component | Version |
| --- | --- |
| Python | 3.11 |
| Java | 17 (Temurin) |
| Spark / PySpark | 3.5.3 |
| Scala | 2.12 |
| Kafka connector | `org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3` |
| Sedona | 1.9.0 (`sedona-spark-shaded-3.5_2.12`) |
| GeoTools wrapper | `org.datasyslab:geotools-wrapper:1.9.0-33.5` |
| Kafka broker | 3.9.0, KRaft mode, single node |

All of these live in [config.py](config.py) — `config.SPARK_PACKAGES` builds the `--packages`
string, so nothing is hard-coded in the scripts. `config.require_supported_java()` aborts with a
readable message if `JAVA_HOME` points at a JDK Spark 3.5 cannot use (21, 25, …), instead of
failing later as an opaque Py4J gateway error.

---

## Setup

```powershell
# 1. Java 17 — Spark reads JAVA_HOME, not PATH, so point it at the JDK 17 root.
#    (A newer JDK may still be first on PATH; that is fine and does not affect Spark.)
[Environment]::SetEnvironmentVariable(
  'JAVA_HOME', 'C:\Program Files\Eclipse Adoptium\jdk-17.0.20.8-hotspot', 'User')
$env:JAVA_HOME = [Environment]::GetEnvironmentVariable('JAVA_HOME','User')  # this session
& "$env:JAVA_HOME\bin\java.exe" -version                                    # -> 17.x

# 2. Python 3.11 venv — the launcher picks the right interpreter even if
#    `python` is a newer version.
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows
# source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt

# 3. Windows only — Hadoop needs winutils.exe to write local files.
#    Already committed under hadoop/bin; config.configure_hadoop_home() wires it up.
#    If missing, fetch winutils.exe + hadoop.dll for hadoop-3.3.6 into hadoop/bin.

# 3. Kafka (Docker Desktop with the WSL2 backend on Windows)
docker compose up -d
docker compose ps                 # kafka: healthy, kafka-init: exited (0)
```

### Kafka smoke test

```bash
# consumer (leave running)
docker exec -it taxi-kafka /opt/kafka/bin/kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 --topic taxi-trips --from-beginning

# producer (another terminal)
docker exec -it taxi-kafka /opt/kafka/bin/kafka-console-producer.sh \
  --bootstrap-server localhost:9092 --topic taxi-trips
```

> **Git Bash on Windows** rewrites `/opt/kafka/...` into a Windows path and the exec
> fails with `stat C:/Program Files/Git/opt/kafka/...: no such file or directory`.
> Prefix with `MSYS_NO_PATHCONV=1`, or just use PowerShell / CMD.

---

## Run order

> Steps 1–10 are implemented; the rest land phase by phase (see `PROJECT_PLAN.md`).

| # | Command | Output |
| --- | --- | --- |
| 1 | `docker compose up -d` | Kafka broker + `taxi-trips` topic |
| 2 | `python -m src.ingest.download_tlc` | `data/raw/yellow_tripdata_2024-0*.parquet`, `data/external/` |
| 3 | `python -m src.ingest.fetch_weather` | `data/external/weather.parquet` |
| 4 | `python -m src.ingest.build_events` | `data/external/events.csv` |
| 5 | `python -m src.batch.clean_aggregate` | `data/processed/demand.parquet` |
| 6a | `python -m src.batch.zone_policy` | `data/processed/modeling_zones.parquet` |
| 6b | `python -m src.batch.geo_join` | `data/processed/zone_centroids.parquet` |
| 7 | `python -m src.batch.features` | `data/processed/features.parquet` |
| 8 | `python -m src.batch.train_kmeans` | `models/kmeans/`, elbow + silhouette plots |
| 9 | `python -m src.batch.evaluate` | baseline-ladder metrics table |
| 10 | `python -m src.stream.producer --reset-topic` | replays trips into Kafka |
| 11 | `python -m src.stream.spark_stream --validate` | windowed predicted-vs-actual demand |
| 12 | `python -m src.stream.predict_live` | single live query -> predicted demand |
| 13 | `python -m src.viz.make_maps` | Folium map, GeoJSON, heatmap |

Scripts are run from the repo root so that `import config` resolves.

### Sample vs full year

Default is the **2024-01…2024-03** sample. For the final run:

```bash
TAXI_MONTHS=full python -m src.ingest.download_tlc      # ~12 monthly files
$env:TAXI_MONTHS="full"                                 # PowerShell equivalent
```

---

## Repository layout

```
config.py               all paths, dates, versions, thresholds, seeds
docker-compose.yml      Kafka KRaft single broker + topic init
data/raw/               TLC monthly parquet (git-ignored)
data/external/          zone lookup, shapefile, weather, events
data/processed/         demand.parquet, features.parquet, stream output
models/                 saved KMeans PipelineModel + cluster profile
reports/                plots, GeoJSON, metrics tables
src/ingest/             download_tlc, fetch_weather, build_events
src/batch/              clean_aggregate, geo_join, features, train_kmeans, evaluate
src/stream/             producer, spark_stream, predict_live
src/viz/                make_maps
notebooks/              exploration only — pipeline logic lives in src/
```

## Data notes worth putting in the report

**Weather is coarser than the zone grid.** Open-Meteo's ERA5 archive has a ~11 km cell,
but NYC's 263 taxi zones span roughly 40 km. The 263 centroids therefore collapse to
only **27 distinct weather series** (largest cell covers 39 zones, median 3). Weather
consequently varies far more across *time* than across *zones*, which is exactly what the
feature-ablation in Phase 7 should be interpreted against — it is close to a
citywide temporal signal, not a per-zone one.

**Timezone handling.** Weather is fetched in UTC and the NY-local wall-clock key is
derived once, in `fetch_weather.py`, so `(zone_id, date_local, hour_local)` joins
directly against TLC pickup times. DST is handled by the tz database: local 02:00 on
2024-03-10 does not exist (2183 hours per zone in Q1, not 2184), and on fall-back days
the duplicated wall-clock hour keeps its first (EDT) reading.

**Events cover the whole of 2024** even when `config.MONTHS` is the 3-month sample, so
scaling to the full year needs no regeneration. `is_holiday` and `is_event` are
independent booleans with separate name columns — three 2024 dates are both
(New Year's Day, 4 July, Thanksgiving) and neither label overwrites the other.
`is_federal_holiday` is kept separate from `is_holiday` because the NY-only observances
(Lincoln's Birthday, Susan B. Anthony Day, Election Day) are ordinary working days for
most people and should not be assumed to move demand the way a federal holiday does.

One curated date was corrected: the **St. Patrick's Day Parade was 2024-03-16**, not the
17th — when 17 March falls on a Sunday NYC moves the parade to the preceding Saturday.
The parade and the day itself are now separate entries.

**Zone policy (zero-fill + exclusion).** Owned by `src/batch/zone_policy.py` so
`features.py`, `train_kmeans.py`, `evaluate.py` and `make_maps.py` cannot drift apart.
`demand.parquet` holds observed demand only; `features.py` builds the full
`kept zones × every hour` grid and zero-fills the gaps. Zones averaging under
`MIN_ZONE_TRIPS_PER_DAY = 1.0` are excluded first — **40 of 263 zones, carrying 634 trips
= 0.0070% of demand**, leaving **223 zones and 99.9930% of demand**. The two rules are
paired on purpose: zero-filling *without* the floor would manufacture ~87k all-zero rows
for places with no taxi service at all (Governor's Island has no road access), which is
precisely the degenerate cluster the floor exists to prevent. The exclusion list is
printed in full on every run rather than applied silently.

**Two distinct modelling artifacts — do not conflate them.**

| Artifact | Grain | Shape | Used by |
| --- | --- | --- | --- |
| `features.parquet` | one row per `(zone_id, date_local, hour_local)` | 223 × 2183 = 486,809 | baselines, `evaluate.py` |
| profile matrix (derived in `train_kmeans.py`, not stored) | one row per zone | 223 × 168 `(hour, dow)` | K-Means clustering |

K-Means clusters *zones* by their weekly demand profile; the per-observation table is
what the baseline ladder is scored against. Building the profile matrix inside
`train_kmeans.py` keeps it obvious which one K-Means actually consumes.

**`hist_avg_demand` is fitted on the training split only.** It is simultaneously a
feature and the first rung of the baseline ladder, so computing it over the whole
quarter would leak test-period demand into training and flatter every metric. The split
is time-based (train `2024-01-01..2024-03-12`, test `2024-03-13..2024-03-31`) and
deterministic — no seed involved — and is stored as `is_train` so `evaluate.py` reuses
exactly the same split rather than re-deriving it.

## Streaming demo

The producer replays history as a simulated live feed. **Event time is
`tpep_pickup_datetime` from the data itself**, never producer wall-clock, so the same
replay yields the same windows however fast it is run.

```bash
python -m src.stream.producer --dry-run              # inspect messages, no broker
python -m src.stream.producer --hours 1 --reset-topic   # clean 1-hour demo slice
python -m src.stream.producer --hours 0 --speedup 86400 # whole range, as fast as possible
```

`config.REPLAY_SPEEDUP` (default 600) maps simulated to real time — 1 real second = 10
simulated minutes — and the mapping is printed on every run. This client sustains
~1,000–1,400 msg/s and a busy NYC hour is ~6,600 trips, so above roughly 800× the
producer stops pacing and just emits as fast as it can; it prints a warning when the
requested speedup was not achieved, because that makes a demo's wall-clock timings
machine-dependent.

**Re-running.** Kafka topics are append-only, so a second run appends a duplicate copy
of the slice and would double-count in the windowed aggregation. The producer refuses to
write to a non-empty topic unless given `--reset-topic` (delete and recreate) or
`--append` (deliberate). Manual reset:

```bash
docker exec taxi-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 \
  --delete --topic taxi-trips
docker compose up -d kafka-init --force-recreate
```

### The streaming job

```bash
python -m src.stream.producer --start "2024-01-10 14:00:00" --hours 6 \
       --speedup 86400 --reset-topic
python -m src.stream.spark_stream --run-seconds 120 --fresh --validate
```

`spark_stream.py` imports `build_filters()` and `local_date_hour()` from
`clean_aggregate.py` rather than re-implementing them, and applies the same 223-zone
modelling set, so streamed windows are comparable to `demand.parquet` cell for cell.
`--validate` checks exactly that and refuses to continue on any disagreement.

| Choice | Value | Why |
| --- | --- | --- |
| Event time | `tpep_pickup_datetime` from the payload | windows come from the data, so a replay is reproducible at any speed |
| Window | tumbling 1 hour | same grain as the batch aggregation, so the two are directly comparable |
| Watermark | 2 hours | a trip is recorded at dropoff, so lateness ≈ trip duration; p99.9 is 1.91 h and only 0.092% exceed 2 h |
| Output mode | append | emits each window once, when final — `update` re-emits partial windows with no marker for which is final, making batch comparison impossible; append is also the only mode the parquet sink supports |
| Model | loaded from `models/`, never retrained | the live forecaster is the same artifact `evaluate.py` scored |

The live rule is **cluster shape × zone level**, the interpretable model — deliberately
not `hist_avg`, which is more accurate but has no compact live representation
(37,464 cells versus 895). With a 2-hour watermark the final ~2 simulated hours of any
replay never close, so replay more hours than you intend to validate.

## Windows gotchas already handled

`src/spark_session.py` sets these for every job — listed here because they are the
three things that break PySpark on Windows:

| Symptom | Cause | Handled by |
| --- | --- | --- |
| `Invalid Spark URL: spark://HeartbeatReceiver@Lee_laptop:...` | hostname contains `_`, illegal in a URI authority | `SPARK_LOCAL_HOSTNAME=localhost` |
| `Python worker failed to connect back` | executor launches `python` from PATH, a different interpreter than the venv driver | `PYSPARK_PYTHON = sys.executable` |
| `HADOOP_HOME and hadoop.home.dir are unset` | Hadoop shells out to `winutils.exe` to set local file permissions | `config.configure_hadoop_home()` |

## Reproducibility

Fixed seed (`config.SEED = 42`) for KMeans and the time-based train/test split, pinned
`requirements.txt`, pinned Spark jar coordinates. Raw parquet is not committed; a small sample is.
