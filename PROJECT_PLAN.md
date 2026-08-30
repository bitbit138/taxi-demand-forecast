# Transportation Demand Forecast — Implementation Plan
Team: Lee Rosenblit, Tom Bitran.

Forecast NYC taxi **demand per `zone × time-window`**, fusing TLC trips + weather + events (novelty #1),
with a real-time live-prediction path (novelty #2). Kafka handles streaming, Spark handles processing
and modeling.

---

## Locked setup (all decisions already made)

> Two of these evolved during implementation and are documented in [REPORT.md](REPORT.md):
> the headline metric is **WAPE** (MAPE is undefined on the ~46% zero bins; MAPE on non-zero
> cells is still reported), and the K-Means prediction rule is **cluster shape × the zone's
> own level** rather than the raw pooled cluster mean (both are scored in the ladder).

| Item | Decision |
| --- | --- |
| Compute environment | Local **Docker Compose** (Kafka in KRaft mode) + local **PySpark**. Self-contained, free, reproducible. |
| Data | NYC **yellow-taxi** TLC parquet. Iterate on **2024-01…2024-03**, produce final results on **full-year 2024** (2016+ only, since zone IDs start then). |
| Demand target | `trip count` per `(pickup_zone, date, hour)`. |
| Streaming | A producer **replays historical parquet as a simulated live stream** into Kafka; Spark Structured Streaming consumes it. |
| Model | Spark MLlib **K-Means** over zone-level demand profiles → predict a query's demand as its **cluster mean**. Trained once in batch, **saved, and reused unchanged** in the streaming job. |
| Baselines to beat | historical average + moving average + weighted MA + exponentially-weighted MA. |
| Metrics | **MAE, RMSE, MAPE.** |
| Cluster count K | chosen automatically by **elbow (WCSS) + silhouette**, not hand-picked. |
| Stack | Python 3.11 · Java 17 (Temurin) · **Spark 3.5.3** · Scala 2.12 · Sedona for Spark 3.5 · Kafka KRaft single broker. |

---

## Phase 1 — Environment & repo

- [x] Install **Java 17** (Temurin), **Python 3.11** + venv, **Docker + Docker Compose** (Windows: WSL2 backend).
- [x] `requirements.txt`: `pyspark==3.5.3`, `apache-sedona`, `geopandas`, `shapely`, `folium`, `requests`, `pandas`, `pyarrow`, `holidays`.
- [x] Spark launched with the matching Kafka connector `org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3` and the Sedona jars for Spark 3.5 (`sedona-spark-shaded-3.5_2.12` + the paired `geotools-wrapper` — take the exact geotools-wrapper string from Sedona's install-python page).
- [x] Scaffold the repo:
```
taxi-demand-forecast/
  docker-compose.yml          # Kafka (KRaft, no Zookeeper)
  requirements.txt  config.py  README.md
  data/{raw,external,processed}/  models/  notebooks/
  src/ingest/  download_tlc.py  fetch_weather.py  build_events.py
  src/batch/   clean_aggregate.py  geo_join.py  features.py  train_kmeans.py  evaluate.py
  src/stream/  producer.py  spark_stream.py  predict_live.py
  src/viz/     make_maps.py
```
- [x] `docker-compose.yml` = single Kafka broker in KRaft mode; topic `taxi-trips`.
- [x] Smoke test: `docker compose up`, produce & consume one console message.

## Phase 2 — Data acquisition (`src/ingest/`)

- [x] **`download_tlc.py`** — pull yellow-taxi monthly parquet from
  `https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_YYYY-MM.parquet` for the months in `config.py` → `data/raw/`.
- [x] Download `taxi_zone_lookup.csv` + the Taxi Zone **shapefile** into `data/external/`.
- [x] **`fetch_weather.py`** — Open-Meteo historical archive (free, no key), one call per **zone centroid** covering the whole date range, cached to `data/external/weather.parquet`. Batch + sleep for rate limits.
  `https://archive-api.open-meteo.com/v1/archive?latitude=..&longitude=..&start_date=..&end_date=..&hourly=temperature_2m,precipitation`
- [x] **`build_events.py`** — `holidays` lib + a short curated list of big NYC event dates → `events.csv` with `is_holiday` / `is_event` flags.

## Phase 3 — Batch processing in Spark (`src/batch/`)

- [x] **`clean_aggregate.py`** — read `data/raw/*.parquet`. **Handle schema drift** with an explicit column select + casts (2025+ files add `cbd_congestion_fee`; casings vary by year). Filter bad rows (non-positive fare/distance, absurd distances, out-of-range passenger count, pickup ≥ dropoff, out-of-range timestamps). Drop zones **264/265** (Unknown/N/A). Aggregate to `(PULocationID, date, hour) -> trip_count` → `data/processed/demand.parquet`.
- [x] **`geo_join.py`** — Sedona (`SedonaContext.create(spark)`): read the zone shapefile, compute **zone centroids** (for the weather join) and keep polygons (for maps). *(Trips already carry LocationID, so trip→zone is a lookup; the real spatial work is polygon → centroid → nearest weather point + map rendering — state this in the report.)*
- [x] **`features.py`** — build the modeling table:
  - **zero-fill** empty `(zone, hour)` bins so each zone's series is continuous;
  - calendar features + **cyclical sin/cos** encodings of hour and day-of-week + a small set of **Fourier terms** on the hour-of-week signal (daily/weekly periodicity);
  - weather `temp`/`precip` joined on zone+hour (**align timezones**: TLC = America/New_York, Open-Meteo = UTC);
  - event/holiday flags;
  - `historical_avg_demand(zone, hour, dow)` (also a baseline).
  → `data/processed/features.parquet`.

## Phase 4 — Modeling (`src/batch/train_kmeans.py`, `evaluate.py`)

- [x] Time-based **train/test split** (train earlier weeks, test later), fixed seed.
- [x] **`train_kmeans.py`** — `VectorAssembler` + `StandardScaler` + `KMeans`; sweep **K=2…12** for WCSS (elbow) and **silhouette**; pick K from those curves; save the model to `models/`. Compute per-cluster mean demand → the prediction rule.
- [x] **`evaluate.py`** — build the **baseline ladder** (historical average, moving average, weighted MA, exponentially-weighted MA) plus the cluster-mean prediction, and score every method with **MAE, RMSE, MAPE** on the test split. Emit one comparison table (rows = methods, columns = metrics).

## Phase 5 — Streaming: Kafka + Spark (`src/stream/`) — the core

- [x] **`producer.py`** — replay `data/raw` rows as JSON to `taxi-trips`, **time-accelerated**, event-time = `tpep_pickup_datetime`, with a defined message schema.
- [x] **`spark_stream.py`** — Structured Streaming from Kafka (launch with the `--packages` connector): parse JSON, reapply cleaning filters, set a **watermark** on event-time, **tumbling-window** demand per zone, write to console + parquet sinks. **Load the saved batch model** and emit **predicted vs actual** per window — the live path reuses the batch artifact and never retrains.

## Phase 6 — Real-time live prediction (novelty #2) (`src/stream/predict_live.py`)

- [x] Load the same saved model + scalers; take a live query `(zone, timestamp, temp, precip, is_event)` → build the feature vector → assign cluster → return **predicted demand** for the next window.

## Phase 7 — Evaluation & answering the open questions

- [x] Produce: elbow plot, silhouette score, the **baseline-ladder metrics table**, cluster **maps** (Folium **and** a per-time-window **GeoJSON** export), and a `zone × hour` demand heatmap.
- [x] Answer the four proposal questions with evidence: best K (elbow/silhouette); do weather & events beat trips-only (**ablation**: train/eval with vs without those features); is streaming worth it vs batch (note streaming latency/throughput); multi-zone trips (assigned to **pickup** zone — state it).

## Phase 8 — Deliverables & submission

- [x] **Presentation** — delivered; the final deliverable is code + results (REPORT.md, notebook, reports/).
- [x] **Short report** referencing the three proposal papers (Sedona/GeoSpark, DMVST-Net, NYC-taxi VAST) — what you used from each.
- [x] **`README.md`**: exact run order — `docker compose up` → download → weather → batch clean/geo/features → train → evaluate → producer → stream — with pinned versions.
- [x] Reproducibility: fixed seeds, pinned `requirements.txt` and jar coordinates. Raw TLC parquet
  is not committed (re-fetched by URL); everything derived from it — `data/external/`,
  `data/processed/`, `models/`, `reports/` (~60 MB) — is, which is what makes every reported
  number checkable and every demo runnable from a clone.
- [ ] Submit on time.

---

## Gotcha list (the things that actually eat days)

1. **Version matrix**: Spark 3.5.3 == Kafka connector 3.5.3; Sedona jar for Spark 3.5; Scala 2.12; Java 17. One mismatch = hours lost.
2. **Kafka `--packages` first run** downloads jars from Maven — needs network; pre-cache in Docker.
3. **Schema drift across months** (`cbd_congestion_fee` in 2025+, casing changes) — explicit column select + cast, never `mergeSchema`.
4. **Timezone mismatch** TLC (local NYC) vs Open-Meteo (UTC) — align before the hourly join.
5. **Zone IDs 264/265 = Unknown** — drop them so they don't pollute clusters.
6. **Never `collect()` / `toPandas()` big data** — keep it distributed; sample only for plots.
7. **Sedona API** = `SedonaContext.create(spark)` in current versions.
8. **Open-Meteo call volume** — one archive call per zone centroid for the whole range, cached to parquet.
9. **Fixed seeds** for KMeans + split, or metrics won't reproduce.
