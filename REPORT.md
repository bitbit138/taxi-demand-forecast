# Transportation Demand Forecast — Final Report

**Course 10351 — ניתוח נתוני עתק (Big Data Analytics), Afeka.**
Tom Bitran (322373028) · Lee Rosenblit (322357880)

---

## 1. Problem and goal

Cities show a persistent mismatch between shared-vehicle supply and demand that
shifts across space and time: long waits in some zones, empty vehicles circling in
others. Our goal, as approved in the proposal: analyze a year of historical trips and
external signals to **forecast demand per `zone × hour`**, with two claimed novelties —
(1) a **multi-signal model** fusing trips with weather and events, and (2) **real-time
live prediction** from live input (zone, time, conditions), not only batch analysis.

## 2. Data

* **NYC TLC yellow-taxi trip records, full-year 2024** — 12 monthly Parquet files,
  ~41M trips after download, **39.2M** after cleaning; 263 taxi zones.
* **Taxi Zone shapefile** (EPSG:2263) — processed with **Apache Sedona**: centroids
  computed in the projected CRS, transformed to WGS84, cross-checked against an
  independent geopandas computation to <0.1 m.
* **Open-Meteo hourly weather** per zone centroid (temperature, precipitation), with
  UTC→NY-local alignment done once, DST-correctly (the skipped hour on 2024-03-10
  does not exist in the grid).
* **Event/holiday calendar** for all of 2024 — `holidays` library plus a curated NYC
  list (parades, marathon, UN week, NYE), with federal vs NY-only holidays kept
  separate.

## 3. Pipeline

All processing is PySpark 3.5.3 (Java 17, Scala 2.12); Kafka 3.9 (KRaft) carries the
streaming path. Every stage validates its own output and refuses to write on failure.

```text
ingest      download_tlc → fetch_weather → build_events
batch       clean_aggregate → zone_policy → geo_join (Sedona) → features
model       train_kmeans (K-Means, MLlib) → evaluate (baseline ladder) → ablation
stream      producer (Kafka replay) → spark_stream (Structured Streaming) → predict_live
viz         make_maps (figures, Folium map, GeoJSON)
```

Decisions worth defending:

* **Cleaning funnel with a tripwire** — any single filter dropping >10% of rows
  aborts the write: a wrong assumption looks exactly like dirty data.
* **Zone policy** — zones under 1 trip/day are excluded *before* zero-filling
  (full year: 225 of 263 zones kept, >99.99% of demand). Zero-filling without the
  floor would manufacture ~90k all-zero rows and hand K-Means a degenerate cluster.
* **Leakage discipline** — time-based 80/20 split (train 2024-01-01..10-18, test
  10-19..12-31). `hist_avg_demand` is both a feature and a baseline, so it is fitted
  on train rows only; the split is stored in the data (`is_train`) so every
  downstream stage reuses the identical rows.

## 4. Models

**K-Means over demand shape (unsupervised).** Each zone is a 168-dimensional
`(hour × day-of-week)` profile, L1-normalised so K-Means clusters *when* a zone is
busy, not how busy. K is chosen by elbow + silhouette **re-swept on the full year**
(Q1 had chosen K=4; the year's seasonality splits differently). Elbow says K=5,
silhouette says K=2; **K=5** was chosen after inspecting cluster contents:

| Cluster | Character (derived from shape, not hand-named) | Zones | Trips |
|---|---|---|---|
| 2 | business core — weekday evenings (peaks Thu 18:00) | 66 | 33.9M |
| 3 | nightlife — weekend small hours (peaks Sun 01:00) | 15 | 3.9M |
| 0 | residential commute (peaks Wed 07:00) | 35 | 0.6M |
| 4 | residential commute (peaks Fri 07:00) | 59 | 0.4M |
| 1 | mixed / low-signal (peaks Sat 23:00) | 50 | 0.4M |

The prediction rule is **cluster shape × the zone's own level** — the raw pooled
cluster mean is mis-specified by construction once shape is normalised, and the
evaluation shows it (WAPE 95% vs 36%).

**Supervised ablation (weather & events).** Four nested feature sets
(`time only` → `+weather` → `+events` → `+both`) × two learners (linear OLS,
gradient-boosted trees), all on the identical split and metrics. Weather and event
terms enter as **interactions with `hist_avg_demand`** — their effect is
proportional to a cell's usual volume, so a global additive coefficient would be
wrong for zones spanning three orders of magnitude.

## 5. Results

Held-out test split, N = 399,600 `(zone, hour)` cells, full-year 2024. Headline
metric is **WAPE** (the grid is ~46% zero bins, where MAPE is undefined):

| Method | MAE | RMSE | WAPE |
|---|---|---|---|
| Global mean (naive floor) | 31.51 | 60.40 | 147.7% |
| Per-zone mean | 13.46 | 38.80 | 63.1% |
| Historical avg (zone, hour, dow) | 5.44 | 18.20 | 25.5% |
| Moving avg (same hour-of-week, 4w)* | 5.14 | 17.63 | 24.1% |
| EWMA (4w, α=0.3)* | 5.28 | 18.19 | 24.7% |
| Cluster mean (raw pooled) | 20.22 | 49.53 | 94.8% |
| Cluster shape × zone level (K-Means) | 7.59 | 24.61 | 35.6% |
| **Linear + weather + events (ours)** | **5.30** | **17.48** | **24.9%** |

\* The moving-average family reads observed demand from previous weeks at inference
time — a real information advantage over the frozen models, flagged, not hidden.

**The ablation verdict** (figure: `reports/ablation_wape.png`): adding weather and
events to trips-only improves held-out WAPE by **+2.5% relative overall, +11.2% on
special days, +8.5% on rain hours** — clearing the meaningfulness convention we
stated *before* reading the results (≥2% overall or ≥5% on a target subset). GBT
agrees (+2.7% / +11.6% / +8.8%). The fitted coefficients read directly:

* each mm of rain: **+0.55%** of a cell's usual demand
* an event day: **−8.8%** — events *reduce* street demand on average (closures,
  crowds off the streets), which is why history alone overshoots Thanksgiving by ~50%
* a federal holiday: **−24.2%**

Honest caveats, printed by the pipeline itself: Open-Meteo's ~11 km grid collapses
225 zones onto 27 distinct weather series, so weather acts as a citywide *temporal*
signal, and GBT under our settings loses to OLS riding `hist_avg` — the strong base
feature does most of the work.

## 6. Streaming and live prediction

The producer replays historical Parquet into Kafka as a simulated live feed;
**event time is the pickup timestamp from the data**, never wall-clock, so any
replay speed yields identical windows. Spark Structured Streaming reapplies the
*batch* cleaning predicates (imported, not re-implemented), aggregates tumbling
1-hour windows with a **2-hour watermark** (measured: p99.9 of trip duration is
1.91 h, so ≈99.9% of genuine lateness is covered), and serves the saved K-Means
artifacts unchanged — the live path never retrains. `--validate` proves streamed
windows equal `demand.parquet` cell for cell.

`predict_live.py` answers a single query instantly (pandas, no JVM) with **two
labelled models**: the shape model (bit-identical to the stream, verified) and the
**conditions model** — the exported linear weather/events model, whose JSON
arithmetic is proven to match Spark's predictions to ~1e-13. `--temp`, `--precip`
and `--is-event` genuinely move the forecast; holiday/event flags are looked up
automatically from the calendar. Querying Midtown Center for Thanksgiving 15:00
returns a 414-trip historical base with a **−134.7 event adjustment** — no flag
supplied. Both proposal novelties are therefore delivered and demonstrable live.

## 7. Answers to the proposal's open questions

1. **How many clusters best capture demand?** K=5 on the full year (elbow), after
   inspection; the silhouette's K=2 merely separates Manhattan from everything.
   Q1's K=4 did not survive the year's added seasonality — K is data-dependent.
2. **Do weather & events meaningfully beat trips-only?** **Yes** — +2.5% relative
   WAPE overall, +11.2% on special days, +8.5% in rain (Section 5). Events carry
   most of the signal; weather is real but citywide-temporal at this resolution.
3. **Is streaming worth it, or is batch enough?** For *forecasting*, batch is
   enough — the model is a saved artifact either way. Streaming earns its place as
   the serving and monitoring layer: windowed predicted-vs-actual in near real time,
   at trivial ingest cost (a busy NYC hour is ~6,600 trips; the single-node replay
   sustains >1,000 msg/s paced). The watermark analysis is the honest cost: a
   1-hour window is only final ~2 h after it opens.
4. **A trip crossing several zones?** Assigned to its **pickup** zone — demand is
   "where a vehicle is requested", which is the quantity a dispatcher must supply.

## 8. Related work

* **Yu, Zhang & Sarwat (2019), GeoInformatica — Apache Sedona/GeoSpark.** Used
  directly: Sedona reads the zone shapefile, computes projected-CRS centroids and
  WGS84 polygons (`geo_join.py`). Since TLC pre-joins trips to zone IDs, the paper's
  distributed spatial-join machinery maps onto our centroid→weather-grid assignment
  and map rendering — stated plainly rather than overclaimed.
* **Yao et al. (2018), AAAI — DMVST-Net.** Its three views shaped our feature
  design at classical-ML scale: spatial view → zone clustering; temporal view →
  cyclical + Fourier hour-of-week encodings; semantic view → our clusters group
  zones that *behave* alike, not that neighbour each other (JFK and Midtown share a
  cluster). Its accuracy ceiling is why deep learning was scoped out, per the
  proposal's pros/cons.
* **Ferreira et al. (2013), IEEE TVCG — Visual exploration of NYC taxi data.**
  Guided the exploration and visualization layer: the zone×hour-of-week heatmap
  grouped by cluster and the per-window choropleth maps are this paper's
  query-pattern idea applied to our clusters.

## 9. Reproducibility

Pinned versions (Python 3.11 · Java 17 · Spark 3.5.3 · Scala 2.12 · Sedona 1.9.0 ·
Kafka 3.9), fixed seed 42 for K-Means and GBT, deterministic time-based split,
`kmeans_metadata.json` records K, both criterion curves, seed and the fitted data
range. One command reproduces everything: `run_pipeline_full_year.bat` (sweep,
review, commit-to-K), then `run_demo.bat` for the streaming demo. The full
walkthrough with outputs is in `notebooks/results_walkthrough.ipynb`.
