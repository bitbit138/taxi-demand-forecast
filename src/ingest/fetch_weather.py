"""Fetch hourly weather per taxi zone from the Open-Meteo archive API.

One row per ``(zone_id, ts_local)`` covering the whole ``config`` date range, cached to
``data/external/weather.parquet``. Re-runs only fetch zones that are missing or
incomplete, so interrupting the script is safe.

Two things this script is deliberately careful about:

**Projection.** The taxi-zone shapefile is EPSG:2263 (NY State Plane, *feet*). Centroids
are computed in that projected CRS — which is what it is designed for — and only then
reprojected to EPSG:4326 lat/lon for the API. Sending raw 2263 coordinates would send
Open-Meteo numbers in the millions.

**Timezone.** TLC pickup timestamps are naive *local NYC* wall-clock; Open-Meteo is asked
explicitly for ``timezone=UTC``. UTC is gapless and unambiguous, so it is what we fetch
and store; the NY local wall-clock join key is derived from it here, once, rather than
being left to every downstream consumer. Both columns are kept so the conversion stays
auditable.

DST is the reason this matters:
  * spring forward (2024-03-10) — local 02:00 never happens, so that hour is absent;
  * fall back (2024-11-03) — local 01:00 happens twice, so the second (EST) reading is
    dropped to keep ``(zone_id, ts_local)`` unique.

    python -m src.ingest.fetch_weather
    python -m src.ingest.fetch_weather --force          # refetch everything
    python -m src.ingest.fetch_weather --zones 1 2 3    # just these zones
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

WGS84 = 4326
TIMEOUT = (10, 180)
MAX_RETRIES = 4
RETRY_BACKOFF = 5.0  # seconds, multiplied by attempt number


def load_centroids() -> pd.DataFrame:
    """Zone centroids as lat/lon. Returns columns ``zone_id, lat, lon``."""
    if not config.ZONE_SHAPEFILE.exists():
        raise FileNotFoundError(
            f"{config.ZONE_SHAPEFILE} not found — run: python -m src.ingest.download_tlc"
        )

    zones = gpd.read_file(config.ZONE_SHAPEFILE)
    if zones.crs is None:
        raise ValueError("Shapefile has no CRS; expected EPSG:2263.")

    # Centroid first (in the projected CRS, where it is metrically meaningful),
    # reproject second. Reprojecting polygons first would distort the centroid.
    centroids = zones.geometry.centroid.to_crs(epsg=WGS84)

    frame = pd.DataFrame(
        {
            "zone_id": zones["LocationID"].astype("int32"),
            "lat": centroids.y.astype("float64"),
            "lon": centroids.x.astype("float64"),
        }
    ).sort_values("zone_id", ignore_index=True)

    frame = frame[~frame["zone_id"].isin(config.EXCLUDED_ZONE_IDS)]

    # NYC bounding-box sanity check — catches a projection mistake immediately.
    if not (frame["lat"].between(40.4, 41.0).all() and frame["lon"].between(-74.3, -73.6).all()):
        raise ValueError(
            "Centroids fall outside NYC — check the CRS.\n"
            f"  lat {frame['lat'].min():.4f}..{frame['lat'].max():.4f}\n"
            f"  lon {frame['lon'].min():.4f}..{frame['lon'].max():.4f}"
        )
    return frame.reset_index(drop=True)


def expected_local_hours() -> pd.DatetimeIndex:
    """Every NY-local wall-clock hour in the configured range, DST-correct.

    Built by walking UTC and converting, so the spring-forward gap and fall-back
    duplicate are handled by the tz database rather than by assuming 24 h/day.
    """
    start_utc = (
        pd.Timestamp(config.START_DATE, tz=config.LOCAL_TZ).tz_convert(config.UTC_TZ)
    )
    end_utc = (
        pd.Timestamp(f"{config.END_DATE} 23:00", tz=config.LOCAL_TZ).tz_convert(config.UTC_TZ)
    )
    utc_hours = pd.date_range(start_utc, end_utc, freq="h", tz=config.UTC_TZ)
    local = utc_hours.tz_convert(config.LOCAL_TZ).tz_localize(None)
    return pd.DatetimeIndex(pd.Series(local).drop_duplicates())  # fall-back dedupe


def _api_dates() -> tuple[str, str]:
    """UTC start/end dates padded a day either side so the local range is covered."""
    start = (pd.Timestamp(config.START_DATE) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(config.END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return start, end


def fetch_batch(batch: pd.DataFrame) -> list[dict]:
    """Fetch one multi-location request. Returns one payload per row of *batch*."""
    start, end = _api_dates()
    params = {
        "latitude": ",".join(f"{v:.6f}" for v in batch["lat"]),
        "longitude": ",".join(f"{v:.6f}" for v in batch["lon"]),
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(config.OPEN_METEO_HOURLY_VARS),
        "timezone": "UTC",  # explicit: never rely on the API default
    }

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                config.OPEN_METEO_ARCHIVE_URL, params=params, timeout=TIMEOUT
            )
            if response.status_code == 429:  # rate limited
                raise requests.HTTPError("429 rate limited")
            response.raise_for_status()
            payload = response.json()
            # A single-location request returns an object, not a list.
            locations = payload if isinstance(payload, list) else [payload]
            if len(locations) != len(batch):
                raise ValueError(
                    f"asked for {len(batch)} locations, got {len(locations)}"
                )
            return locations
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"    retry {attempt}/{MAX_RETRIES - 1} after {wait:.0f}s ({exc})")
                time.sleep(wait)
    raise RuntimeError(f"Open-Meteo failed after {MAX_RETRIES} attempts: {last_error}")


def to_frame(zone_ids: list[int], locations: list[dict]) -> pd.DataFrame:
    """Turn API payloads into tidy rows keyed by ``(zone_id, ts_local)``."""
    keep = expected_local_hours()
    frames = []

    for zone_id, location in zip(zone_ids, locations):
        hourly = location["hourly"]
        ts_utc = pd.to_datetime(hourly["time"]).tz_localize(config.UTC_TZ)
        ts_local = ts_utc.tz_convert(config.LOCAL_TZ)

        frame = pd.DataFrame(
            {
                "zone_id": zone_id,
                "ts_utc": ts_utc.tz_localize(None),
                "ts_local": ts_local.tz_localize(None),
                "temp_c": pd.to_numeric(hourly["temperature_2m"], errors="coerce"),
                "precip_mm": pd.to_numeric(hourly["precipitation"], errors="coerce"),
            }
        )
        # Trim the UTC padding to the configured local range, then drop the
        # fall-back duplicate wall-clock hour (keeps the first, i.e. EDT).
        frame = frame[frame["ts_local"].isin(keep)]
        frame = frame.drop_duplicates(subset=["zone_id", "ts_local"], keep="first")
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    # Explicit join keys so features.py is a plain lookup on (zone, date, hour).
    out["date_local"] = out["ts_local"].dt.date.astype("string")
    out["hour_local"] = out["ts_local"].dt.hour.astype("int16")
    out["zone_id"] = out["zone_id"].astype("int32")
    return out[
        ["zone_id", "ts_local", "date_local", "hour_local", "ts_utc", "temp_c", "precip_mm"]
    ]


def load_cache() -> pd.DataFrame | None:
    """Existing weather.parquet, or None."""
    if not config.WEATHER_PARQUET.exists():
        return None
    return pd.read_parquet(config.WEATHER_PARQUET)


def complete_zones(cache: pd.DataFrame | None, expected_rows: int) -> set[int]:
    """Zone ids already holding a full hourly series."""
    if cache is None or cache.empty:
        return set()
    counts = cache.groupby("zone_id").size()
    return set(counts[counts >= expected_rows].index.astype(int))


def validate(frame: pd.DataFrame, centroids: pd.DataFrame) -> bool:
    """Check zone and hour coverage. Returns True when everything lines up."""
    expected_hours = expected_local_hours()
    expected_zones = set(centroids["zone_id"].astype(int))
    ok = True

    print("\n" + "-" * 68)
    print("VALIDATION")
    print("-" * 68)

    got_zones = set(frame["zone_id"].astype(int))
    missing_zones = expected_zones - got_zones
    extra_zones = got_zones - expected_zones
    print(f"  zones            : {len(got_zones)} / {len(expected_zones)}")
    if missing_zones:
        ok = False
        print(f"    MISSING zones  : {sorted(missing_zones)[:20]}")
    if extra_zones:
        ok = False
        print(f"    UNEXPECTED     : {sorted(extra_zones)[:20]}")

    print(f"  hours per zone   : expected {len(expected_hours)}")
    print(f"    local range    : {expected_hours.min()} .. {expected_hours.max()}")

    counts = frame.groupby("zone_id").size()
    bad = counts[counts != len(expected_hours)]
    if bad.empty:
        print(f"    every zone has : {int(counts.iloc[0])} rows  (no gaps)")
    else:
        ok = False
        print(f"    WRONG COUNT in {len(bad)} zones, e.g. {bad.head().to_dict()}")

    # Any zone individually missing an expected hour.
    per_zone_missing = (
        frame.groupby("zone_id")["ts_local"]
        .apply(lambda s: len(expected_hours.difference(pd.DatetimeIndex(s))))
        .pipe(lambda s: s[s > 0])
    )
    if per_zone_missing.empty:
        print("    hour coverage  : complete for every zone")
    else:
        ok = False
        print(f"    ZONES WITH GAPS: {per_zone_missing.head().to_dict()}")

    dupes = frame.duplicated(subset=["zone_id", "ts_local"]).sum()
    print(f"  duplicate keys   : {dupes}")
    ok = ok and dupes == 0

    nulls = int(frame[["temp_c", "precip_mm"]].isna().sum().sum())
    print(f"  null temp/precip : {nulls}")
    ok = ok and nulls == 0

    print(f"  temp_c    range  : {frame['temp_c'].min():.1f} .. {frame['temp_c'].max():.1f} C")
    print(f"  precip_mm range  : {frame['precip_mm'].min():.2f} .. {frame['precip_mm'].max():.2f} mm")
    if not frame["temp_c"].between(-30, 45).all():
        ok = False
        print("    IMPLAUSIBLE temperature values")

    # DST evidence: the spring-forward hour must be absent, and present either side.
    spring = pd.Timestamp("2024-03-10 02:00")
    if expected_hours.min() <= spring <= expected_hours.max():
        n_gap = int((frame["ts_local"] == spring).sum())
        n_before = int((frame["ts_local"] == pd.Timestamp("2024-03-10 01:00")).sum())
        n_after = int((frame["ts_local"] == pd.Timestamp("2024-03-10 03:00")).sum())
        print(f"  DST 2024-03-10   : 01:00={n_before} rows, 02:00={n_gap} rows "
              f"(correctly absent), 03:00={n_after} rows")
        ok = ok and n_gap == 0 and n_before > 0 and n_after > 0

    print("-" * 68)
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="refetch all zones")
    parser.add_argument("--zones", nargs="+", type=int, help="only these zone ids")
    parser.add_argument(
        "--batch-size", type=int, default=config.OPEN_METEO_BATCH_SIZE,
        help=f"locations per request (default {config.OPEN_METEO_BATCH_SIZE})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    centroids = load_centroids()
    if args.zones:
        centroids = centroids[centroids["zone_id"].isin(args.zones)].reset_index(drop=True)
        if centroids.empty:
            print(f"ERROR: no such zone ids: {args.zones}", file=sys.stderr)
            return 2

    expected_hours = expected_local_hours()
    start, end = _api_dates()

    print("=" * 68)
    print("Open-Meteo hourly weather")
    print("=" * 68)
    print(f"  zones            : {len(centroids)}")
    print(f"  local range      : {config.START_DATE} .. {config.END_DATE} ({config.LOCAL_TZ})")
    print(f"  hours per zone   : {len(expected_hours)}")
    print(f"  API request      : {start} .. {end} in UTC (1-day pad each side)")
    print(f"  variables        : {', '.join(config.OPEN_METEO_HOURLY_VARS)}")

    cache = None if args.force else load_cache()
    done = complete_zones(cache, len(expected_hours))
    todo = centroids[~centroids["zone_id"].isin(done)].reset_index(drop=True)
    print(f"  cached complete  : {len(done)}")
    print(f"  to fetch         : {len(todo)}")

    fetched: list[pd.DataFrame] = []
    if not todo.empty:
        n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
        print(f"\nFetching in {n_batches} batch(es) of up to {args.batch_size}, "
              f"{config.OPEN_METEO_SLEEP_SECONDS}s apart")
        for i in range(n_batches):
            batch = todo.iloc[i * args.batch_size : (i + 1) * args.batch_size]
            zone_ids = batch["zone_id"].astype(int).tolist()
            print(f"  [{i + 1:>2}/{n_batches}] zones {zone_ids[0]}..{zone_ids[-1]} "
                  f"({len(zone_ids)} locations)", flush=True)
            fetched.append(to_frame(zone_ids, fetch_batch(batch)))
            if i < n_batches - 1:
                time.sleep(config.OPEN_METEO_SLEEP_SECONDS)

    parts = [df for df in ([cache] if cache is not None else []) + fetched if df is not None]
    if not parts:
        print("ERROR: nothing fetched and no cache present", file=sys.stderr)
        return 1

    weather = pd.concat(parts, ignore_index=True)
    weather = (
        weather.drop_duplicates(subset=["zone_id", "ts_local"], keep="last")
        .sort_values(["zone_id", "ts_local"], ignore_index=True)
    )
    weather.to_parquet(config.WEATHER_PARQUET, index=False)

    ok = validate(weather, load_centroids() if not args.zones else centroids)

    print("\n" + "=" * 68)
    print("DONE")
    print(f"  rows written : {len(weather):,}")
    print(f"  file         : {config.WEATHER_PARQUET}")
    print(f"  size         : {config.WEATHER_PARQUET.stat().st_size / 1024**2:.1f} MB")
    print("=" * 68)

    print("\nSample rows (zone 161 = Midtown Center, first 5 hours):")
    sample = weather[weather["zone_id"] == 161].head(5)
    print(sample.to_string(index=False) if not sample.empty
          else weather.head(5).to_string(index=False))

    print("\nWettest hours in the range:")
    print(weather.nlargest(3, "precip_mm").to_string(index=False))

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
