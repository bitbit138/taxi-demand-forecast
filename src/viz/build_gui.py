"""Export the trained model into ``gui/`` so the browser console can serve it.

The GUI is a **static local web app**: no Python, no JVM and no network at
presentation time. This script is what makes that possible — it reads the saved
artifacts and writes a single JSON payload the page loads:

  * ``models/conditions_model.json``  intercept + 21 OLS coefficients + normals
  * ``models/hist_avg.parquet``       225 zones x 168 hour-of-week means
  * ``models/zone_clusters.parquet``  zone -> cluster + the zone's own level
  * ``models/cluster_demand.parquet`` cluster -> 168 demand shares
  * ``data/processed/zone_centroids.parquet``  polygons, simplified to SVG paths
  * ``data/external/events.csv``      holiday/event flags for the calendar lookup
  * ``reports/*.csv``                 the evidence tables shown in the lower deck

**The arithmetic is not re-derived.** ``gui/model.js`` applies exactly the
coefficients exported here, the same way ``src/stream/predict_live.py`` applies
them in pandas. To prove the payload is faithful, this script re-applies the
model *from the JSON it just wrote* and compares against
``Forecaster.predict_conditions`` over random queries — a wrong export fails the
build instead of shipping a plausible-looking wrong number.

Two builds are written, because a demo should not depend on a working server:

  ``gui/payload.json``     loaded by ``gui/index.html`` when served over http
  ``gui/standalone.html``  one file with the payload inlined; opens from disk

    python -m src.viz.build_gui
    python -m src.viz.build_gui --checks 1000    # more parity queries
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import pandas as pd
from shapely import wkt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402
from src.stream.predict_live import Forecaster, describe_shape  # noqa: E402

GUI_DIR = config.ROOT / "gui"
PAYLOAD_JSON = GUI_DIR / "payload.json"
STANDALONE_HTML = GUI_DIR / "standalone.html"

# Polygon simplification: ~35 m at NYC latitude. Small enough that zone shapes stay
# recognisable, coarse enough to keep all 225 zones under a few hundred KB.
SIMPLIFY_DEG = 0.00035
NYC_LAT = 40.72
SVG_WIDTH = 1000.0
FOURIER = config.FOURIER_TERMS
HOURS = config.HOURS_PER_WEEK


def zone_table(forecaster: Forecaster) -> list[dict]:
    """One row per served zone, ordered borough then name for the picker."""
    frame = forecaster.zones.sort_values(["borough", "zone_name"])
    return [
        {
            "id": int(r.zone_id),
            "n": r.zone_name,
            "b": r.borough,
            "c": int(r.cluster),
            "lvl": round(float(r.zone_mean_demand), 6),
            "tr": int(r.total_trips),
        }
        for r in frame.itertuples()
    ]


def hist_by_zone() -> dict[str, list[float]]:
    """``zone_id -> [168 means]`` indexed by hour-of-week (Mon 00:00 = 0)."""
    frame = pd.read_parquet(config.HIST_AVG_PARQUET)
    frame["how"] = frame["dow"] * 24 + frame["hour_local"]
    out: dict[str, list[float]] = {}
    for zone_id, group in frame.groupby("zone_id"):
        row = [0.0] * HOURS
        for how, value in zip(group["how"], group["hist_avg_demand"]):
            row[int(how)] = round(float(value), 6)
        out[str(int(zone_id))] = row
    return out


def cluster_shapes(forecaster: Forecaster) -> tuple[dict, dict]:
    """``cluster -> 168 shares`` (full precision) and the derived characters."""
    shares, characters = {}, {}
    for cluster, group in forecaster.shape.groupby("cluster"):
        series = group.set_index("hour_of_week")["cluster_share"].sort_index()
        key = str(int(cluster))
        shares[key] = [float(v) for v in series.values]
        members = forecaster.zones[forecaster.zones["cluster"] == cluster]
        described = describe_shape(series, len(members))
        characters[key] = {
            "label": described["label"],
            "peak": described["peak_label"],
            "n": int(len(members)),
            "trips": int(members["total_trips"].sum()),
        }
    return shares, characters


def event_flags() -> dict[str, dict]:
    """Only the days carrying a flag; every other date defaults to ordinary."""
    if not Path(config.EVENTS_CSV).exists():
        return {}
    frame = pd.read_csv(config.EVENTS_CSV)
    flagged = frame[(frame["is_holiday"]) | (frame["is_event"])]
    return {
        r.date_local: {
            "h": int(bool(r.is_holiday)),
            "f": int(bool(r.is_federal_holiday)),
            "e": int(bool(r.is_event)),
            "hn": "" if pd.isna(r.holiday_name) else str(r.holiday_name),
            "en": "" if pd.isna(r.event_name) else str(r.event_name),
        }
        for r in flagged.itertuples()
    }


def svg_paths(zone_ids: set[int]) -> tuple[dict[str, str], dict[str, float]]:
    """Zone polygons -> SVG path strings in a 1000-unit-wide viewBox.

    Equirectangular, scaled by cos(latitude) so the city is not horizontally
    stretched. Projection happens here rather than in the browser: the page then
    ships plain path data and needs no geo code at all.
    """
    centroids = pd.read_parquet(config.ZONE_CENTROIDS_PARQUET)
    geoms = {
        int(r.zone_id): wkt.loads(r.geometry_wkt)
        for r in centroids.itertuples()
        if int(r.zone_id) in zone_ids
    }
    if not geoms:
        raise ValueError("No geometry for the served zones — re-run src.batch.geo_join")

    scale = math.cos(math.radians(NYC_LAT))
    xs: list[float] = []
    ys: list[float] = []
    for geom in geoms.values():
        minx, miny, maxx, maxy = geom.bounds
        xs += [minx * scale, maxx * scale]
        ys += [miny, maxy]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    height = SVG_WIDTH * (y1 - y0) / (x1 - x0)

    def project(lon: float, lat: float) -> tuple[float, float]:
        return (
            (lon * scale - x0) / (x1 - x0) * SVG_WIDTH,
            (y1 - lat) / (y1 - y0) * height,
        )

    paths: dict[str, str] = {}
    for zone_id, geom in geoms.items():
        simplified = geom.simplify(SIMPLIFY_DEG, preserve_topology=True)
        parts = list(simplified.geoms) if simplified.geom_type == "MultiPolygon" else [simplified]
        pieces = []
        for part in parts:
            if part.is_empty:
                continue
            points = [project(x, y) for x, y in part.exterior.coords]
            pieces.append("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in points) + "Z")
        if pieces:
            paths[str(zone_id)] = "".join(pieces)

    return paths, {"w": round(SVG_WIDTH, 1), "h": round(height, 1)}


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evidence() -> dict:
    """The report tables the lower deck shows, read from reports/."""
    out: dict = {"ladder": [], "sig": [], "bench": [], "rules": []}

    if config.METRICS_CSV.exists():
        out["ladder"] = [
            {"m": r["method"], "mae": float(r["mae"]), "rmse": float(r["rmse"]),
             "wape": float(r["wape"])}
            for r in _rows(config.METRICS_CSV) if r["subset"] == "all"
        ]
    sig_csv = config.REPORTS_DIR / "significance.csv"
    if sig_csv.exists():
        out["sig"] = [
            {"s": r["subset"], "d": float(r["n_days"]), "hist": float(r["wape_hist_pct"]),
             "full": float(r["wape_full_pct"]), "gain": float(r["delta_rel_pct"]),
             "lo": float(r["ci95_lo"]), "hi": float(r["ci95_hi"]),
             "p": float(r["p_one_sided"])}
            for r in _rows(sig_csv)
        ]
    bench_csv = config.REPORTS_DIR / "scale_benchmark.csv"
    if bench_csv.exists():
        out["bench"] = [
            {"mo": int(r["months"]), "rin": int(r["rows_in"]), "sec": float(r["seconds"]),
             "rps": int(r["rows_per_second"])}
            for r in _rows(bench_csv)
        ]
    rules_csv = config.REPORTS_DIR / "association_rules.csv"
    if rules_csv.exists():
        top = sorted(_rows(rules_csv), key=lambda r: -float(r["lift"]))[:8]
        out["rules"] = [
            {"pu": r["pu_name"], "do": r["do_name"], "lift": float(r["lift"]),
             "conf": float(r["confidence"])}
            for r in top
        ]
    return out


def apply_from_payload(payload: dict, zone_id: int, date: str, dow: int, hour: int,
                       temp_c: float | None, precip_mm: float | None,
                       force_event: bool) -> float:
    """Re-apply the model using ONLY what the payload contains.

    This is the parity check: it deliberately ignores the pandas objects and
    reads the same numbers ``gui/model.js`` will read, so a lossy export is
    caught here rather than in front of an audience.
    """
    model = payload["model"]
    coef = model["coefficients"]
    how = dow * 24 + hour
    hist = payload["hist"][str(zone_id)][how]

    if temp_c is None:
        temp_c = float(model["monthly_temp_normals_c"][str(int(date[5:7]))])
    if precip_mm is None:
        precip_mm = float(model["default_precip_mm"])

    flags = payload["events"].get(date)
    holiday = bool(flags["h"]) if flags else False
    fedhol = bool(flags["f"]) if flags else False
    event = bool(flags["e"]) if flags else False
    if force_event:
        event = True

    two_pi = 2.0 * math.pi
    dev = temp_c - float(model["train_mean_temp_c"])
    features = {
        "hist_avg_demand": hist,
        "hour_sin": math.sin(two_pi * hour / 24.0),
        "hour_cos": math.cos(two_pi * hour / 24.0),
        "dow_sin": math.sin(two_pi * dow / 7.0),
        "dow_cos": math.cos(two_pi * dow / 7.0),
        "weekend_d": 1.0 if dow in (5, 6) else 0.0,
        "temp_c": temp_c,
        "precip_mm": precip_mm,
        "temp_dev_x_hist": dev * hist,
        "precip_x_hist": precip_mm * hist,
        "holiday_d": 1.0 if holiday else 0.0,
        "fedhol_d": 1.0 if fedhol else 0.0,
        "event_d": 1.0 if event else 0.0,
        "fedhol_x_hist": (1.0 if fedhol else 0.0) * hist,
        "event_x_hist": (1.0 if event else 0.0) * hist,
    }
    for k in range(1, FOURIER + 1):
        angle = two_pi * k * how / float(HOURS)
        features[f"how_sin_{k}"] = math.sin(angle)
        features[f"how_cos_{k}"] = math.cos(angle)

    raw = float(model["intercept"]) + sum(coef[n] * features[n] for n in coef)
    return max(0.0, raw)


def verify(payload: dict, forecaster: Forecaster, n_checks: int) -> tuple[bool, float, float]:
    """Payload arithmetic vs the pandas implementation, over random queries."""
    rng = random.Random(config.SEED)
    zone_ids = [int(z) for z in forecaster.zones["zone_id"]]
    dates = pd.date_range(config.START_DATE, config.END_DATE, freq="D")
    worst_cond = worst_shape = 0.0

    for _ in range(n_checks):
        zone_id = rng.choice(zone_ids)
        date = rng.choice(dates).strftime("%Y-%m-%d")
        hour = rng.randrange(24)
        precip = rng.choice([None, 0.0, 0.6, 3.0, 11.0, 25.0])
        temp = rng.choice([None, -9.0, 5.5, 22.0, 34.0])
        force = rng.random() < 0.25
        when = pd.Timestamp(f"{date} {hour:02d}:00")
        dow = when.weekday()

        expected = forecaster.predict_conditions(
            zone_id, when, temp_c=temp, precip_mm=precip, force_event=force
        )["predicted_demand"]
        got = apply_from_payload(payload, zone_id, date, dow, hour, temp, precip, force)
        worst_cond = max(worst_cond, abs(got - expected))

        shape_expected = forecaster.predict(zone_id, when)["predicted_demand"]
        zone = next(z for z in payload["zones"] if z["id"] == zone_id)
        share = payload["share"][str(zone["c"])][dow * 24 + hour]
        worst_shape = max(worst_shape, abs(zone["lvl"] * share * HOURS - shape_expected))

    # The payload rounds hist_avg and the zone level to 6 decimals, so exact
    # equality is not the bar; a wrong formula or a mis-keyed table misses by
    # whole trips, which this tolerance still catches decisively.
    tolerance = 1e-3
    return (worst_cond < tolerance and worst_shape < tolerance), worst_cond, worst_shape


def write_standalone(payload_text: str) -> None:
    """One self-contained file, for when there is no server (or no confidence)."""
    index = (GUI_DIR / "index.html").read_text(encoding="utf-8")
    css = (GUI_DIR / "styles.css").read_text(encoding="utf-8")
    model_js = (GUI_DIR / "model.js").read_text(encoding="utf-8")
    app_js = (GUI_DIR / "app.js").read_text(encoding="utf-8")

    html = index.replace(
        '<link rel="stylesheet" href="styles.css">',
        "<style>\n" + css + "\n</style>",
    ).replace(
        '<script src="model.js"></script>',
        '<script id="payload" type="application/json">' + payload_text + "</script>\n"
        "<script>\n" + model_js + "\n</script>",
    ).replace(
        '<script src="app.js"></script>',
        "<script>\n" + app_js + "\n</script>",
    )
    if "<style>" not in html or "id=\"payload\"" not in html:
        raise ValueError("index.html no longer matches the standalone template")
    STANDALONE_HTML.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checks", type=int, default=400,
                        help="parity queries against predict_live (default 400)")
    args = parser.parse_args()

    GUI_DIR.mkdir(parents=True, exist_ok=True)
    for path, hint in (
        (config.ZONE_CLUSTERS_PARQUET, "src.batch.train_kmeans"),
        (config.CLUSTER_PROFILE_PARQUET, "src.batch.train_kmeans"),
        (config.MODELING_ZONES_PARQUET, "src.batch.zone_policy"),
        (config.CONDITIONS_MODEL_JSON, "src.batch.ablation"),
        (config.HIST_AVG_PARQUET, "src.batch.ablation"),
        (config.ZONE_CENTROIDS_PARQUET, "src.batch.geo_join"),
    ):
        if not Path(path).exists():
            print(f"ERROR: {path} missing — run: python -m {hint}", file=sys.stderr)
            return 2

    print("=" * 78)
    print("Build the browser console payload")
    print("=" * 78)

    forecaster = Forecaster()
    zones = zone_table(forecaster)
    shares, characters = cluster_shapes(forecaster)
    paths, view = svg_paths({z["id"] for z in zones})
    model = json.loads(config.CONDITIONS_MODEL_JSON.read_text())

    payload = {
        "meta": {
            "k": int(forecaster.metadata.get("chosen_k", 0)),
            "elbow": forecaster.metadata.get("k_suggested_by_elbow"),
            "sil": forecaster.metadata.get("k_suggested_by_silhouette"),
            "months": len((forecaster.metadata.get("data_range") or {}).get("months", [])),
            "zones": len(zones),
            "fitted_on": model.get("fitted_on", {}),
        },
        "view": view,
        "zones": zones,
        "hist": hist_by_zone(),
        "share": shares,
        "chars": characters,
        "model": {
            k: model[k] for k in (
                "intercept", "coefficients", "train_mean_temp_c",
                "monthly_temp_normals_c", "default_precip_mm",
            )
        },
        "events": event_flags(),
        "paths": paths,
        **evidence(),
    }

    print(f"  zones served       : {len(zones)}")
    print(f"  hour-of-week means : {len(payload['hist']):,} zones x {HOURS}")
    print(f"  clusters           : {len(shares)} (K={payload['meta']['k']})")
    print(f"  zone polygons      : {len(paths)}  viewBox {view['w']} x {view['h']}")
    print(f"  flagged dates      : {len(payload['events'])}")
    print(f"  evidence tables    : ladder {len(payload['ladder'])} rows, "
          f"significance {len(payload['sig'])}, benchmark {len(payload['bench'])}, "
          f"rules {len(payload['rules'])}")

    print("\nVALIDATION — payload arithmetic vs src/stream/predict_live.py")
    ok, worst_cond, worst_shape = verify(payload, forecaster, args.checks)
    print(f"  queries checked            : {args.checks:,}")
    print(f"  max |payload - pandas|     : conditions {worst_cond:.2e}  "
          f"shape {worst_shape:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL — do not ship this payload'}")
    if not ok:
        return 1

    text = json.dumps(payload, separators=(",", ":"))
    PAYLOAD_JSON.write_text(text, encoding="utf-8")
    write_standalone(text)

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  payload    : {PAYLOAD_JSON}  ({PAYLOAD_JSON.stat().st_size / 1024:.0f} KB)")
    print(f"  standalone : {STANDALONE_HTML}  "
          f"({STANDALONE_HTML.stat().st_size / 1024:.0f} KB)")
    print("  serve it   : run_gui.bat        (http://localhost:8765)")
    print("  or just    : open gui\\standalone.html in a browser")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
