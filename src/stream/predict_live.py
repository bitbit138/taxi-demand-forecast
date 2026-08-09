"""Single-query live forecaster: ``(zone_id, timestamp, conditions) -> demand``.

Serves **two models side by side**, and says which is which:

* **Shape model** — cluster shape x zone level, the exact arithmetic
  ``spark_stream.py`` serves, loaded from the same ``zone_clusters.parquet`` /
  ``cluster_demand.parquet``. A single query returns exactly the number the stream
  and the batch produce for the same ``(zone, hour-of-week)``. This is the
  interpretable model and the one validated against the streaming output.

* **Conditions model** — the linear weather/events model exported by
  ``src/batch/ablation.py`` (``models/conditions_model.json`` +
  ``models/hist_avg.parquet``). It actually **uses** ``--temp``, ``--precip`` and
  ``--is-event``: rain, temperature and event/holiday flags shift the forecast by
  fitted, test-set-verified coefficients. Defaults when a condition is not given
  are stated in the output: monthly-normal temperature, no rain, and event/holiday
  flags looked up from ``events.csv`` for the query date.

Pandas rather than Spark: the artifacts are small and a demo query should answer
instantly, not spend 20 seconds starting a JVM. The ablation run proves the JSON
arithmetic matches Spark's own predictions to ~1e-13 before exporting.

If the conditions model has not been trained yet, the tool degrades to the shape
model alone and says the condition inputs were **ignored** — never silently.

    python -m src.stream.predict_live --zone 79  --at "2024-01-13 01:00"
    python -m src.stream.predict_live --zone 161 --at "2024-11-28 15:00" --precip 4
    python -m src.stream.predict_live --zone 161 --at "2024-01-10 15:00" --json
    python -m src.stream.predict_live --validate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CONDITION_ARGS = ("temp", "precip", "is_event")


def how_label(hour_of_week: int) -> str:
    """168-hour index -> e.g. 'Sat 01:00'."""
    return f"{DAY_NAMES[hour_of_week // 24]} {hour_of_week % 24:02d}:00"


def hour_of_week(when: pd.Timestamp) -> int:
    """Mon 00:00 = 0 .. Sun 23:00 = 167.

    ``Timestamp.weekday()`` is Mon=0..Sun=6, which is the same convention as the
    ``(dayofweek + 5) % 7`` remap used in features.py and spark_stream.py.
    """
    return when.weekday() * 24 + when.hour


def describe_shape(shares: pd.Series, n_zones: int) -> dict:
    """Label a cluster from its 168-hour share profile.

    Pure function of the shape, so the label stays truthful if K or the data
    changes. Shared with ``train_kmeans.py`` — a full-year re-sweep must describe
    its candidate clusters the same way the saved model does.

    Args:
        shares: 168 values indexed by hour-of-week, summing to ~1.0.
        n_zones: members of the cluster.
    """
    peak = int(shares.idxmax())
    weekend_night = float(
        shares[[d * 24 + h for d in (4, 5) for h in (22, 23)]
               + [d * 24 + h for d in (5, 6) for h in (0, 1, 2)]].sum()
    )
    weekday_morning = float(
        shares[[d * 24 + h for d in range(5) for h in (6, 7, 8, 9)]].sum()
    )
    weekday_evening = float(
        shares[[d * 24 + h for d in range(5) for h in (16, 17, 18, 19)]].sum()
    )

    if n_zones <= 2:
        # A one- or two-zone cluster is an artifact of zones sitting just above the
        # exclusion floor: their profiles are mostly noise, so naming a "character"
        # would give the shape more credit than it has.
        label = "single-zone artifact — sparse, mostly noise"
    elif weekend_night > 0.15:
        label = "nightlife — weekend small hours"
    elif weekday_morning > 0.20:
        label = "residential commute — weekday mornings"
    elif weekday_evening > weekday_morning:
        label = "business core — weekday evenings"
    else:
        label = "mixed / low-signal"

    return {
        "label": label,
        "peak_how": peak,
        "peak_label": how_label(peak),
        "peak_share": float(shares.max()),
        "weekend_night_share": weekend_night,
        "weekday_morning_share": weekday_morning,
        "weekday_evening_share": weekday_evening,
        "n_zones": n_zones,
    }


class Forecaster:
    """The saved model, loaded once."""

    def __init__(self) -> None:
        missing = [
            path
            for path in (config.ZONE_CLUSTERS_PARQUET, config.CLUSTER_PROFILE_PARQUET,
                         config.MODELING_ZONES_PARQUET)
            if not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Model artifacts missing — run: python -m src.batch.train_kmeans\n  "
                + "\n  ".join(str(p) for p in missing)
            )

        self.zones = pd.read_parquet(config.ZONE_CLUSTERS_PARQUET)
        self.shape = pd.read_parquet(config.CLUSTER_PROFILE_PARQUET)
        names = pd.read_parquet(config.MODELING_ZONES_PARQUET)[
            ["zone_id", "zone_name", "borough", "total_trips"]
        ]
        self.zones = self.zones.merge(names, on="zone_id", how="left")
        self.metadata = (
            json.loads(config.KMEANS_METADATA_JSON.read_text())
            if config.KMEANS_METADATA_JSON.exists()
            else {}
        )
        self._characters = self._describe_clusters()

        # Conditions model (optional): trained by src/batch/ablation.py. Absent ->
        # the tool degrades to the shape model and says so.
        self.conditions: dict | None = None
        self.hist_avg: pd.Series | None = None
        if config.CONDITIONS_MODEL_JSON.exists() and config.HIST_AVG_PARQUET.exists():
            self.conditions = json.loads(config.CONDITIONS_MODEL_JSON.read_text())
            self.hist_avg = pd.read_parquet(config.HIST_AVG_PARQUET).set_index(
                ["zone_id", "hour_local", "dow"]
            )["hist_avg_demand"]

        # Event calendar for default flags — covers all of 2024 by construction.
        self.events: pd.DataFrame | None = None
        if Path(config.EVENTS_CSV).exists():
            self.events = pd.read_csv(config.EVENTS_CSV).set_index("date_local")

    def _describe_clusters(self) -> dict[int, dict]:
        """Label each cluster from its own shape — never hard-coded.

        Derived at load time so the labels stay truthful if K or the data changes.
        """
        characters: dict[int, dict] = {}
        for cluster, group in self.shape.groupby("cluster"):
            shares = group.set_index("hour_of_week")["cluster_share"]
            n_zones = int((self.zones["cluster"] == cluster).sum())
            characters[int(cluster)] = describe_shape(shares, n_zones)
        return characters

    @property
    def characters(self) -> dict[int, dict]:
        """Derived cluster characters, keyed by cluster id."""
        return self._characters

    def predict(self, zone_id: int, when: pd.Timestamp) -> dict:
        """Predicted demand plus the reasoning behind it."""
        row = self.zones[self.zones["zone_id"] == zone_id]
        if row.empty:
            raise KeyError(zone_id)
        row = row.iloc[0]

        how = hour_of_week(when)
        cluster = int(row["cluster"])
        cell = self.shape[
            (self.shape["cluster"] == cluster) & (self.shape["hour_of_week"] == how)
        ]
        if cell.empty:
            raise LookupError(f"no shape for cluster {cluster} at hour-of-week {how}")

        share = float(cell.iloc[0]["cluster_share"])
        level = float(row["zone_mean_demand"])
        # Identical arithmetic to spark_stream.add_prediction().
        predicted = level * share * config.HOURS_PER_WEEK

        character = self._characters[cluster]
        return {
            "zone_id": zone_id,
            "zone_name": row["zone_name"],
            "borough": row["borough"],
            "timestamp": when.strftime("%Y-%m-%d %H:%M"),
            "hour_of_week": how,
            "hour_of_week_label": how_label(how),
            "cluster": cluster,
            "cluster_character": character["label"],
            "cluster_n_zones": character["n_zones"],
            "cluster_peak": character["peak_label"],
            "zone_mean_demand": level,
            "cluster_share": share,
            "share_vs_flat": share * config.HOURS_PER_WEEK,
            "predicted_demand": predicted,
            "predicted_demand_rounded": round(predicted, 2),
        }

    def predict_conditions(
        self,
        zone_id: int,
        when: pd.Timestamp,
        temp_c: float | None = None,
        precip_mm: float | None = None,
        force_event: bool = False,
    ) -> dict | None:
        """Weather/events-aware forecast from the exported linear model.

        Applies exactly the coefficients ``ablation.py`` fitted and verified
        against Spark. Returns None when the conditions model is not trained.
        Missing inputs fall back to stated defaults: monthly-normal temperature,
        zero precipitation, and event/holiday flags from ``events.csv`` for the
        query date (``force_event=True`` overrides the calendar).
        """
        if self.conditions is None or self.hist_avg is None:
            return None

        dow = when.weekday()
        hour = when.hour
        how = hour_of_week(when)
        try:
            hist = float(self.hist_avg.loc[(zone_id, hour, dow)])
        except KeyError:
            raise LookupError(
                f"no hist_avg for zone {zone_id}, hour {hour}, dow {dow}"
            ) from None

        # Defaults, each with its provenance recorded for the output.
        assumed: dict[str, str] = {}
        if temp_c is None:
            normals = self.conditions["monthly_temp_normals_c"]
            temp_c = float(normals[str(when.month)])
            assumed["temp_c"] = f"{temp_c:.1f} (monthly normal for {when.month:02d})"
        if precip_mm is None:
            precip_mm = float(self.conditions["default_precip_mm"])
            assumed["precip_mm"] = f"{precip_mm:g} (no rain assumed)"

        date_key = when.strftime("%Y-%m-%d")
        holiday = fedhol = event = False
        event_name = holiday_name = ""
        if self.events is not None and date_key in self.events.index:
            row = self.events.loc[date_key]
            holiday = bool(row["is_holiday"])
            fedhol = bool(row["is_federal_holiday"])
            event = bool(row["is_event"])
            holiday_name = "" if pd.isna(row.get("holiday_name")) else str(row["holiday_name"])
            event_name = "" if pd.isna(row.get("event_name")) else str(row["event_name"])
            assumed["flags"] = f"from events.csv for {date_key}"
        else:
            assumed["flags"] = f"{date_key} not in events.csv — all flags False"
        if force_event:
            event = True
            assumed["flags"] += " (event forced by --is-event)"

        two_pi = 2.0 * math.pi
        temp_dev = temp_c - float(self.conditions["train_mean_temp_c"])
        features = {
            "hist_avg_demand": hist,
            "hour_sin": math.sin(two_pi * hour / 24.0),
            "hour_cos": math.cos(two_pi * hour / 24.0),
            "dow_sin": math.sin(two_pi * dow / 7.0),
            "dow_cos": math.cos(two_pi * dow / 7.0),
            "weekend_d": 1.0 if dow in (5, 6) else 0.0,
            "temp_c": temp_c,
            "precip_mm": precip_mm,
            "temp_dev_x_hist": temp_dev * hist,
            "precip_x_hist": precip_mm * hist,
            "holiday_d": 1.0 if holiday else 0.0,
            "fedhol_d": 1.0 if fedhol else 0.0,
            "event_d": 1.0 if event else 0.0,
            "fedhol_x_hist": (1.0 if fedhol else 0.0) * hist,
            "event_x_hist": (1.0 if event else 0.0) * hist,
        }
        for k in range(1, config.FOURIER_TERMS + 1):
            angle = two_pi * k * how / float(config.HOURS_PER_WEEK)
            features[f"how_sin_{k}"] = math.sin(angle)
            features[f"how_cos_{k}"] = math.cos(angle)

        coef = self.conditions["coefficients"]
        contribution = {name: coef[name] * features[name] for name in coef}
        raw = float(self.conditions["intercept"]) + sum(contribution.values())
        predicted = max(0.0, raw)

        weather_terms = ("temp_c", "precip_mm", "temp_dev_x_hist", "precip_x_hist")
        event_terms = ("holiday_d", "fedhol_d", "event_d", "fedhol_x_hist",
                       "event_x_hist")
        weather_delta = sum(contribution[t] for t in weather_terms if t in contribution)
        event_delta = sum(contribution[t] for t in event_terms if t in contribution)

        return {
            "predicted_demand": predicted,
            "predicted_demand_rounded": round(predicted, 2),
            "clamped": raw < 0.0,
            "hist_avg_base": hist,
            "weather_delta": weather_delta,
            "event_delta": event_delta,
            "calendar_delta": raw - hist - weather_delta - event_delta,
            "inputs": {
                "temp_c": temp_c, "precip_mm": precip_mm,
                "is_holiday": holiday, "is_federal_holiday": fedhol,
                "is_event": event,
                "holiday_name": holiday_name, "event_name": event_name,
            },
            "assumed": assumed,
            "model": self.conditions["learner"],
            "feature_set": self.conditions["feature_set"],
        }

    def is_served(self, zone_id: int) -> bool:
        return bool((self.zones["zone_id"] == zone_id).any())


def reject(forecaster: Forecaster, zone_id: int) -> str:
    """Explain precisely why a zone is not served."""
    lookup = pd.read_csv(config.ZONE_LOOKUP_CSV)
    match = lookup[lookup["LocationID"] == zone_id]

    lines = [f"Zone {zone_id} is NOT in the modelling set — no prediction returned."]
    if match.empty:
        lines.append(f"  No such taxi zone. Valid ids are "
                     f"{config.VALID_ZONE_MIN}..{config.VALID_ZONE_MAX}.")
    else:
        name = match.iloc[0]["Zone"]
        borough = match.iloc[0]["Borough"]
        lines.append(f"  {name} ({borough})")
        if zone_id in config.EXCLUDED_ZONE_IDS:
            lines.append("  Dropped upstream as an Unknown/N-A zone.")
        else:
            lines.append(
                f"  Below the {config.MIN_ZONE_TRIPS_PER_DAY} trips/day exclusion floor "
                "(see src/batch/zone_policy.py) — its demand profile is almost all "
                "zeros, so no reliable shape could be learned."
            )
    lines.append(f"  {len(forecaster.zones)} zones are served.")
    return "\n".join(lines)


def render(result: dict, ignored: dict) -> str:
    """Human-readable answer with its provenance."""
    flat = result["zone_mean_demand"]
    multiplier = result["share_vs_flat"]
    conditions = result.get("conditions")

    headline = result["predicted_demand_rounded"]
    if conditions:
        headline = conditions["predicted_demand_rounded"]
    lines = [
        "=" * 72,
        f"PREDICTED DEMAND  {headline:.2f} trips",
        "=" * 72,
        f"  query            zone {result['zone_id']} "
        f"({result['zone_name']}, {result['borough']})",
        f"                   {result['timestamp']}  "
        f"= {result['hour_of_week_label']} (hour-of-week "
        f"{result['hour_of_week']})",
        "",
        f"  SHAPE MODEL      {result['predicted_demand_rounded']:.2f} trips "
        "(what spark_stream.py serves)",
        f"    cluster        {result['cluster']} — {result['cluster_character']}",
        f"                   {result['cluster_n_zones']} zones, "
        f"peaks {result['cluster_peak']}",
        f"    zone level     {flat:.3f} trips/hour averaged over the training split",
        f"    cluster share  {100 * result['cluster_share']:.4f}% of the week's demand "
        "falls in this hour",
        f"    combined       {flat:.3f} x {result['cluster_share']:.6f} x 168 "
        f"= {result['predicted_demand']:.4f}",
        f"    i.e. {multiplier:.2f}x this zone's flat hourly average",
    ]

    if conditions:
        inputs = conditions["inputs"]
        flags = []
        if inputs["is_federal_holiday"]:
            flags.append(f"federal holiday ({inputs['holiday_name']})")
        elif inputs["is_holiday"]:
            flags.append(f"holiday ({inputs['holiday_name']})")
        if inputs["is_event"]:
            flags.append(f"event ({inputs['event_name'] or 'forced'})")
        lines += [
            "",
            f"  CONDITIONS MODEL {conditions['predicted_demand_rounded']:.2f} trips "
            "(weather/events-aware — the headline)",
            f"    base           {conditions['hist_avg_base']:.2f} historical average "
            "for this (zone, hour, weekday)",
            f"    calendar       {conditions['calendar_delta']:+.2f}",
            f"    weather        {conditions['weather_delta']:+.2f}   "
            f"(temp {inputs['temp_c']:.1f} C, precip {inputs['precip_mm']:g} mm)",
            f"    events         {conditions['event_delta']:+.2f}   "
            f"({', '.join(flags) if flags else 'no holiday, no event'})",
        ]
        if conditions["clamped"]:
            lines.append("    clamped        raw prediction was negative -> 0")
        if conditions["assumed"]:
            lines.append("    defaults used  "
                         + "; ".join(f"{k}: {v}" for k, v in conditions["assumed"].items()))
        lines.append(f"    fitted by      src/batch/ablation.py "
                     f"({conditions['feature_set']})")
    elif ignored:
        lines += [
            "",
            "  IGNORED INPUTS",
            f"    {', '.join(f'{k}={v}' for k, v in ignored.items())}",
            "    The conditions model is not trained (run python -m src.batch.ablation),",
            "    so only the K-Means shape model is available — a function of",
            "    (zone, hour-of-week) only. The values above changed nothing.",
        ]
    lines.append("=" * 72)
    return "\n".join(lines)


def validate(forecaster: Forecaster) -> bool:
    """Prove the single-query answer equals the stream's, and demo the shape."""
    ok = True
    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)

    # --- 1. exact agreement with the streaming output -------------------------
    print("\n1. Agreement with spark_stream.py output")
    if not Path(config.STREAM_OUTPUT_DIR).exists():
        print("   SKIPPED — no stream output. Run:")
        print("     python -m src.stream.producer --start '2024-01-10 14:00:00' "
              "--hours 6 --speedup 86400 --reset-topic")
        print("     python -m src.stream.spark_stream --run-seconds 120 --fresh")
    else:
        streamed = pd.read_parquet(config.STREAM_OUTPUT_DIR)
        sample = streamed.nlargest(3, "actual_demand")
        print(f"   {'zone':>5} {'window':<17} {'stream':>9} {'predict_live':>13} {'match':>7}")
        for _, row in sample.iterrows():
            when = pd.Timestamp(row["window_start"])
            got = forecaster.predict(int(row["zone_id"]), when)
            same = abs(got["predicted_demand_rounded"] - float(row["predicted_demand"])) < 1e-9
            ok = ok and same
            print(f"   {int(row['zone_id']):>5} {when.strftime('%Y-%m-%d %H:%M'):<17} "
                  f"{float(row['predicted_demand']):>9.2f} "
                  f"{got['predicted_demand_rounded']:>13.2f} {'OK' if same else 'DIFFER':>7}")

    # --- 2. the learned shape is visible --------------------------------------
    print("\n2. Nightlife cluster reflects the learned shape (zone 79, East Village)")
    saturday_night = pd.Timestamp("2024-01-13 01:00")   # Saturday 01:00
    tuesday_morning = pd.Timestamp("2024-01-09 07:00")  # Tuesday 07:00
    night = forecaster.predict(79, saturday_night)
    morning = forecaster.predict(79, tuesday_morning)
    print(f"   cluster {night['cluster']} — {night['cluster_character']}")
    print(f"   {night['hour_of_week_label']:<9} -> {night['predicted_demand']:>7.2f} "
          f"trips  ({night['share_vs_flat']:.2f}x the zone's flat average)")
    print(f"   {morning['hour_of_week_label']:<9} -> {morning['predicted_demand']:>7.2f} "
          f"trips  ({morning['share_vs_flat']:.2f}x)")
    ratio = night["predicted_demand"] / max(morning["predicted_demand"], 1e-9)
    print(f"   Saturday 01:00 is {ratio:.1f}x Tuesday 07:00 for the same zone —")
    print("   the level is identical, so the whole difference is the cluster's shape.")
    shape_ok = ratio > 3.0
    ok = ok and shape_ok
    if not shape_ok:
        print("   UNEXPECTED — a nightlife zone should peak hard in weekend small hours")

    # Contrast with a commuter zone at the same two instants.
    commuter = forecaster.zones[
        forecaster.zones["cluster"] != night["cluster"]
    ].nlargest(1, "total_trips").iloc[0]
    c_night = forecaster.predict(int(commuter["zone_id"]), saturday_night)
    c_morning = forecaster.predict(int(commuter["zone_id"]), tuesday_morning)
    print(f"\n   contrast — zone {c_night['zone_id']} ({c_night['zone_name']}), "
          f"cluster {c_night['cluster']}:")
    print(f"     Sat 01:00 {c_night['predicted_demand']:>8.2f}   "
          f"Tue 07:00 {c_morning['predicted_demand']:>8.2f}   "
          f"ratio {c_night['predicted_demand'] / max(c_morning['predicted_demand'], 1e-9):.2f}x")

    # --- 3. out-of-set zone is rejected ---------------------------------------
    print("\n3. Out-of-set zones are rejected, not silently answered")
    for zone_id in (103, 264, 999):
        served = forecaster.is_served(zone_id)
        print(f"   zone {zone_id:>3}: {'SERVED (unexpected)' if served else 'rejected'}")
        ok = ok and not served
    print()
    print("   " + reject(forecaster, 103).replace("\n", "\n   "))

    # --- 4. the conditions model responds to conditions -----------------------
    print("\n4. Conditions model uses weather and events (novelty #1, live)")
    if forecaster.conditions is None:
        print("   SKIPPED — not trained. Run: python -m src.batch.ablation")
    else:
        when = pd.Timestamp("2024-11-20 18:00")  # ordinary Wednesday evening
        dry = forecaster.predict_conditions(161, when, precip_mm=0.0)
        wet = forecaster.predict_conditions(161, when, precip_mm=5.0)
        coef = forecaster.conditions["coefficients"]["precip_x_hist"]
        moved = wet["predicted_demand"] != dry["predicted_demand"]
        right_way = (wet["predicted_demand"] > dry["predicted_demand"]) == (coef > 0)
        print(f"   zone 161, {when:%a %H:%M}: dry {dry['predicted_demand']:.2f} vs "
              f"5 mm rain {wet['predicted_demand']:.2f} "
              f"({wet['predicted_demand'] - dry['predicted_demand']:+.2f})")
        print(f"   prediction moves with rain: {moved}; direction matches the "
              f"fitted coefficient ({coef:+.5f}): {right_way}")
        ok = ok and moved and right_way

        plain = forecaster.predict_conditions(161, when)
        forced = forecaster.predict_conditions(161, when, force_event=True)
        event_moved = forced["predicted_demand"] != plain["predicted_demand"]
        print(f"   --is-event shifts the forecast: {event_moved} "
              f"({plain['predicted_demand']:.2f} -> {forced['predicted_demand']:.2f})")
        ok = ok and event_moved

        thanksgiving = forecaster.predict_conditions(161, pd.Timestamp("2024-11-28 15:00"))
        auto = thanksgiving["inputs"]["is_federal_holiday"] and thanksgiving["inputs"]["is_event"]
        print(f"   2024-11-28 auto-flagged from events.csv "
              f"(federal holiday + event): {auto}")
        ok = ok and auto

    print("\n" + "=" * 78)
    print("  RESULT:", "PASS" if ok else "FAIL")
    print("=" * 78)
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zone", type=int, help="pickup zone id (LocationID)")
    parser.add_argument("--at", help="timestamp, e.g. '2024-01-13 01:00' (NY local)")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    parser.add_argument("--validate", action="store_true", help="run the checks")
    # Served by the conditions model (models/conditions_model.json). If that model
    # has not been trained yet, they are reported as ignored — never silently.
    parser.add_argument("--temp", type=float,
                        help="temperature C (default: monthly normal)")
    parser.add_argument("--precip", type=float,
                        help="precipitation mm (default: 0, no rain)")
    parser.add_argument("--is-event", action="store_true",
                        help="force the event flag on (default: events.csv lookup)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    forecaster = Forecaster()

    if args.validate:
        return 0 if validate(forecaster) else 1

    if args.zone is None or args.at is None:
        print("ERROR: --zone and --at are both required (or use --validate)",
              file=sys.stderr)
        return 2

    if not forecaster.is_served(args.zone):
        print(reject(forecaster, args.zone), file=sys.stderr)
        return 3

    when = pd.Timestamp(args.at)
    result = forecaster.predict(args.zone, when)

    conditions = forecaster.predict_conditions(
        args.zone, when,
        temp_c=args.temp, precip_mm=args.precip, force_event=args.is_event,
    )
    ignored: dict = {}
    if conditions is not None:
        result["conditions"] = conditions
        result["models"] = {
            "shape": "K-Means cluster shape x zone level (served by the stream)",
            "conditions": "linear weather/events model (headline; src/batch/ablation.py)",
        }
    else:
        ignored = {
            name: getattr(args, name)
            for name in CONDITION_ARGS
            if getattr(args, name) not in (None, False)
        }
        result["ignored_inputs"] = ignored
        result["model_uses"] = ["zone_id", "hour_of_week"]
        result["model_ignores"] = ["weather", "events", "holidays"]

    if args.json:
        print(json.dumps(result, indent=2, default=float))
    else:
        print(render(result, ignored))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
