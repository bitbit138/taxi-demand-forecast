"""Single-query live forecaster: ``(zone_id, timestamp) -> predicted demand``.

Loads the same artifacts ``spark_stream.py`` serves — ``zone_clusters.parquet`` and
``cluster_demand.parquet`` — and applies the identical arithmetic, so a single query
returns exactly the number the stream and the batch produce for the same
``(zone, hour-of-week)``. Pandas rather than Spark: the artifacts are a few hundred
rows and a demo query should answer instantly, not spend 20 seconds starting a JVM.

**The model is a function of ``(zone, hour-of-week)`` and nothing else.** It is
K-Means over L1-normalised weekly demand profiles, so it knows a zone's *level* and its
cluster's *temporal shape*. It has no weather term and no event term. ``--temp``,
``--precip`` and ``--is-event`` are accepted so the interface is stable for a future
feature-based model, but they are **ignored**, and every response that supplies one says
so in as many words. That matters: on New Year's morning the East Village saw 440 trips
at 02:00 against a prediction of 88.65, because the model can only offer an average
Monday 2 a.m. An interface that silently accepted ``--is-event`` would imply an
event-awareness this model does not have.

    python -m src.stream.predict_live --zone 79  --at "2024-01-13 01:00"
    python -m src.stream.predict_live --zone 161 --at "2024-01-10 15:00" --json
    python -m src.stream.predict_live --validate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
IGNORED_ARGS = ("temp", "precip", "is_event")


def how_label(hour_of_week: int) -> str:
    """168-hour index -> e.g. 'Sat 01:00'."""
    return f"{DAY_NAMES[hour_of_week // 24]} {hour_of_week % 24:02d}:00"


def hour_of_week(when: pd.Timestamp) -> int:
    """Mon 00:00 = 0 .. Sun 23:00 = 167.

    ``Timestamp.weekday()`` is Mon=0..Sun=6, which is the same convention as the
    ``(dayofweek + 5) % 7`` remap used in features.py and spark_stream.py.
    """
    return when.weekday() * 24 + when.hour


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

    def _describe_clusters(self) -> dict[int, dict]:
        """Label each cluster from its own shape — never hard-coded.

        Derived at load time so the labels stay truthful if K or the data changes.
        """
        characters: dict[int, dict] = {}
        for cluster, group in self.shape.groupby("cluster"):
            shares = group.set_index("hour_of_week")["cluster_share"]
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

            n_zones = int((self.zones["cluster"] == cluster).sum())
            if n_zones <= 2:
                # A one- or two-zone cluster is an artifact of zones sitting just
                # above the exclusion floor: their profiles are mostly noise, so
                # naming a "character" would give the shape more credit than it has.
                label = f"single-zone artifact — sparse, mostly noise"
            elif weekend_night > 0.15:
                label = "nightlife — weekend small hours"
            elif weekday_morning > 0.20:
                label = "residential commute — weekday mornings"
            elif weekday_evening > weekday_morning:
                label = "business core — weekday evenings"
            else:
                label = "mixed / low-signal"

            characters[int(cluster)] = {
                "label": label,
                "peak_how": peak,
                "peak_label": how_label(peak),
                "peak_share": float(shares.max()),
                "weekend_night_share": weekend_night,
                "weekday_morning_share": weekday_morning,
                "n_zones": n_zones,
            }
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
    lines = [
        "=" * 72,
        f"PREDICTED DEMAND  {result['predicted_demand_rounded']:.2f} trips",
        "=" * 72,
        f"  query            zone {result['zone_id']} "
        f"({result['zone_name']}, {result['borough']})",
        f"                   {result['timestamp']}  "
        f"= {result['hour_of_week_label']} (hour-of-week "
        f"{result['hour_of_week']})",
        "",
        "  how it got there",
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
    if ignored:
        lines += [
            "",
            "  IGNORED INPUTS",
            f"    {', '.join(f'{k}={v}' for k, v in ignored.items())}",
            "    This model is K-Means over weekly demand SHAPE — a function of",
            "    (zone, hour-of-week) only. It has no weather or event term, so the",
            "    values above changed nothing. Accepted for forward compatibility.",
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
    # Accepted for forward compatibility; the current model ignores them entirely.
    parser.add_argument("--temp", type=float, help="temperature C (IGNORED by this model)")
    parser.add_argument("--precip", type=float, help="precipitation mm (IGNORED)")
    parser.add_argument("--is-event", action="store_true", help="event flag (IGNORED)")
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

    ignored = {
        name: getattr(args, name)
        for name in IGNORED_ARGS
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
