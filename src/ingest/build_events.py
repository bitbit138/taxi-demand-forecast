"""Build the holiday / event calendar -> ``data/external/events.csv``.

Emits **one row per calendar day of the full year**, not just the flagged days, so
``features.py`` can left-join on ``date_local`` and never deal with nulls. The full year
is always covered even when ``config.MONTHS`` is the 3-month sample, so scaling to the
full year needs no regeneration of this table.

Keyed on ``date_local`` — the same NY-local calendar date as ``weather.parquet`` and as
TLC pickup timestamps, so joins carry no timezone ambiguity. Holidays are naturally
local-calendar facts; there is no UTC conversion anywhere in this file by design.

``is_holiday`` and ``is_event`` are **independent** booleans. A date can be both
(New Year's Day, 4 July, Thanksgiving), and each keeps its own name column so neither
label overwrites the other.

    python -m src.ingest.build_events
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import holidays
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

# US federal + New York State observances.
HOLIDAY_COUNTRY = "US"
HOLIDAY_SUBDIV = "NY"


def calendar_year() -> int:
    """The year the events table covers (taken from the configured range)."""
    return pd.Timestamp(config.START_DATE).year


def build_events(year: int) -> pd.DataFrame:
    """One row per day of *year*, flagged for holidays and curated NYC events."""
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")

    us_ny = holidays.country_holidays(HOLIDAY_COUNTRY, subdiv=HOLIDAY_SUBDIV, years=year)
    holiday_names = {pd.Timestamp(d): name for d, name in us_ny.items()}

    # Federal holidays close offices and shift travel patterns; NY-only observances
    # (Lincoln's Birthday, Susan B. Anthony Day, Election Day) mostly do not. Keeping
    # them apart lets features.py use whichever signal actually helps.
    federal = holidays.country_holidays(HOLIDAY_COUNTRY, years=year)
    federal_dates = {pd.Timestamp(d) for d in federal}

    curated = _validated_events(year)

    frame = pd.DataFrame({"date": days})
    frame["date_local"] = frame["date"].dt.strftime("%Y-%m-%d")
    frame["day_of_week"] = frame["date"].dt.day_name()

    frame["holiday_name"] = frame["date"].map(holiday_names).fillna("")
    frame["event_name"] = frame["date"].map(curated).fillna("")

    frame["is_holiday"] = frame["holiday_name"].ne("")
    frame["is_federal_holiday"] = frame["date"].isin(federal_dates)
    frame["is_event"] = frame["event_name"].ne("")
    # Explicit, so a downstream reader cannot mistake one flag for the other.
    frame["is_holiday_and_event"] = frame["is_holiday"] & frame["is_event"]
    # Convenience union: "is today unusual for any reason".
    frame["is_special_day"] = frame["is_holiday"] | frame["is_event"]

    return frame[
        [
            "date_local",
            "day_of_week",
            "is_holiday",
            "is_federal_holiday",
            "is_event",
            "is_holiday_and_event",
            "is_special_day",
            "holiday_name",
            "event_name",
        ]
    ]


def _validated_events(year: int) -> dict[pd.Timestamp, str]:
    """Parse ``config.NYC_EVENT_DATES``, rejecting non-dates and wrong-year entries."""
    parsed: dict[pd.Timestamp, str] = {}
    problems: list[str] = []

    for raw, label in config.NYC_EVENT_DATES.items():
        try:
            # errors="raise" so 2024-02-30 fails loudly instead of becoming NaT.
            stamp = pd.to_datetime(raw, format="%Y-%m-%d", errors="raise")
        except ValueError:
            problems.append(f"{raw!r} ({label}) is not a real calendar date")
            continue
        if stamp.year != year:
            problems.append(f"{raw!r} ({label}) is not in {year}")
            continue
        if stamp in parsed:
            problems.append(f"{raw!r} is listed twice in config.NYC_EVENT_DATES")
            continue
        parsed[stamp] = label

    if problems:
        raise ValueError(
            "config.NYC_EVENT_DATES has bad entries:\n  " + "\n  ".join(problems)
        )
    return parsed


def validate(frame: pd.DataFrame, year: int) -> bool:
    """Coverage, uniqueness and calendar-reality checks."""
    ok = True
    print("\n" + "-" * 68)
    print("VALIDATION")
    print("-" * 68)

    expected_days = 366 if pd.Timestamp(f"{year}-12-31").dayofyear == 366 else 365
    print(f"  rows             : {len(frame)} (expected {expected_days} for {year})")
    ok = ok and len(frame) == expected_days

    dupes = int(frame["date_local"].duplicated().sum())
    print(f"  duplicate dates  : {dupes}")
    ok = ok and dupes == 0

    # Every date must round-trip as a real calendar day, and the series must be
    # contiguous with no missing or repeated days.
    parsed = pd.to_datetime(frame["date_local"], format="%Y-%m-%d", errors="coerce")
    unreal = int(parsed.isna().sum())
    print(f"  unparseable dates: {unreal}")
    ok = ok and unreal == 0

    gaps = parsed.diff().dropna().ne(pd.Timedelta(days=1)).sum()
    print(f"  calendar gaps    : {gaps}")
    ok = ok and gaps == 0
    print(f"  range            : {frame['date_local'].iloc[0]} .. {frame['date_local'].iloc[-1]}")

    # Curated events must all have landed on a row.
    curated = set(config.NYC_EVENT_DATES)
    flagged = set(frame.loc[frame["is_event"], "date_local"])
    lost = curated - flagged
    print(f"  curated events   : {len(curated)} configured, {len(flagged)} flagged")
    if lost:
        ok = False
        print(f"    NOT FLAGGED    : {sorted(lost)}")

    for label, sub in (
        (f"full year {year}", frame),
        ("Q1 (Jan-Mar)   ", frame[frame["date_local"] <= f"{year}-03-31"]),
    ):
        print(f"\n  {label} : {int(sub['is_holiday'].sum())} holiday days "
              f"({int(sub['is_federal_holiday'].sum())} federal, "
              f"{int(sub['is_holiday'].sum() - sub['is_federal_holiday'].sum())} NY-only)")
        print(f"  {' ' * len(label)} : {int(sub['is_event'].sum())} event days")
        print(f"  {' ' * len(label)} : {int(sub['is_special_day'].sum())} distinct special days")

    both = frame[frame["is_holiday_and_event"]]
    print(f"\n  dates that are BOTH holiday and curated event: {len(both)}")
    for _, row in both.iterrows():
        print(f"    {row['date_local']} ({row['day_of_week'][:3]})  "
              f"holiday={row['holiday_name']!r}  event={row['event_name']!r}")
    if len(both) > 0:
        print("    -> both flags set; neither name overwritten")

    print("-" * 68)
    print("  RESULT:", "PASS" if ok else "FAIL")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--year", type=int, default=None,
        help="calendar year to build (default: the year in config.START_DATE)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    year = args.year or calendar_year()

    print("=" * 68)
    print(f"Event calendar for {year}")
    print("=" * 68)
    print(f"  holidays source  : {HOLIDAY_COUNTRY}/{HOLIDAY_SUBDIV} via `holidays` "
          f"{holidays.__version__}")
    print(f"  curated events   : {len(config.NYC_EVENT_DATES)} from config.NYC_EVENT_DATES")
    print(f"  coverage         : full year (config.MONTHS is currently "
          f"{len(config.MONTHS)} month(s))")

    frame = build_events(year)
    frame.to_csv(config.EVENTS_CSV, index=False)

    ok = validate(frame, year)

    print("\n" + "=" * 68)
    print("DONE")
    print(f"  file : {config.EVENTS_CSV}")
    print(f"  rows : {len(frame)}")
    print("=" * 68)

    print(f"\nAll flagged days in Q1 {year}:")
    q1 = frame[(frame["date_local"] <= f"{year}-03-31") & frame["is_special_day"]]
    print(q1[["date_local", "day_of_week", "is_holiday", "is_federal_holiday",
              "is_event", "holiday_name", "event_name"]].to_string(index=False))

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
