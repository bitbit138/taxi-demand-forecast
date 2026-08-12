"""Download NYC TLC yellow-taxi monthly parquet + the taxi-zone reference files.

Downloads the months listed in ``config.MONTHS`` (default: the 2024-01..2024-03 sample)
into ``data/raw/``, plus ``taxi_zone_lookup.csv`` and the taxi-zone shapefile into
``data/external/``.

Idempotent: a file already present with a matching size is skipped, so re-running is
cheap. Downloads land in a ``.part`` file and are renamed only on success, so an
interrupted run never leaves a truncated parquet behind.

    python -m src.ingest.download_tlc                 # sample months
    python -m src.ingest.download_tlc --months 2024-04 2024-05
    TAXI_MONTHS=full python -m src.ingest.download_tlc --yes   # all 12 months of 2024
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

CHUNK_BYTES = 1 << 20  # 1 MiB
TIMEOUT = (10, 120)    # (connect, read) seconds
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

# Downloading more than the sample is a deliberate act — see PROJECT_PLAN.md.
CONFIRM_THRESHOLD = len(config.SAMPLE_MONTHS)


def _human(num_bytes: float) -> str:
    """Format a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:,.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:,.1f} GB"


def _remote_size(url: str) -> int | None:
    """Content-Length for *url*, or None if the server does not report it.

    The CDN answers HEAD on taxi_zone_lookup.csv with ``Content-Length: 0`` even
    though the GET returns ~12 KB. Zero is therefore "unknown", not "empty file":
    taken literally it fails every download as a short read, and re-fetches the
    file forever because the cached copy can never match.
    """
    try:
        response = requests.head(url, timeout=TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        size = int(response.headers["Content-Length"])
    except (requests.RequestException, KeyError, ValueError):
        return None
    return size if size > 0 else None


def download(url: str, dest: Path, force: bool = False) -> tuple[bool, int]:
    """Stream *url* to *dest*. Returns ``(downloaded, size_bytes)``.

    Skips the download when *dest* already matches the remote Content-Length.
    """
    expected = _remote_size(url)

    if dest.exists() and not force:
        actual = dest.stat().st_size
        if expected is None or actual == expected:
            print(f"  [skip]     {dest.name} ({_human(actual)}, already present)")
            return False, actual
        print(f"  [re-fetch] {dest.name} is {_human(actual)}, expected {_human(expected)}")

    part = dest.with_suffix(dest.suffix + ".part")
    written = 0
    with requests.get(url, stream=True, timeout=TIMEOUT) as response:
        response.raise_for_status()
        with part.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                handle.write(chunk)
                written += len(chunk)
                # Only animate on a real terminal — piped to a log, \r turns one
                # download into hundreds of unreadable lines.
                if expected and sys.stdout.isatty():
                    pct = 100 * written / expected
                    print(f"\r  [get]      {dest.name}  {pct:5.1f}%", end="", flush=True)
    print(f"\r  [get]      {dest.name}  {_human(written)}          ")

    if expected is not None and written != expected:
        part.unlink(missing_ok=True)
        raise IOError(f"{dest.name}: got {written} bytes, expected {expected}")

    part.replace(dest)
    return True, written


def download_trip_months(months: list[str], force: bool = False) -> int:
    """Download one monthly parquet per entry in *months*. Returns total bytes."""
    print(f"\nTrip data -> {config.RAW_DIR}")
    total = 0
    for month in months:
        url = config.TLC_PARQUET_URL.format(month=month)
        dest = config.RAW_DIR / f"yellow_tripdata_{month}.parquet"
        _, size = download(url, dest, force=force)
        total += size
    return total


def download_zone_reference(force: bool = False) -> int:
    """Download the zone lookup CSV and the taxi-zone shapefile. Returns total bytes."""
    print(f"\nZone reference -> {config.EXTERNAL_DIR}")
    total = 0

    _, size = download(config.ZONE_LOOKUP_URL, config.ZONE_LOOKUP_CSV, force=force)
    total += size

    zip_path = config.EXTERNAL_DIR / "taxi_zones.zip"
    _, size = download(config.ZONE_SHAPEFILE_URL, zip_path, force=force)
    total += size

    if config.ZONE_SHAPEFILE.exists() and not force:
        print(f"  [skip]     {config.ZONE_SHAPEFILE.name} (already extracted)")
    else:
        # The archive already contains a top-level `taxi_zones/` folder, so extract
        # into data/external/ — extracting into taxi_zones/ would nest it twice.
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(config.EXTERNAL_DIR)
        print(f"  [unzip]    -> {config.ZONE_SHAPEFILE_DIR}")
        if not config.ZONE_SHAPEFILE.exists():
            found = sorted(str(p) for p in config.EXTERNAL_DIR.rglob("*.shp"))
            raise FileNotFoundError(
                f"Expected {config.ZONE_SHAPEFILE} after unzip; found .shp: {found}"
            )
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--months", nargs="+", metavar="YYYY-MM",
        help=f"override config.MONTHS (default: {' '.join(config.MONTHS)})",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help=f"confirm downloading more than {CONFIRM_THRESHOLD} months",
    )
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument(
        "--skip-zones", action="store_true", help="skip the lookup CSV and shapefile",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    months = args.months or config.MONTHS

    bad = [m for m in months if not MONTH_RE.match(m)]
    if bad:
        print(f"ERROR: not YYYY-MM: {', '.join(bad)}", file=sys.stderr)
        return 2

    if len(months) > CONFIRM_THRESHOLD and not args.force and not args.yes:
        print(
            f"ERROR: {len(months)} months requested ({months[0]}..{months[-1]}), which is "
            f"more than the {CONFIRM_THRESHOLD}-month sample.\n"
            f"       Roughly {len(months) * 50} MB. Re-run with --yes to confirm.",
            file=sys.stderr,
        )
        return 2

    print("=" * 68)
    print(f"TLC download | {len(months)} month(s): {', '.join(months)}")
    print("=" * 68)

    total = download_trip_months(months, force=args.force)
    if not args.skip_zones:
        total += download_zone_reference(force=args.force)

    parquet_files = sorted(config.RAW_DIR.glob("yellow_tripdata_*.parquet"))
    print("\n" + "=" * 68)
    print("DONE")
    print(f"  monthly parquet in data/raw : {len(parquet_files)}")
    print(f"  bytes accounted for         : {_human(total)}")
    print(f"  zone lookup                 : {config.ZONE_LOOKUP_CSV.exists()}")
    print(f"  zone shapefile              : {config.ZONE_SHAPEFILE.exists()}")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
