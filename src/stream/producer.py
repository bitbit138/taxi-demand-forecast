"""Replay historical TLC trips into Kafka as a time-accelerated JSON stream.

A plain producer: it reads ``data/raw``, sorts by pickup time, and emits one JSON
message per trip. **No cleaning, no aggregation** — ``spark_stream.py`` reapplies the
cleaning filters and does the windowing, exactly as it would for a genuinely live feed.

**Event time comes from the data, not the clock.** Every message carries
``tpep_pickup_datetime`` (naive NY-local wall-clock, as stored in the TLC parquet), and
the Kafka record timestamp is set to the same instant. Structured Streaming windows and
watermarks are driven by that field, so a replay produces the same windows regardless of
when it is run or how fast.

The record timestamp is the wall-clock value interpreted as UTC — i.e. NY-local
"2024-01-01 00:00:04" becomes epoch 1704067204000, which reads back as the same
wall-clock. That is the project-wide convention (``spark.sql.session.timeZone = UTC``
over naive local timestamps, see ``src/spark_session.py``): nothing is ever shifted.
``spark_stream.py`` parses event time from the payload field regardless, so the record
timestamp is corroboration rather than the source of truth.

**Acceleration.** ``config.REPLAY_SPEEDUP`` maps simulated time to real time and the
mapping is logged on every run. 3600 means 1 real second == 1 simulated hour, so a
simulated day takes 24 seconds. Lower it for a live presentation.

**Re-running.** Kafka topics are append-only: re-running never corrupts the topic, it
appends a second copy of the slice, which would double-count in the windowed
aggregation. The producer therefore refuses to write to a non-empty topic unless told
what to do:

    python -m src.stream.producer --reset-topic   # clean slate (recommended for a demo)
    python -m src.stream.producer --append        # deliberately add to what is there

Manual reset, equivalent to --reset-topic::

    docker exec taxi-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server \
      localhost:9092 --delete --topic taxi-trips
    docker compose up -d kafka-init --force-recreate

Usage::

    python -m src.stream.producer --dry-run            # print messages, write nothing
    python -m src.stream.producer --hours 6 --reset-topic
    python -m src.stream.producer --hours 0 --speedup 86400   # whole range, fast
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config  # noqa: E402

FIELD_NAMES = [name for name, _ in config.KAFKA_MESSAGE_FIELDS]
# Columns to read from parquet: the message fields, all of which are TLC columns.
READ_COLUMNS = list(dict.fromkeys(FIELD_NAMES))

# Topic reset: how long one delete gets to take effect, and how many deletes to try
# when something recreates the topic underneath us (see reset_topic).
RESET_ABSENT_TIMEOUT_S = 20
RESET_DELETE_ATTEMPTS = 3


def _import_kafka():
    """Import kafka lazily so --dry-run works without a broker installed."""
    try:
        from kafka import KafkaAdminClient, KafkaProducer  # noqa: PLC0415
        from kafka.admin import NewTopic  # noqa: PLC0415
        from kafka.errors import UnknownTopicOrPartitionError  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "kafka-python-ng is not installed — pip install -r requirements.txt"
        ) from exc
    return KafkaAdminClient, KafkaProducer, NewTopic, UnknownTopicOrPartitionError


def load_slice(start: pd.Timestamp, end: pd.Timestamp | None) -> pd.DataFrame:
    """Read the trips whose pickup falls in ``[start, end)``, sorted by pickup.

    Only the months overlapping the window are opened, and the row filter is pushed
    into the parquet reader, so a short demo slice never materialises the full 9.5M
    rows.
    """
    wanted_months = []
    for month in config.MONTHS:
        month_start = pd.Timestamp(f"{month}-01")
        month_end = month_start + pd.offsets.MonthBegin(1)
        if month_end > start and (end is None or month_start < end):
            wanted_months.append(month)

    if not wanted_months:
        raise ValueError(f"No configured month covers {start} .. {end}")

    filters = [(config.KAFKA_EVENT_TIME_FIELD, ">=", start.to_pydatetime())]
    if end is not None:
        filters.append((config.KAFKA_EVENT_TIME_FIELD, "<", end.to_pydatetime()))

    frames = []
    for month in wanted_months:
        path = config.RAW_DIR / f"yellow_tripdata_{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run: python -m src.ingest.download_tlc"
            )
        table = pq.read_table(path, columns=READ_COLUMNS, filters=filters)
        frames.append(table.to_pandas())

    trips = pd.concat(frames, ignore_index=True)
    trips = trips.dropna(subset=[config.KAFKA_EVENT_TIME_FIELD])
    return trips.sort_values(config.KAFKA_EVENT_TIME_FIELD, ignore_index=True)


def to_message(row: pd.Series) -> dict:
    """One trip -> the JSON payload. No cleaning: nulls and bad values pass through."""
    message = {}
    for name, spark_type in config.KAFKA_MESSAGE_FIELDS:
        value = row[name]
        if pd.isna(value):
            message[name] = None
        elif spark_type == "string":
            message[name] = pd.Timestamp(value).strftime(config.KAFKA_EVENT_TIME_STRFTIME)
        elif spark_type == "int":
            message[name] = int(value)
        else:
            message[name] = float(value)
    return message


def topic_message_count(admin_cls, bootstrap: str) -> int:
    """Total messages currently on the topic, across partitions."""
    from kafka import KafkaConsumer  # noqa: PLC0415

    consumer = KafkaConsumer(
        bootstrap_servers=bootstrap, consumer_timeout_ms=5000, group_id=None
    )
    try:
        partitions = consumer.partitions_for_topic(config.KAFKA_TOPIC)
        if not partitions:
            return 0
        from kafka import TopicPartition  # noqa: PLC0415

        tps = [TopicPartition(config.KAFKA_TOPIC, p) for p in partitions]
        beginning = consumer.beginning_offsets(tps)
        end = consumer.end_offsets(tps)
        return sum(end[tp] - beginning[tp] for tp in tps)
    finally:
        consumer.close()


def _wait_until_absent(admin, timeout_s: int) -> bool:
    """True once the broker stops listing the topic. Deletion is asynchronous."""
    for _ in range(timeout_s):
        time.sleep(1.0)
        if config.KAFKA_TOPIC not in admin.list_topics():
            return True
    return False


def reset_topic(admin_cls, new_topic_cls, unknown_topic_exc, bootstrap: str) -> None:
    """Delete and recreate the topic so a demo starts from a clean slate.

    A single delete is not enough to guarantee the topic stays gone. Two things
    recreate it behind our back: the broker runs with auto.create.topics.enable=true,
    so any client asking for its metadata resurrects it, and docker-compose's
    ``kafka-init`` container re-runs ``--create --if-not-exists`` on every
    ``docker compose up -d``. Losing that race used to look like "the delete never
    took effect" and failed the demo, so re-issue the delete instead of giving up.
    """
    admin = admin_cls(bootstrap_servers=bootstrap, client_id="taxi-producer-admin")
    try:
        for attempt in range(1, RESET_DELETE_ATTEMPTS + 1):
            try:
                admin.delete_topics([config.KAFKA_TOPIC])
                print(f"  deleted topic {config.KAFKA_TOPIC}")
            except unknown_topic_exc:
                print(f"  topic {config.KAFKA_TOPIC} did not exist")

            if _wait_until_absent(admin, RESET_ABSENT_TIMEOUT_S):
                admin.create_topics(
                    [
                        new_topic_cls(
                            name=config.KAFKA_TOPIC,
                            num_partitions=config.KAFKA_NUM_PARTITIONS,
                            replication_factor=config.KAFKA_REPLICATION_FACTOR,
                        )
                    ]
                )
                print(f"  recreated topic {config.KAFKA_TOPIC} "
                      f"({config.KAFKA_NUM_PARTITIONS} partitions)")
                return

            print(f"  {config.KAFKA_TOPIC} was recreated within "
                  f"{RESET_ABSENT_TIMEOUT_S}s (attempt {attempt}/"
                  f"{RESET_DELETE_ATTEMPTS}) — retrying the delete")

        # Whatever recreated it made a *fresh* topic, so the only thing this function
        # actually owes the demo — an empty topic — may already hold. Accept that
        # rather than failing; only a topic with messages on it is a real problem.
        remaining = topic_message_count(admin_cls, bootstrap)
        if remaining == 0:
            print(f"  {config.KAFKA_TOPIC} keeps being recreated (kafka-init or "
                  f"broker auto-create) but is empty — continuing")
            return
        raise TimeoutError(
            f"{config.KAFKA_TOPIC} still holds {remaining:,} messages after "
            f"{RESET_DELETE_ATTEMPTS} delete attempts — a live client keeps "
            f"recreating it. Close any stray demo windows (Spark stream, producer) "
            f"and re-run."
        )
    finally:
        admin.close()


def build_producer(producer_cls, bootstrap: str):
    """Producer configured for ordered, at-least-once delivery."""
    kwargs = dict(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
        acks="all",
        retries=5,
        # Ordering matters for a replay: without this a retry can reorder events
        # within a partition and the windowed aggregation sees them out of sequence.
        max_in_flight_requests_per_connection=1,
        linger_ms=50,
        compression_type="gzip",
    )
    try:
        return producer_cls(enable_idempotence=True, **kwargs)
    except (TypeError, AssertionError):
        # kafka-python-ng asserts on unrecognised configs rather than raising
        # TypeError. Without broker-side idempotence, acks=all plus ordered retries
        # still gives at-least-once with per-partition order preserved.
        print("  note: client has no enable_idempotence; using acks=all with "
              "ordered retries (at-least-once)")
        return producer_cls(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hours", type=float, default=config.REPLAY_DEFAULT_HOURS,
        help=f"simulated hours to replay, 0 = whole range "
             f"(default {config.REPLAY_DEFAULT_HOURS})",
    )
    parser.add_argument(
        "--start", default=None,
        help=f"simulated start timestamp (default {config.START_DATE} 00:00:00)",
    )
    parser.add_argument(
        "--speedup", type=float, default=config.REPLAY_SPEEDUP,
        help=f"simulated seconds per real second (default {config.REPLAY_SPEEDUP:g})",
    )
    parser.add_argument(
        "--max-messages", type=int, default=config.REPLAY_MAX_MESSAGES,
        help=f"safety cap (default {config.REPLAY_MAX_MESSAGES:,})",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--reset-topic", action="store_true",
        help="delete and recreate the topic first (clean demo)",
    )
    group.add_argument(
        "--append", action="store_true",
        help="produce into a non-empty topic on purpose",
    )
    group.add_argument(
        "--reset-only", action="store_true",
        help="delete and recreate the topic, then exit without producing. Use this "
             "BEFORE starting a consumer — resetting while a stream is subscribed "
             "can fault the query on a vanished topic.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the first messages and exit; no broker needed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.reset_only:
        admin_cls, _, new_topic_cls, unknown_exc = _import_kafka()
        print("Resetting topic before any consumer attaches...")
        reset_topic(admin_cls, new_topic_cls, unknown_exc, config.KAFKA_BOOTSTRAP_SERVERS)
        print(f"  {config.KAFKA_TOPIC} is empty and ready.")
        return 0

    start = pd.Timestamp(args.start) if args.start else pd.Timestamp(
        f"{config.START_DATE} 00:00:00"
    )
    end = None if args.hours <= 0 else start + pd.Timedelta(hours=args.hours)

    sim_per_real = args.speedup
    print("=" * 78)
    print("Kafka replay producer")
    print("=" * 78)
    print(f"  topic          : {config.KAFKA_TOPIC} @ {config.KAFKA_BOOTSTRAP_SERVERS}")
    print(f"  simulated span : {start} .. {end if end else 'end of range'}")
    print(f"  speedup        : {sim_per_real:g}x")
    print(f"                   1 real second = {sim_per_real / 3600:g} simulated hours")
    print(f"                   1 simulated day = {86400 / sim_per_real:.1f} real seconds")
    print(f"  event time     : {config.KAFKA_EVENT_TIME_FIELD} "
          f"(from the data, never wall-clock)")
    print(f"  key            : {config.KAFKA_MESSAGE_KEY_FIELD}")
    print(f"  fields         : {', '.join(FIELD_NAMES)}")

    print("\nLoading slice...")
    trips = load_slice(start, end)
    total = len(trips)
    if args.max_messages and total > args.max_messages:
        print(f"  {total:,} trips in slice — capping at {args.max_messages:,}")
        trips = trips.head(args.max_messages)
        total = len(trips)
    print(f"  {total:,} trips, "
          f"{trips[config.KAFKA_EVENT_TIME_FIELD].min()} .. "
          f"{trips[config.KAFKA_EVENT_TIME_FIELD].max()}")

    if total == 0:
        print("Nothing to produce.")
        return 0

    if args.dry_run:
        print("\nDRY RUN — first 3 messages exactly as they would be sent:\n")
        for _, row in trips.head(3).iterrows():
            message = to_message(row)
            print(f"  key={message[config.KAFKA_MESSAGE_KEY_FIELD]}")
            print(f"  value={json.dumps(message)}")
            print()
        return 0

    admin_cls, producer_cls, new_topic_cls, unknown_exc = _import_kafka()

    print("\nTopic state:")
    if args.reset_topic:
        reset_topic(admin_cls, new_topic_cls, unknown_exc, config.KAFKA_BOOTSTRAP_SERVERS)
        existing = 0
    else:
        existing = topic_message_count(admin_cls, config.KAFKA_BOOTSTRAP_SERVERS)
        print(f"  {existing:,} messages already on {config.KAFKA_TOPIC}")
        if existing and not args.append:
            print("\n" + "!" * 78)
            print("REFUSING TO PRODUCE — the topic is not empty.")
            print("Appending a second copy of the same slice would double-count in the")
            print("windowed aggregation. Choose one:")
            print("  --reset-topic   delete and recreate the topic (clean demo)")
            print("  --append        add to what is already there, deliberately")
            print("!" * 78)
            return 3

    producer = build_producer(producer_cls, config.KAFKA_BOOTSTRAP_SERVERS)

    print(f"\nProducing {total:,} messages...")
    t_zero = trips[config.KAFKA_EVENT_TIME_FIELD].iloc[0]
    wall_zero = time.monotonic()
    sent = 0
    slept = 0.0

    try:
        for _, row in trips.iterrows():
            event_time = row[config.KAFKA_EVENT_TIME_FIELD]

            # Pace by the data's own clock, compressed by the speedup factor.
            sim_elapsed = (event_time - t_zero).total_seconds()
            target = wall_zero + sim_elapsed / sim_per_real
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
                slept += delay

            message = to_message(row)
            producer.send(
                config.KAFKA_TOPIC,
                key=message[config.KAFKA_MESSAGE_KEY_FIELD],
                value=message,
                # Record timestamp == event time, so the broker-side timestamp
                # agrees with the payload rather than with the replay clock.
                timestamp_ms=int(pd.Timestamp(event_time).timestamp() * 1000),
            )
            sent += 1

            if sent % config.REPLAY_PROGRESS_EVERY == 0:
                real = time.monotonic() - wall_zero
                print(f"  {sent:>8,}/{total:,}  sim {event_time}  "
                      f"real {real:>6.1f}s  {sent / max(real, 1e-9):>8,.0f} msg/s")
    except KeyboardInterrupt:
        print("\n  interrupted — flushing what has been sent")
    finally:
        producer.flush()
        producer.close()

    real_total = time.monotonic() - wall_zero
    sim_total = (
        trips[config.KAFKA_EVENT_TIME_FIELD].iloc[min(sent, total) - 1] - t_zero
    ).total_seconds()

    expected_real = sim_total / sim_per_real

    print("\n" + "=" * 78)
    print("DONE")
    print(f"  sent           : {sent:,} messages")
    print(f"  simulated span : {sim_total / 3600:.2f} hours")
    print(f"  real elapsed   : {real_total:.1f} s (of which {slept:.1f} s pacing)")
    print(f"  target elapsed : {expected_real:.1f} s at {sim_per_real:g}x")
    print(f"  effective rate : {sent / max(real_total, 1e-9):,.0f} msg/s")
    print(f"  topic total    : {existing + sent:,} messages")

    # A demo is only reproducible if the requested speedup was actually achieved.
    # Above a few hundred x the producer becomes throughput-bound on busy hours and
    # silently runs slower than asked, which would make timings differ per machine.
    if expected_real > 0 and real_total > expected_real * 1.2:
        achieved = sim_total / real_total
        print("\n  " + "!" * 74)
        print(f"  SPEEDUP NOT ACHIEVED — asked {sim_per_real:g}x, actually ran "
              f"{achieved:,.0f}x.")
        print("  The producer was throughput-bound, not pacing: it never slept, so the")
        print("  stream was emitted as fast as the client could serialise. Event-time")
        print("  windows are unaffected (they come from the data), but wall-clock")
        print(f"  timings will vary by machine. For a reproducible demo use")
        print(f"  --speedup {max(60, int(achieved * 0.6 // 60 * 60)):g} or lower.")
        print("  " + "!" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
