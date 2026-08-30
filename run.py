#!/usr/bin/env python3
"""Cross-platform launcher — the three ``run_*.bat`` files, for Windows AND macOS/Linux.

Standard library only, so it runs under any Python 3.9+; every pipeline step is
then executed with the project's ``.venv`` interpreter (``.venv\\Scripts\\python.exe``
on Windows, ``.venv/bin/python`` elsewhere — override with ``VENV_PY=<path>`` or
``--python``).

    python run.py pipeline            # steps 1-7, then sweep K and stop for review
    python run.py pipeline k 5        # commit to K=5: train, evaluate, ablation, maps, ...
    python run.py pipeline smoke      # 3-month sample, K pinned to 4 (sequencing check)
    python run.py demo                # Kafka + Spark Structured Streaming, two windows
    python run.py gui                 # rebuild gui/payload.json, then serve on :8765
    python run.py gui serve           # serve the committed payload, no rebuild
    python run.py gui build           # rebuild only

    python run.py <anything> --dry-run   # print every command instead of running it

Behaviour mirrors the .bat files step for step (same order, same halting on first
failure, same prerequisite checks, same "smoke skips F/G" rule). Two deliberate
differences, both for portability: the streaming consumer is launched in module
form (``python -m src.stream.spark_stream``, which resolves the Kafka connector from
``config.py``) rather than through ``spark-submit.cmd``, and the new terminal windows
are opened with the platform's own mechanism (``start`` / Terminal.app / a Linux
terminal), falling back to background processes with log files when no terminal
can be opened.
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IS_WINDOWS = os.name == "nt"
IS_MACOS = platform.system() == "Darwin"

# --- demo constants (identical to run_demo.bat) -------------------------------
READY_FILE = ROOT / ".stream_ready"
READY_TIMEOUT = 180
SPEEDUP = 600
DEMO_START = "2024-01-10 14:00:00"
DEMO_HOURS = 6
STREAM_SECONDS = 420
GUI_PORT = 8765

DRY_RUN = False


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def banner(title: str) -> None:
    print("=" * 76)
    print(f" {title}")
    print("=" * 76)


def fail(message: str, code: int = 1) -> "NoReturn":  # noqa: F821
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(code)


def venv_python(override: str | None) -> str:
    """The project's interpreter, by full path — never whatever ``python`` is on PATH."""
    candidate = override or os.environ.get("VENV_PY")
    if not candidate:
        candidate = str(
            ROOT / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
        )
    if Path(candidate).exists():
        return candidate
    if DRY_RUN:
        print(f"  (dry-run) venv interpreter not found at {candidate}; showing commands anyway")
        return candidate
    create = (
        "py -3.11 -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt"
        if IS_WINDOWS
        else "python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    )
    fail(f".venv not found at {candidate}. Create it first:\n    {create}\n"
         "  or point VENV_PY / --python at an existing interpreter.")


def require_java_home() -> str:
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if java_home:
        return java_home
    if DRY_RUN:
        return "<unset>"
    hint = (
        "set it to a Temurin 17 install, e.g.\n"
        "    [Environment]::SetEnvironmentVariable('JAVA_HOME', 'C:\\Program Files\\Eclipse Adoptium\\jdk-17...', 'User')"
        if IS_WINDOWS
        else "e.g.\n    export JAVA_HOME=$(/usr/libexec/java_home -v 17)\n"
             "  or with Homebrew:\n"
             "    export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
    )
    fail(f"JAVA_HOME is not set. Spark 3.5.3 needs a Java 17 JDK — {hint}")


def run_step(label: str, argv: list[str], env: dict[str, str]) -> None:
    """Run one step in the foreground; stop the whole run on the first failure."""
    print(f"\n[{label}]")
    print("   $ " + " ".join(shlex.quote(a) for a in argv))
    if DRY_RUN:
        return
    result = subprocess.run(argv, cwd=ROOT, env=env)
    if result.returncode != 0:
        print("\n" + "#" * 76)
        print(f" FAILED at step {label} (exit code {result.returncode}).")
        print(" Nothing after this step was run. Completed steps are idempotent and")
        print(" will be skipped or overwritten safely on re-run.")
        print("#" * 76)
        sys.exit(result.returncode or 1)


def open_in_terminal(title: str, argv: list[str], env: dict[str, str],
                     exports: dict[str, str]) -> None:
    """Open ``argv`` in a NEW terminal window (or a logged background process).

    ``exports`` are the environment variables the child must see even when the
    platform's terminal does not inherit this process's environment (Terminal.app).
    """
    cmd = subprocess.list2cmdline(argv) if IS_WINDOWS else " ".join(shlex.quote(a) for a in argv)
    print(f"   -> new window '{title}': {cmd}")
    if DRY_RUN:
        return

    if IS_WINDOWS:
        subprocess.Popen(f'start "{title}" cmd /k "{cmd}"', shell=True, cwd=ROOT, env=env)
        return

    prefix = "".join(f"export {k}={shlex.quote(v)}; " for k, v in exports.items())
    script = f"cd {shlex.quote(str(ROOT))} && {prefix}{cmd}"

    if IS_MACOS:
        escaped = script.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.run(
            ["osascript", "-e", 'tell application "Terminal" to activate',
             "-e", f'tell application "Terminal" to do script "{escaped}"'],
            check=True,
        )
        return

    for terminal in (["gnome-terminal", "--"], ["konsole", "-e"], ["xterm", "-e"]):
        if shutil.which(terminal[0]):
            subprocess.Popen(terminal + ["bash", "-c", f"{script}; exec bash"], cwd=ROOT)
            return

    # No terminal available (SSH session, CI): run in the background with a log.
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    log_path = logs / (title.lower().replace(" ", "_") + ".log")
    child_env = {**env, **exports}
    with log_path.open("w") as handle:
        subprocess.Popen(argv, cwd=ROOT, env=child_env, stdout=handle, stderr=subprocess.STDOUT)
    print(f"      (no terminal found — running in the background, log: {log_path})")


def docker_inspect(container: str, template: str) -> str:
    try:
        out = subprocess.run(
            ["docker", "inspect", "-f", template, container],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except FileNotFoundError:
        fail("docker is not installed or not on PATH.")


def wait_until(description: str, predicate, timeout: int, hint: str) -> None:
    waited = 0
    while not predicate():
        if waited >= timeout:
            fail(f"{description} not reached after {timeout}s. {hint}")
        time.sleep(3)
        waited += 3
        print(f"        ... waiting ({waited}s)")


# --------------------------------------------------------------------------- #
# pipeline — mirrors run_pipeline_full_year.bat
# --------------------------------------------------------------------------- #
def cmd_pipeline(args: argparse.Namespace) -> None:
    mode = "sweep"
    chosen_k: str | None = None
    if args.mode:
        first = args.mode[0].lower()
        if first == "smoke":
            mode = "smoke"
        elif first == "k":
            if len(args.mode) < 2:
                fail('"k" mode needs a value, e.g.  python run.py pipeline k 5')
            mode, chosen_k = "finish", args.mode[1]
        elif first == "sweep":
            mode = "sweep"
        else:
            fail(f"unknown pipeline mode {args.mode!r} — use: sweep | smoke | k <N>")

    py = venv_python(args.python)
    java_home = require_java_home()

    env = dict(os.environ)
    if mode == "smoke":
        env["TAXI_MONTHS"] = "sample"
        label = "SMOKE TEST - 3-month sample, K pinned to 4"
        download_args: list[str] = []
    else:
        env["TAXI_MONTHS"] = "full"
        label = "FULL YEAR - 12 months of 2024"
        download_args = ["--yes"]  # confirms downloading more than the sample

    banner(f"Taxi demand pipeline - {label}")
    print(f"  python       {py}")
    print(f"  JAVA_HOME    {java_home}")
    print(f"  TAXI_MONTHS  {env['TAXI_MONTHS']}")
    print(f"  mode         {mode} {chosen_k or ''}")

    if mode != "finish":
        run_step("1/8 download TLC parquet + zone lookup + shapefile",
                 [py, "-m", "src.ingest.download_tlc", *download_args], env)
        run_step("2/8 fetch Open-Meteo hourly weather per zone centroid",
                 [py, "-m", "src.ingest.fetch_weather"], env)
        run_step("3/8 build holiday + event calendar",
                 [py, "-m", "src.ingest.build_events"], env)
        run_step("4/8 clean trips and aggregate to hourly demand per zone",
                 [py, "-m", "src.batch.clean_aggregate"], env)
        run_step("5/8 apply the zone exclusion policy",
                 [py, "-m", "src.batch.zone_policy"], env)
        run_step("6/8 Sedona geo join - centroids and polygons",
                 [py, "-m", "src.batch.geo_join"], env)
        run_step("7/8 build the modelling feature table",
                 [py, "-m", "src.batch.features"], env)

        if mode == "sweep":
            print("\n[8/8] Sweep K on the full-year data and describe the candidates")
            print("      K is NOT pinned: a full year adds seasonal structure the Q1")
            print("      sample could not show, so the choice is re-made on this data.")
            run_step("8/8 train_kmeans --inspect",
                     [py, "-m", "src.batch.train_kmeans", "--inspect", "--candidates", "3"], env)
            banner("SWEEP COMPLETE - REVIEW REQUIRED, nothing was saved")
            print("  Read the cluster characters printed above and reports/k_sweep.csv,")
            print("  then commit to a K:")
            print("      python run.py pipeline k 5      (if 5 still holds)")
            print("      python run.py pipeline k <N>    (if the data splits differently)")
            return

        chosen_k = "4"  # smoke: pinned, a sequencing check rather than a model choice
        print("\n[8/8] Train K-Means, evaluate, render  (smoke: K pinned to 4)")

    run_step(f"A train K-Means with K={chosen_k} and save the model",
             [py, "-m", "src.batch.train_kmeans", "--k", str(chosen_k)], env)
    run_step("B score the baseline ladder on the held-out split",
             [py, "-m", "src.batch.evaluate"], env)
    run_step("C weather/events ablation (open question #2) + conditions model export",
             [py, "-m", "src.batch.ablation"], env)
    run_step("D render report figures, maps and GeoJSON",
             [py, "-m", "src.viz.make_maps"], env)
    run_step("E block bootstrap - is the weather/events gain statistically real?",
             [py, "-m", "src.batch.significance"], env)

    if mode == "smoke":
        print("\n  Skipping [F] association_rules and [G] benchmark_scale - both require")
        print("  TAXI_MONTHS=full and refuse the 3-month sample by design.")
    else:
        run_step("F FP-Growth association rules - pickup to dropoff flows",
                 [py, "-m", "src.batch.association_rules"], env)
        run_step("G scaling benchmark - wall-clock vs input size",
                 [py, "-m", "src.batch.benchmark_scale"], env)

    banner(f"PIPELINE COMPLETE - {label}  (K={chosen_k})")
    print("  data/processed/  demand.parquet, features.parquet, modeling_zones.parquet")
    print("  models/          kmeans/, zone_clusters.parquet, cluster_demand.parquet,")
    print("                   kmeans_metadata.json, conditions_model.json, hist_avg.parquet")
    print("  reports/         figures, maps, geojson/, metrics CSVs")
    print("\n  Streaming demo:  python run.py demo")


# --------------------------------------------------------------------------- #
# demo — mirrors run_demo.bat
# --------------------------------------------------------------------------- #
def cmd_demo(args: argparse.Namespace) -> None:
    py = venv_python(args.python)
    java_home = require_java_home()

    banner("Kafka + Spark streaming demo")
    print(f"  replay      {DEMO_START}  for {DEMO_HOURS} simulated hours")
    print(f"  speedup     {SPEEDUP}x  (1 real second = 10 simulated minutes)")
    print(f"  python      {py}")
    print(f"  JAVA_HOME   {java_home}")

    # Same prerequisite list as the .bat: models/ is committed but data/ is not, so
    # a fresh clone passes the model check and still has nothing to stream.
    demo_month = DEMO_START[:7]
    required = [
        ROOT / "models" / "cluster_demand.parquet",
        ROOT / "data" / "raw" / f"yellow_tripdata_{demo_month}.parquet",
        ROOT / "data" / "processed" / "demand.parquet",
        ROOT / "data" / "processed" / "modeling_zones.parquet",
        ROOT / "data" / "external" / "events.csv",
        ROOT / "models" / "conditions_model.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing and not DRY_RUN:
        fail("the pipeline has not fully run in this clone; missing:\n    "
             + "\n    ".join(str(p.relative_to(ROOT)) for p in missing)
             + "\n  Minimum for this demo:  python run.py pipeline smoke")

    env = dict(os.environ)
    exports = {"JAVA_HOME": java_home}
    if "TAXI_MONTHS" in env:
        exports["TAXI_MONTHS"] = env["TAXI_MONTHS"]

    print("\n[1/5] Starting Kafka...")
    if not DRY_RUN:
        if subprocess.run(["docker", "compose", "up", "-d"], cwd=ROOT).returncode != 0:
            fail("docker compose failed. Is Docker Desktop running?")

    print("[2/5] Waiting for the broker to report healthy...")
    if not DRY_RUN:
        wait_until("broker health",
                   lambda: docker_inspect("taxi-kafka", "{{.State.Health.Status}}") == "healthy",
                   120, "Check: docker compose logs kafka")
        print("       broker healthy. Waiting for taxi-kafka-init to finish...")
        # `compose up -d` returns while the one-shot init container is still creating
        # the topic; resetting before it exits would race with that create.
        wait_until("topic init",
                   lambda: docker_inspect("taxi-kafka-init", "{{.State.Status}}") in ("", "exited"),
                   60, "Check: docker compose logs kafka-init")
        print("       topic init done.")

    # Reset BEFORE any consumer attaches: deleting a subscribed topic faults the query.
    run_step("3/5 reset the topic to a clean slate",
             [py, "-m", "src.stream.producer", "--reset-only"], env)
    if READY_FILE.exists():
        READY_FILE.unlink()

    print("\n[4/5] Launching the Spark consumer in a new window...")
    open_in_terminal(
        "Taxi demand - SPARK STREAM (consumer)",
        [py, "-m", "src.stream.spark_stream", "--fresh",
         "--run-seconds", str(STREAM_SECONDS), "--ready-file", str(READY_FILE),
         "--validate"],
        env, exports,
    )
    print("       waiting for the query to start (first run downloads the connector)...")
    if not DRY_RUN:
        wait_until("consumer ready", READY_FILE.exists, READY_TIMEOUT,
                   "Read the consumer window for the error.")
        print("       consumer is running - safe to produce.")

    print("\n[5/5] Launching the producer in a new window...")
    open_in_terminal(
        "Taxi demand - KAFKA PRODUCER",
        [py, "-m", "src.stream.producer", "--start", DEMO_START,
         "--hours", str(DEMO_HOURS), "--speedup", str(SPEEDUP), "--append"],
        env, exports,
    )

    banner("Demo running in two windows")
    print("  PRODUCER  emits trips paced by their own event time.")
    print("  STREAM    prints each closed (zone, window) cell with predicted vs actual,")
    print(f"            validates against demand.parquet, and stops after {STREAM_SECONDS}s.")
    print("  With a 2-hour watermark the last ~2 simulated hours never close.")
    print("\n  Single-query forecaster (instant, weather/events-aware):")
    print(f'    {py} -m src.stream.predict_live --zone 161 --at "2024-11-28 15:00"')
    print(f'    {py} -m src.stream.predict_live --zone 79 --at "2024-01-13 01:00" --precip 5')


# --------------------------------------------------------------------------- #
# gui — mirrors run_gui.bat
# --------------------------------------------------------------------------- #
def cmd_gui(args: argparse.Namespace) -> None:
    mode = (args.mode[0].lower() if args.mode else "all")
    if mode not in ("all", "serve", "build"):
        fail(f"unknown gui mode {mode!r} — use: serve | build (or nothing for both)")

    banner("NYC Demand Console - local web GUI")
    gui_dir = ROOT / "gui"

    if mode in ("all", "build"):
        py = venv_python(args.python)
        run_step("1/2 export the model into gui/payload.json",
                 [py, "-m", "src.viz.build_gui"], dict(os.environ))
        if mode == "build":
            print("\nBuilt. Open gui/standalone.html directly, or run:  python run.py gui serve")
            return

    if not (gui_dir / "payload.json").exists() and not DRY_RUN:
        fail("gui/payload.json is missing - run  python run.py gui  (no arguments).")

    port = args.port
    url = f"http://127.0.0.1:{port}/"  # not "localhost": browsers may try ::1 first
    print(f"\n[2/2] Serving gui/ at {url}")
    print("  The console opens in your default browser. Leave this running while")
    print("  presenting; press Ctrl+C to stop. No server wanted? gui/standalone.html")
    print("  holds the same console in one file.")
    if DRY_RUN:
        print(f"   $ {sys.executable} -m http.server {port} --bind 127.0.0.1 --directory gui")
        return

    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    # --bind 127.0.0.1: a presentation tool has no business on the room's network.
    try:
        subprocess.run(
            [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
             "--directory", str(gui_dir)],
            cwd=ROOT, check=False,
        )
    except KeyboardInterrupt:
        print("\nstopped.")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    global DRY_RUN
    parser = argparse.ArgumentParser(
        description="Cross-platform launcher for the taxi demand project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    # Shared flags live on a parent so they work before OR after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true",
                        help="print every command instead of running it")
    common.add_argument("--python", help="interpreter to use instead of .venv's")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pipeline", parents=[common],
                       help="batch pipeline: sweep | smoke | k <N>")
    p.add_argument("mode", nargs="*")
    p.set_defaults(func=cmd_pipeline)

    d = sub.add_parser("demo", parents=[common],
                       help="Kafka + Spark Structured Streaming demo")
    d.set_defaults(func=cmd_demo)

    g = sub.add_parser("gui", parents=[common],
                       help="local web console: serve | build | (both)")
    g.add_argument("mode", nargs="*")
    g.add_argument("--port", type=int, default=GUI_PORT)
    g.set_defaults(func=cmd_gui)

    args = parser.parse_args(argv)
    DRY_RUN = args.dry_run
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
