#!/usr/bin/env python3
"""Cross-platform launcher — the three ``run_*.bat`` files, for Windows AND macOS/Linux.

Standard library only, so it runs under any Python 3.9+; every pipeline step is
then executed with the project's ``.venv`` interpreter (``.venv\\Scripts\\python.exe``
on Windows, ``.venv/bin/python`` elsewhere — override with ``VENV_PY=<path>`` or
``--python``).

    python run.py                     # no arguments: SET UP (create .venv with the
                                      #   right Python, install requirements, find
                                      #   Java, start Docker) then START the console
                                      #   and the streaming demo
    python run.py pipeline            # steps 1-7, then sweep K and stop for review
    python run.py pipeline k 5        # commit to K=5: train, evaluate, ablation, maps, ...
    python run.py pipeline smoke      # 3-month sample, K pinned to 4 (sequencing check)
    python run.py demo                # Kafka + Spark Structured Streaming, two windows
                                      #   + the live page on http://127.0.0.1:8765/stream.html
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
import glob
import hashlib
import os
import platform
import shlex
import shutil
import socket
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
# 60x: 1 real second = 1 simulated minute, so a window closes about once a
# minute and a 32-hour replay (a whole day plus the 2 h the watermark holds back)
# runs ~32 minutes — long enough to talk over, with the morning and evening
# rushes both crossing the map.
SPEEDUP = 60
DEMO_START = "2024-01-10 06:00:00"
DEMO_HOURS = 32
STREAM_SECONDS = DEMO_HOURS * 3600 // SPEEDUP + 60   # replay time + a minute's margin
GUI_PORT = 8765
STREAM_STATE = ROOT / "gui" / "stream_state.json"   # rewritten by the consumer per batch
STREAM_PAGE = "stream.html"

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


def find_python311() -> str | None:
    """A CPython 3.11 to create .venv with — the newest Python PySpark 3.5 supports."""
    if sys.version_info[:2] == (3, 11):
        return sys.executable
    if IS_WINDOWS:
        try:
            out = subprocess.run(["py", "-3.11", "-c", "import sys; print(sys.executable)"],
                                 capture_output=True, text=True, check=False)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except FileNotFoundError:
            pass
        return None
    for candidate in (shutil.which("python3.11"),
                      "/opt/homebrew/bin/python3.11",   # Homebrew, Apple silicon
                      "/usr/local/bin/python3.11",      # Homebrew, Intel
                      "/usr/bin/python3.11"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def ensure_requirements(py: str) -> None:
    """Install requirements.txt into .venv, once per change to the file (a hash
    stamp inside .venv makes every later start cost one file read)."""
    req = ROOT / "requirements.txt"
    stamp = ROOT / ".venv" / ".requirements.stamp"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    if stamp.exists() and stamp.read_text().strip() == digest:
        return
    print("\n[setup] installing requirements into .venv (first time takes a few minutes)...")
    if DRY_RUN:
        return
    if subprocess.run([py, "-m", "pip", "install", "-r", str(req)], cwd=ROOT).returncode != 0:
        fail("pip install -r requirements.txt failed — read the pip output above.")
    stamp.write_text(digest)


def venv_python(override: str | None) -> str:
    """The project's interpreter, by full path — CREATED on the spot when missing.

    ``--python``/``VENV_PY`` overrides are honoured verbatim. Otherwise, if .venv
    lacks this platform's interpreter (fresh clone, or a venv carried over from
    the other OS — this repo moves between Windows and macOS), it is built with
    Python 3.11 and requirements are installed, so any command starts from
    nothing with no manual setup on either OS.
    """
    candidate = override or os.environ.get("VENV_PY")
    if candidate:
        if Path(candidate).exists() or DRY_RUN:
            return candidate
        fail(f"interpreter not found at {candidate} (from --python / VENV_PY).")

    expected = ROOT / ".venv" / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
    if not expected.exists():
        venv_dir = ROOT / ".venv"
        if venv_dir.exists():
            # No interpreter for THIS platform => it is the other OS's venv.
            # Set it aside rather than delete it.
            bak, n = ROOT / ".venv-foreign-bak", 1
            while bak.exists():
                n += 1
                bak = ROOT / f".venv-foreign-bak{n}"
            print(f"[setup] .venv has no {platform.system()} interpreter — moving it to {bak.name}/")
            if not DRY_RUN:
                venv_dir.rename(bak)
        py311 = find_python311()
        if py311 is None:
            install = ("winget install -e --id Python.Python.3.11" if IS_WINDOWS
                       else "brew install python@3.11")
            fail("Python 3.11 not found (PySpark 3.5.3 supports up to 3.11). Install it:\n"
                 f"    {install}\n  then re-run this script.")
        print(f"[setup] creating .venv with {py311} ...")
        if not DRY_RUN and subprocess.run([py311, "-m", "venv", str(venv_dir)],
                                          cwd=ROOT).returncode != 0:
            fail("could not create .venv — read the output above.")
    if Path(expected).exists() or not DRY_RUN:
        ensure_requirements(str(expected))
    return str(expected)


def require_java_home() -> str:
    java_home = os.environ.get("JAVA_HOME", "").strip()
    if java_home:
        return java_home
    if IS_MACOS:
        # Unset is the normal state on a Mac: ask the system for a JDK 17, then try
        # the Homebrew keg, so the demo is one command here too.
        try:
            found = subprocess.run(["/usr/libexec/java_home", "-v", "17"],
                                   capture_output=True, text=True, check=False)
            if found.returncode == 0 and found.stdout.strip():
                os.environ["JAVA_HOME"] = found.stdout.strip()
                return os.environ["JAVA_HOME"]
        except FileNotFoundError:
            pass
        brew = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"
        if Path(brew, "bin", "java").exists():
            os.environ["JAVA_HOME"] = brew
            return brew
    if IS_WINDOWS:
        # Same one-command promise on Windows: pick up a JDK 17 from the usual
        # vendors' default install locations.
        for pattern in (r"C:\Program Files\Eclipse Adoptium\jdk-17*",
                        r"C:\Program Files\Microsoft\jdk-17*",
                        r"C:\Program Files\Java\jdk-17*",
                        r"C:\Program Files\Amazon Corretto\jdk17*"):
            hits = sorted(glob.glob(pattern))
            if hits and Path(hits[-1], "bin", "java.exe").exists():
                os.environ["JAVA_HOME"] = hits[-1]
                return hits[-1]
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


def docker_daemon_up() -> bool:
    return subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


def ensure_docker_running() -> None:
    """Docker installed AND its daemon answering — start Docker Desktop if not."""
    if shutil.which("docker") is None:
        fail("docker is not installed or not on PATH. Install Docker Desktop:\n"
             "    https://www.docker.com/products/docker-desktop/")
    if DRY_RUN or docker_daemon_up():
        return
    print("       Docker daemon not running - starting Docker Desktop...")
    if IS_MACOS:
        subprocess.run(["open", "-g", "-a", "Docker"], check=False)
    elif IS_WINDOWS:
        exe = Path(r"C:\Program Files\Docker\Docker\Docker Desktop.exe")
        if exe.exists():
            subprocess.Popen([str(exe)])
        else:
            fail("Docker Desktop not found at its default path - start it by hand, then re-run.")
    wait_until("Docker daemon", docker_daemon_up, 180,
               "Start Docker Desktop by hand, then re-run.")


def port_in_use(port: int) -> bool:
    """True when something already listens on 127.0.0.1:port (e.g. ``run.py gui``)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def serve_gui(port: int) -> "subprocess.Popen | None":
    """Start ``http.server`` on gui/ in the background, or reuse a server already there.

    Returns the child process, or None when the port was already taken (then the
    caller just opens the browser against whatever is serving). Request logging
    goes to /dev/null: the live page polls every two seconds, and that would
    otherwise bury the launcher's own output.
    """
    if port_in_use(port):
        print(f"       port {port} is already serving - reusing it (run.py gui?)")
        return None
    # --bind 127.0.0.1: a presentation tool has no business on the room's network.
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1",
         "--directory", str(ROOT / "gui")],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


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
    print(f"  speedup     {SPEEDUP}x  (1 real second = 1 simulated minute; "
          f"~{STREAM_SECONDS // 60} min in all)")
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
        ensure_docker_running()
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
    run_step("3/6 reset the topic to a clean slate",
             [py, "-m", "src.stream.producer", "--reset-only"], env)
    if READY_FILE.exists():
        READY_FILE.unlink()
    # A snapshot left by the previous run would show on the page until the first
    # batch of this one lands (the consumer's --fresh clears it too).
    if STREAM_STATE.exists():
        STREAM_STATE.unlink()

    print("\n[4/6] Launching the Spark consumer in a new window...")
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

    print("\n[5/6] Launching the producer in a new window...")
    open_in_terminal(
        "Taxi demand - KAFKA PRODUCER",
        [py, "-m", "src.stream.producer", "--start", DEMO_START,
         "--hours", str(DEMO_HOURS), "--speedup", str(SPEEDUP), "--append"],
        env, exports,
    )

    url = f"http://127.0.0.1:{args.port}/{STREAM_PAGE}"
    print(f"\n[6/6] Serving the live page at {url}")
    server = None
    if DRY_RUN:
        print(f"   $ {sys.executable} -m http.server {args.port} --bind 127.0.0.1 --directory gui")
    else:
        server = serve_gui(args.port)
        if server is not None:
            wait_until("live page server", lambda: port_in_use(args.port), 15,
                       "http.server did not come up.")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    banner("Demo running in two windows + the live page")
    print("  PRODUCER  emits trips paced by their own event time.")
    print("  STREAM    prints each closed (zone, window) cell with predicted vs actual,")
    print(f"            validates against demand.parquet, and stops after {STREAM_SECONDS}s.")
    print(f"  PAGE      {url} - the choropleth, feed and tiles refresh")
    print("            from gui/stream_state.json every 2 s; the verdict appears at the end.")
    print("  With a 2-hour watermark the last ~2 simulated hours never close.")
    print("\n  Single-query forecaster (instant, weather/events-aware):")
    print(f'    {py} -m src.stream.predict_live --zone 161 --at "2024-11-28 15:00"')
    print(f'    {py} -m src.stream.predict_live --zone 79 --at "2024-01-13 01:00" --precip 5')

    if server is None:
        return
    # Keep serving so the page (and its final verdict) stays up after the consumer
    # finishes; the demo is over when the presenter says so.
    print("\n  Leave this window open while presenting; press Ctrl+C to stop the page server.")
    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
        print("\nstopped.")


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
# start — the no-argument default: set up, then start everything
# --------------------------------------------------------------------------- #
def cmd_start(args: argparse.Namespace) -> None:
    banner("START - set up, then console + streaming demo")
    py = venv_python(args.python)   # creates .venv + installs requirements if needed

    run_step("1/3 export the model into gui/payload.json",
             [py, "-m", "src.viz.build_gui"], dict(os.environ))

    url = f"http://127.0.0.1:{args.port}/"
    print(f"\n[2/3] Serving the console at {url}")
    server = serve_gui(args.port)
    if not DRY_RUN:
        if server is not None:
            wait_until("console server", lambda: port_in_use(args.port), 15,
                       "http.server did not come up.")
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    print("\n[3/3] Streaming demo (Kafka + Spark)...")
    try:
        cmd_demo(args)   # reuses the server above and opens the live stream page
    except SystemExit:
        print("\nThe streaming demo could not start — the reason and its fix are")
        print(f"printed above. The console stays up at {url}")
        print("Fixed it? Run:  python run.py demo")

    if server is None or DRY_RUN:
        return
    print("\n  Press Ctrl+C here to stop the console server when you are done.")
    try:
        server.wait()
    except KeyboardInterrupt:
        server.terminate()
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

    s = sub.add_parser("start", parents=[common],
                       help="set up (.venv, deps, Java, Docker) and start console + demo "
                            "— what plain `python run.py` does")
    s.add_argument("--port", type=int, default=GUI_PORT)
    s.set_defaults(func=cmd_start)

    p = sub.add_parser("pipeline", parents=[common],
                       help="batch pipeline: sweep | smoke | k <N>")
    p.add_argument("mode", nargs="*")
    p.set_defaults(func=cmd_pipeline)

    d = sub.add_parser("demo", parents=[common],
                       help="Kafka + Spark Structured Streaming demo + live page")
    d.add_argument("--port", type=int, default=GUI_PORT,
                   help="port for the live page (default %(default)s)")
    d.set_defaults(func=cmd_demo)

    g = sub.add_parser("gui", parents=[common],
                       help="local web console: serve | build | (both)")
    g.add_argument("mode", nargs="*")
    g.add_argument("--port", type=int, default=GUI_PORT)
    g.set_defaults(func=cmd_gui)

    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        argv = ["start"]   # zero-argument run: set up and start everything
    args = parser.parse_args(argv)
    DRY_RUN = args.dry_run
    os.chdir(ROOT)
    args.func(args)


if __name__ == "__main__":
    main()
