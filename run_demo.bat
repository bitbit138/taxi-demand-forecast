@echo off
REM Two-terminal Kafka streaming demo. Starts the Spark consumer FIRST, waits until
REM its query is actually running, then starts the producer - so the stream is seen
REM filling live rather than draining a backlog.
REM Prerequisite: Docker Desktop running, .venv built, JAVA_HOME on JDK 17,
REM and the batch model trained (run_pipeline_full_year.bat, or the Q1 steps).

setlocal
cd /d "%~dp0"

set "KAFKA_PKG=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3"
set "READY=.stream_ready"
set "READY_TIMEOUT=180"
REM Demo pacing: 600 => 1 real second = 10 simulated minutes. Above ~800 the
REM producer becomes throughput-bound and stops pacing (see config.REPLAY_SPEEDUP).
set "SPEEDUP=600"
set "DEMO_START=2024-01-10 14:00:00"
set "DEMO_HOURS=6"
set "STREAM_SECONDS=420"

echo ============================================================================
echo  Kafka + Spark streaming demo
echo ============================================================================
echo   replay      %DEMO_START%  for %DEMO_HOURS% simulated hours
echo   speedup     %SPEEDUP%x  (1 real second = 10 simulated minutes)
echo   connector   %KAFKA_PKG%
echo.

REM ---- prerequisites -------------------------------------------------------
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv not found - see the README setup section.
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :fail

if "%JAVA_HOME%"=="" (
    echo ERROR: JAVA_HOME is not set. Spark 3.5.3 needs a Java 17 JDK.
    exit /b 1
)
if not exist "models\cluster_demand.parquet" (
    echo ERROR: no trained model found at models\cluster_demand.parquet
    echo        Run run_pipeline_full_year.bat smoke  first.
    exit /b 1
)

REM models\ is committed but data\ is gitignored, so a fresh clone passes the model
REM check and still has nothing to stream. Without this the consumer would exit on
REM the first missing parquet in its own window, and this script would sit out the
REM full READY_TIMEOUT before reporting a timeout instead of the real cause.
REM data\external\events.csv and models\conditions_model.json are checked because
REM this script ADVERTISES the weather/events forecaster at the end, including the
REM "Thanksgiving is flagged automatically from events.csv" query. Without events.csv
REM predict_live still answers, but every calendar flag reads False and the event
REM adjustment silently disappears - the demo appears to work while showing the wrong
REM number, which is worse than refusing to start.
set "DEMO_MONTH=%DEMO_START:~0,7%"
for %%P in (
    "data\raw\yellow_tripdata_%DEMO_MONTH%.parquet"
    "data\processed\demand.parquet"
    "data\processed\modeling_zones.parquet"
    "data\external\events.csv"
    "models\conditions_model.json"
) do (
    if not exist "%%~P" (
        echo ERROR: %%~P is missing - the pipeline has not fully run in this clone.
        echo        Minimum for this demo ^(3-month sample; leaves models\ untouched^):
        echo            .venv\Scripts\python.exe -m src.ingest.download_tlc
        echo            .venv\Scripts\python.exe -m src.ingest.build_events
        echo            .venv\Scripts\python.exe -m src.batch.clean_aggregate
        echo            .venv\Scripts\python.exe -m src.batch.zone_policy
        echo        events.csv alone:   -m src.ingest.build_events   ^(seconds, no network^)
        echo        conditions model:   -m src.batch.ablation        ^(needs features.parquet^)
        echo        Or the whole pipeline:  run_pipeline_full_year.bat smoke
        exit /b 1
    )
)

REM HADOOP_HOME must be set BEFORE spark-submit: it launches the JVM itself, so
REM config.configure_hadoop_home() - which sets it from inside the Python driver -
REM runs too late and Spark dies with "HADOOP_HOME and hadoop.home.dir are unset".
REM That is why `python -m src.stream.spark_stream` works without this and
REM spark-submit does not.
set "HADOOP_HOME=%CD%\hadoop"
set "PATH=%HADOOP_HOME%\bin;%PATH%"

REM Call the venv's executables by full path. A Microsoft Store Python 3.12 on this
REM machine also ships pyspark, and its spark-submit resolves a different SPARK_HOME.
set "VENV_PY=%CD%\.venv\Scripts\python.exe"
set "SPARK_SUBMIT=%CD%\.venv\Scripts\spark-submit.cmd"

REM spark-submit's .cmd wrapper expands %PYSPARK_DRIVER_PYTHON% unquoted, so a path
REM with spaces makes it print a harmless but alarming "The system cannot
REM find the path specified." before Spark starts. Hand it the 8.3 short name instead:
REM same interpreter, no spaces, no message. On a volume with 8.3 names disabled the
REM expansion returns the long path and the (still harmless) message comes back.
for %%I in ("%VENV_PY%") do set "VENV_PY_SHORT=%%~sI"
if not defined VENV_PY_SHORT set "VENV_PY_SHORT=%VENV_PY%"
set "PYSPARK_PYTHON=%VENV_PY_SHORT%"
set "PYSPARK_DRIVER_PYTHON=%VENV_PY_SHORT%"

REM SPARK_HOME must be set too. PySpark's find-spark-home.cmd shells out to the
REM Python executable without quoting the path, so a repo path containing spaces
REM (e.g. "My Projects") makes it report "Missing Python executable" and then
REM "Failed to find Spark jars directory". Pointing at the venv's pyspark package
REM skips the discovery entirely.
set "SPARK_HOME=%CD%\.venv\Lib\site-packages\pyspark"

echo [1/5] Starting Kafka...
docker compose up -d
if errorlevel 1 (
    echo ERROR: docker compose failed. Is Docker Desktop running?
    goto :fail
)

echo [2/5] Waiting for the broker to report healthy...
set /a WAITED=0
:health
for /f "tokens=*" %%H in ('docker inspect -f "{{.State.Health.Status}}" taxi-kafka 2^>nul') do set "HEALTH=%%H"
if /i "%HEALTH%"=="healthy" goto :healthy
if %WAITED% GEQ 120 (
    echo ERROR: broker not healthy after 120s. Check: docker compose logs kafka
    goto :fail
)
REM `ping` not `timeout`: timeout /t aborts with "Input redirection is not
REM supported" whenever stdin is not a real console, which silently turns the
REM wait loop into a busy spin.
ping -n 4 127.0.0.1 >nul 2>&1
set /a WAITED+=3
echo        ... %HEALTH% (%WAITED%s)
goto :health
:healthy
echo        broker healthy.

REM `docker compose up -d` returns as soon as taxi-kafka-init has STARTED, and that
REM one-shot container runs `kafka-topics.sh --create --if-not-exists taxi-trips` a
REM few seconds later. If that create lands after step [3/5]'s delete, the topic
REM reappears immediately and the reset reports "still present after 30s" - the topic
REM was deleted, then recreated behind it. Wait for the container to exit first.
echo        waiting for taxi-kafka-init to finish...
set /a WAITED=0
:initwait
set "INIT_STATE="
for /f "tokens=*" %%S in ('docker inspect -f "{{.State.Status}}" taxi-kafka-init 2^>nul') do set "INIT_STATE=%%S"
REM No such container (e.g. `docker compose up -d kafka` only): nothing to wait for.
if "%INIT_STATE%"=="" goto :initdone
if /i "%INIT_STATE%"=="exited" goto :initdone
if %WAITED% GEQ 60 (
    echo ERROR: taxi-kafka-init still %INIT_STATE% after 60s.
    echo        Check: docker compose logs kafka-init
    goto :fail
)
ping -n 4 127.0.0.1 >nul 2>&1
set /a WAITED+=3
echo        ... %INIT_STATE% (%WAITED%s)
goto :initwait
:initdone
echo        topic init done.

REM Reset BEFORE any consumer attaches: deleting a topic that a running query is
REM subscribed to can fault the query on a vanished topic.
echo [3/5] Resetting the topic to a clean slate...
"%VENV_PY%" -m src.stream.producer --reset-only
if errorlevel 1 goto :fail

if exist "%READY%" del "%READY%"

echo [4/5] Launching the Spark consumer in a new window...
start "Taxi demand - SPARK STREAM (consumer)" cmd /k ""%SPARK_SUBMIT%" --packages %KAFKA_PKG% src\stream\spark_stream.py --fresh --run-seconds %STREAM_SECONDS% --ready-file %READY% --validate"

echo        waiting for the query to start (first run downloads the connector)...
set /a WAITED=0
:waitready
if exist "%READY%" goto :streamready
if %WAITED% GEQ %READY_TIMEOUT% (
    echo ERROR: consumer not ready after %READY_TIMEOUT%s. Check its window.
    goto :fail
)
ping -n 4 127.0.0.1 >nul 2>&1
set /a WAITED+=3
echo        ... %WAITED%s
goto :waitready
:streamready
echo        consumer is running - safe to produce.

echo [5/5] Launching the producer in a new window...
start "Taxi demand - KAFKA PRODUCER" cmd /k ""%VENV_PY%" -m src.stream.producer --start "%DEMO_START%" --hours %DEMO_HOURS% --speedup %SPEEDUP% --append"

echo.
echo ============================================================================
echo  Demo running in two windows.
echo ============================================================================
echo   PRODUCER  emits trips paced by their own event time.
echo   STREAM    prints each closed (zone, window) cell with predicted vs actual,
echo             then validates streamed totals against demand.parquet and stops
echo             after %STREAM_SECONDS%s.
echo.
echo   With a 2-hour watermark the last ~2 simulated hours never close, so a
echo   %DEMO_HOURS%-hour replay yields ~%DEMO_HOURS% minus 3 closed windows.
echo   Output lands in data\processed\stream_predictions\.
echo.
echo   Single-query forecaster (same artifacts, instant, weather/events-aware):
echo     %VENV_PY% -m src.stream.predict_live --zone 161 --at "2024-11-28 15:00"
echo         (Thanksgiving is flagged automatically from events.csv)
echo     %VENV_PY% -m src.stream.predict_live --zone 79 --at "2024-01-13 01:00" --precip 5
echo         (5 mm of rain shifts the forecast by the fitted coefficient)
echo.
endlocal
exit /b 0

:fail
echo.
echo ############################################################################
echo  DEMO SETUP FAILED (exit code %errorlevel%).
echo  If the consumer never became ready, its window is still open - read the
echo  error there. Close any stray windows before re-running.
echo ############################################################################
endlocal
exit /b 1
