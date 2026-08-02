@echo off
REM Runs the whole batch pipeline for full-year 2024, in dependency order, halting on
REM the first failure. Prerequisite: the .venv exists and JAVA_HOME points at JDK 17.
REM Kafka/Docker are NOT needed - this is the batch half only.
REM Pass "smoke" as the first argument to run the identical order over the 3-month
REM sample instead, to verify sequencing before committing to the 12-month run.

setlocal
cd /d "%~dp0"

if /i "%~1"=="smoke" (
    set "TAXI_MONTHS=sample"
    set "RUN_LABEL=SMOKE TEST - 3-month sample (2024-01..03)"
    set "DOWNLOAD_ARGS="
) else (
    set "TAXI_MONTHS=full"
    set "RUN_LABEL=FULL YEAR - 12 months of 2024"
    REM --yes confirms downloading more than the 3-month sample.
    set "DOWNLOAD_ARGS=--yes"
)

echo ============================================================================
echo  Taxi demand pipeline - %RUN_LABEL%
echo ============================================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: .venv not found. Create it first:
    echo     py -3.11 -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    exit /b 1
)
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :fail

if "%JAVA_HOME%"=="" (
    echo ERROR: JAVA_HOME is not set. Spark 3.5.3 needs a Java 17 JDK.
    exit /b 1
)
echo   JAVA_HOME    %JAVA_HOME%
echo   TAXI_MONTHS  %TAXI_MONTHS%
echo.

set "STEP=0"

echo [1/10] Download TLC parquet + zone lookup + shapefile
"%VENV_PY%" -m src.ingest.download_tlc %DOWNLOAD_ARGS%
if errorlevel 1 (set "STEP=1 download_tlc" & goto :fail)

echo.
echo [2/10] Fetch Open-Meteo hourly weather per zone centroid
"%VENV_PY%" -m src.ingest.fetch_weather
if errorlevel 1 (set "STEP=2 fetch_weather" & goto :fail)

echo.
echo [3/10] Build holiday + event calendar
"%VENV_PY%" -m src.ingest.build_events
if errorlevel 1 (set "STEP=3 build_events" & goto :fail)

echo.
echo [4/10] Clean trips and aggregate to hourly demand per zone
"%VENV_PY%" -m src.batch.clean_aggregate
if errorlevel 1 (set "STEP=4 clean_aggregate" & goto :fail)

echo.
echo [5/10] Apply the zone exclusion policy
"%VENV_PY%" -m src.batch.zone_policy
if errorlevel 1 (set "STEP=5 zone_policy" & goto :fail)

echo.
echo [6/10] Sedona geo join - centroids and polygons
"%VENV_PY%" -m src.batch.geo_join
if errorlevel 1 (set "STEP=6 geo_join" & goto :fail)

echo.
echo [7/10] Build the modelling feature table
"%VENV_PY%" -m src.batch.features
if errorlevel 1 (set "STEP=7 features" & goto :fail)

echo.
echo [8/10] Train K-Means  (K=4 was chosen from the Q1 analysis - re-inspect the
echo        elbow/silhouette output before trusting it for the full year)
"%VENV_PY%" -m src.batch.train_kmeans --k 4
if errorlevel 1 (set "STEP=8 train_kmeans" & goto :fail)

echo.
echo [9/10] Score the baseline ladder on the held-out split
"%VENV_PY%" -m src.batch.evaluate
if errorlevel 1 (set "STEP=9 evaluate" & goto :fail)

echo.
echo [10/10] Render report figures, map and GeoJSON
"%VENV_PY%" -m src.viz.make_maps
if errorlevel 1 (set "STEP=10 make_maps" & goto :fail)

echo.
echo ============================================================================
echo  PIPELINE COMPLETE - %RUN_LABEL%
echo ============================================================================
echo   data\processed\  demand.parquet, features.parquet, modeling_zones.parquet
echo   models\          kmeans\, zone_clusters.parquet, cluster_demand.parquet
echo   reports\         k_selection.png, zone_hour_heatmap.png, cluster_map.html,
echo                    geojson\, k_sweep.csv, baseline_metrics.csv
echo.
echo   Streaming demo:  run_demo.bat
endlocal
exit /b 0

:fail
echo.
echo ############################################################################
echo  PIPELINE FAILED at step %STEP%  (exit code %errorlevel%)
echo  Nothing after this step was run. Fix the error above and re-run -
echo  completed steps are idempotent and will be skipped or overwritten safely.
echo ############################################################################
endlocal
exit /b 1
