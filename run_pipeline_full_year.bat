@echo off
REM Batch pipeline for full-year 2024, in dependency order, halting on first failure.
REM Prerequisite: the .venv exists and JAVA_HOME points at JDK 17. Docker not needed.
REM
REM   run_pipeline_full_year.bat          steps 1-7, then SWEEP K and stop for review
REM   run_pipeline_full_year.bat k 6      commit to K=6: train, evaluate, render
REM   run_pipeline_full_year.bat smoke    same order over the 3-month sample, K pinned
REM                                       to 4 - a sequencing check, not a model choice

setlocal
cd /d "%~dp0"

set "MODE=sweep"
set "CHOSEN_K="
if /i "%~1"=="smoke" set "MODE=smoke"
if /i "%~1"=="k" (
    set "MODE=finish"
    set "CHOSEN_K=%~2"
)

if "%MODE%"=="finish" if "%CHOSEN_K%"=="" (
    echo ERROR: "k" mode needs a value, e.g.  run_pipeline_full_year.bat k 6
    exit /b 1
)

if "%MODE%"=="smoke" (
    set "TAXI_MONTHS=sample"
    set "RUN_LABEL=SMOKE TEST - 3-month sample, K pinned to 4"
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

REM Call the venv interpreter by full path. A Microsoft Store Python 3.12 on this
REM machine also has pyspark installed, and PATH shadowing would silently run the
REM wrong one.
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if "%JAVA_HOME%"=="" (
    echo ERROR: JAVA_HOME is not set. Spark 3.5.3 needs a Java 17 JDK.
    exit /b 1
)
echo   JAVA_HOME    %JAVA_HOME%
echo   TAXI_MONTHS  %TAXI_MONTHS%
echo   mode         %MODE% %CHOSEN_K%
echo.

set "STEP=0"
if "%MODE%"=="finish" goto :modelling

echo [1/8] Download TLC parquet + zone lookup + shapefile
"%VENV_PY%" -m src.ingest.download_tlc %DOWNLOAD_ARGS%
if errorlevel 1 (set "STEP=1 download_tlc" & goto :fail)

echo.
echo [2/8] Fetch Open-Meteo hourly weather per zone centroid
"%VENV_PY%" -m src.ingest.fetch_weather
if errorlevel 1 (set "STEP=2 fetch_weather" & goto :fail)

echo.
echo [3/8] Build holiday + event calendar
"%VENV_PY%" -m src.ingest.build_events
if errorlevel 1 (set "STEP=3 build_events" & goto :fail)

echo.
echo [4/8] Clean trips and aggregate to hourly demand per zone
"%VENV_PY%" -m src.batch.clean_aggregate
if errorlevel 1 (set "STEP=4 clean_aggregate" & goto :fail)

echo.
echo [5/8] Apply the zone exclusion policy
"%VENV_PY%" -m src.batch.zone_policy
if errorlevel 1 (set "STEP=5 zone_policy" & goto :fail)

echo.
echo [6/8] Sedona geo join - centroids and polygons
"%VENV_PY%" -m src.batch.geo_join
if errorlevel 1 (set "STEP=6 geo_join" & goto :fail)

echo.
echo [7/8] Build the modelling feature table
"%VENV_PY%" -m src.batch.features
if errorlevel 1 (set "STEP=7 features" & goto :fail)

if "%MODE%"=="smoke" goto :smokemodel

echo.
echo [8/8] Sweep K on the full-year data and describe the candidates
echo        K is NOT pinned: a full year adds seasonal structure the Q1 sample
echo        could not show, so the choice is re-made on this data.
"%VENV_PY%" -m src.batch.train_kmeans --inspect --candidates 3
if errorlevel 1 (set "STEP=8 train_kmeans --inspect" & goto :fail)

echo.
echo ============================================================================
echo  SWEEP COMPLETE - REVIEW REQUIRED, nothing was saved
echo ============================================================================
echo   Read the cluster characters printed above and the curves in
echo   reports\k_sweep.csv, then commit to a K:
echo.
echo       run_pipeline_full_year.bat k 4      (if 4 still holds)
echo       run_pipeline_full_year.bat k ^<N^>     (if the full year splits differently)
echo.
echo   That finishes the run: train + save, evaluate, and render the figures.
echo ============================================================================
endlocal
exit /b 0

:smokemodel
echo.
echo [8/8] Train K-Means, evaluate, render  (smoke: K pinned to 4)
set "CHOSEN_K=4"

:modelling
echo.
echo [A] Train K-Means with K=%CHOSEN_K% and save the model
"%VENV_PY%" -m src.batch.train_kmeans --k %CHOSEN_K%
if errorlevel 1 (set "STEP=A train_kmeans --k %CHOSEN_K%" & goto :fail)

echo.
echo [B] Score the baseline ladder on the held-out split
"%VENV_PY%" -m src.batch.evaluate
if errorlevel 1 (set "STEP=B evaluate" & goto :fail)

echo.
echo [C] Weather/events ablation - answers open question #2, exports the
echo     conditions model that predict_live.py serves
"%VENV_PY%" -m src.batch.ablation
if errorlevel 1 (set "STEP=C ablation" & goto :fail)

echo.
echo [D] Render report figures, map and GeoJSON
"%VENV_PY%" -m src.viz.make_maps
if errorlevel 1 (set "STEP=D make_maps" & goto :fail)

echo.
echo ============================================================================
echo  PIPELINE COMPLETE - %RUN_LABEL%  (K=%CHOSEN_K%)
echo ============================================================================
echo   data\processed\  demand.parquet, features.parquet, modeling_zones.parquet
echo   models\          kmeans\, zone_clusters.parquet, cluster_demand.parquet,
echo                    kmeans_metadata.json (records chosen K + both sweeps),
echo                    conditions_model.json, hist_avg.parquet
echo   reports\         k_selection.png, zone_hour_heatmap.png, cluster_map.html,
echo                    geojson\, k_sweep.csv, baseline_metrics.csv,
echo                    ablation_metrics.csv
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
