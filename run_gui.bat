@echo off
REM Local presentation GUI: rebuilds the payload from models\, then serves gui\ on
REM 127.0.0.1 and opens a browser. No Docker, no JAVA_HOME, no Spark - the page is
REM static and the model arithmetic runs in the browser.
REM
REM   run_gui.bat            rebuild the payload, then serve on http://localhost:8765
REM   run_gui.bat serve      skip the rebuild (use the committed payload)
REM   run_gui.bat build      rebuild only, do not serve

setlocal
cd /d "%~dp0"

set "PORT=8765"
set "MODE=all"
if /i "%~1"=="serve" set "MODE=serve"
if /i "%~1"=="build" set "MODE=build"

echo ============================================================================
echo  NYC Demand Console - local web GUI
echo ============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found. Create it first:
    echo     py -3.11 -m venv .venv
    echo     .venv\Scripts\activate
    echo     pip install -r requirements.txt
    exit /b 1
)
set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if "%MODE%"=="serve" goto :serve

echo.
echo [1/2] Exporting the model into gui\payload.json
"%VENV_PY%" -m src.viz.build_gui
if errorlevel 1 (
    echo.
    echo ############################################################################
    echo  BUILD FAILED - the payload was not written, so nothing was changed.
    echo  If artifacts are missing, run run_pipeline_full_year.bat first.
    echo ############################################################################
    exit /b 1
)

if "%MODE%"=="build" (
    echo.
    echo Built. Open gui\standalone.html directly, or run:  run_gui.bat serve
    endlocal
    exit /b 0
)

:serve
if not exist "gui\payload.json" (
    echo ERROR: gui\payload.json is missing - run  run_gui.bat  without arguments.
    exit /b 1
)

echo.
echo [2/2] Serving gui\ at http://localhost:%PORT%
echo.
echo   The console opens in your default browser.
echo   Leave THIS window open while presenting - closing it stops the server.
echo   Press Ctrl+C here to stop.
echo.
echo   No server wanted? gui\standalone.html holds the same console in one file.
echo.

REM Give the server a moment to bind before the browser asks for the page.
start "" /b cmd /c "ping -n 2 127.0.0.1 >nul & start "" http://localhost:%PORT%/"

REM --bind 127.0.0.1: a presentation tool has no business on the room's network.
"%VENV_PY%" -m http.server %PORT% --bind 127.0.0.1 --directory gui

endlocal
exit /b 0
