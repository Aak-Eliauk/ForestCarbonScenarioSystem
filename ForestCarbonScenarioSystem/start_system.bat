@echo off
cd /d "%~dp0"

echo.
echo Starting Forest Loss Scenario Control and Monte Carlo Carbon Stock Simulation System V1.0...
echo Short name: Forest Loss Carbon Simulation
echo Please keep this terminal window open.
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Please install Python or add Python to PATH.
    echo.
    pause
    exit /b 1
)

python run_system.py %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if "%EXIT_CODE%"=="0" (
    echo System stopped.
) else (
    echo System stopped with error code %EXIT_CODE%.
)
pause
exit /b %EXIT_CODE%
