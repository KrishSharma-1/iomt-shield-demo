@echo off
echo.
echo ============================================
echo   IoMT Shield - Attack Simulator CLI
echo ============================================
echo.
echo Usage: Choose a scenario when prompted.
echo.
cd /d "%~dp0"
set /p SCENARIO="Enter scenario (DDoS / DoS / MIXED / BENIGN / ALL) [default: MIXED]: "
if "%SCENARIO%"=="" set SCENARIO=MIXED
set /p EVENTS="Number of events [default: 30]: "
if "%EVENTS%"=="" set EVENTS=30
.venv\Scripts\python.exe simulate_attack.py --scenario %SCENARIO% --n %EVENTS%
pause

