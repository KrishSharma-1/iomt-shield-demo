@echo off
echo.
echo ============================================
echo   IoMT Shield v2 - Hospital Cyber Defence
echo ============================================
echo.
echo Starting Streamlit dashboard on http://localhost:8501
echo Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.headless true

