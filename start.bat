@echo off
echo ==============================================
echo NRL AI Predictor - Fast Setup ^& Launch
echo ==============================================
echo.

:: Ensure dependencies are installed quietly
echo Checking and installing requirements...
pip install -r requirements.txt > nul 2>&1

echo Starting FastAPI Server...
echo.
echo ==============================================
echo Go to: http://localhost:8000
echo ==============================================
echo.
uvicorn api:app --reload --host 0.0.0.0 --port 8000
