@echo off
chcp 65001 >nul
title Z-Image Advanced WebUI

echo ============================================
echo    Z-Image Advanced WebUI Launcher
echo ============================================
echo.

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run the following commands first:
    echo   python -m venv venv
    echo   .\venv\Scripts\activate
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    echo   pip install -r requirements.txt
    pause
    exit /b 1
)

:: Check if node_modules exists
if not exist "frontend\node_modules" (
    echo [WARNING] Frontend dependencies not installed.
    echo Installing npm packages...
    cd frontend
    call npm install
    cd ..
    echo.
)

:: Create required directories
if not exist "outputs" mkdir outputs
if not exist "uploads" mkdir uploads
if not exist "controls" mkdir controls
if not exist "masks" mkdir masks

echo [1/2] Starting Backend Server...
start "Z-Image Backend" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload"

:: Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 3 /nobreak >nul

echo [2/2] Starting Frontend Server...
start "Z-Image Frontend" cmd /k "cd /d %~dp0\frontend && npm run dev"

echo.
echo ============================================
echo    Servers are starting...
echo ============================================
echo.
echo    Backend:  http://localhost:8080
echo    Frontend: http://localhost:3000
echo    API Docs: http://localhost:8080/docs
echo.
echo    Press any key to open the WebUI...
echo ============================================

pause >nul

:: Open browser
start http://localhost:3000

echo.
echo To stop the servers, close the terminal windows.
echo.
