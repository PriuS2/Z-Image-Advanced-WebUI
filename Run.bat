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

:: Read config from config.yaml using Python
echo Reading configuration from config.yaml...
for /f %%i in ('call venv\Scripts\python.exe -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['server']['port'])"') do set BACKEND_PORT=%%i
for /f %%i in ('call venv\Scripts\python.exe -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['server']['host'])"') do set BACKEND_HOST=%%i
for /f %%i in ('call venv\Scripts\python.exe -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(str(c['server']['debug']).lower())"') do set DEBUG_MODE=%%i

if "%DEBUG_MODE%"=="true" (
    set RELOAD_FLAG=--reload
) else (
    set RELOAD_FLAG=
)

echo [1/2] Starting Backend Server...
start "Z-Image Backend" cmd /k "cd /d %~dp0 && call venv\Scripts\activate.bat && python -m uvicorn backend.main:app --host %BACKEND_HOST% --port %BACKEND_PORT% %RELOAD_FLAG%"

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
echo    Backend:  http://localhost:%BACKEND_PORT%
echo    Frontend: http://localhost:3000
echo    API Docs: http://localhost:%BACKEND_PORT%/docs
echo.
echo    Press any key to open the WebUI...
echo ============================================

pause >nul

:: Open browser
start http://localhost:3000

echo.
echo To stop the servers, close the terminal windows.
echo.
