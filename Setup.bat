@echo off
chcp 65001 >nul
title Z-Image Advanced WebUI - Setup

echo ============================================
echo    Z-Image Advanced WebUI Setup
echo ============================================
echo.

:: Get the script directory
cd /d %~dp0

:: ==========================================
:: Step 1: Python Virtual Environment
:: ==========================================
echo [1/5] Checking Python virtual environment...

if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        echo Please make sure Python 3.10+ is installed.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.

:: ==========================================
:: Step 2: Upgrade pip
:: ==========================================
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

:: ==========================================
:: Step 3: Install PyTorch with CUDA
:: ==========================================
echo [3/5] Installing PyTorch with CUDA 12.6 support...
echo This may take a while...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (
    echo [WARNING] PyTorch installation may have issues.
    echo Continuing with other dependencies...
)
echo.

:: ==========================================
:: Step 4: Install Python dependencies
:: ==========================================
echo [4/5] Installing Python dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install some Python dependencies.
    echo Please check the error messages above.
    pause
    exit /b 1
)
echo.

:: ==========================================
:: Step 5: Frontend Setup (Node.js / npm)
:: ==========================================
echo [5/5] Setting up Frontend...

:: Check if npm is installed
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm is not installed!
    echo.
    echo Please install Node.js from:
    echo   https://nodejs.org/
    echo.
    echo After installing Node.js, run this script again.
    echo.
    pause
    exit /b 1
)

echo npm found. Installing frontend dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install frontend dependencies.
    echo Please check the error messages above.
    cd ..
    pause
    exit /b 1
)
cd ..

echo.

:: ==========================================
:: Create required directories
:: ==========================================
echo Creating required directories...
if not exist "outputs" mkdir outputs
if not exist "uploads" mkdir uploads
if not exist "controls" mkdir controls
if not exist "masks" mkdir masks
if not exist "models" mkdir models

echo.
echo ============================================
echo    Setup Complete!
echo ============================================
echo.
echo You can now run the application using:
echo   Run.bat
echo.
echo Or manually:
echo   Backend:  .\venv\Scripts\activate ^&^& python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
echo   Frontend: cd frontend ^&^& npm run dev
echo.
pause
