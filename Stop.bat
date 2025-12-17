@echo off
chcp 65001 >nul
title Z-Image - Stop Servers

echo ============================================
echo    Stopping Z-Image WebUI Servers...
echo ============================================
echo.

:: Kill uvicorn (backend)
echo Stopping Backend Server...
taskkill /f /im python.exe /fi "WINDOWTITLE eq Z-Image Backend*" 2>nul

:: Kill npm/node (frontend)
echo Stopping Frontend Server...
taskkill /f /im node.exe /fi "WINDOWTITLE eq Z-Image Frontend*" 2>nul

echo.
echo ============================================
echo    All servers stopped.
echo ============================================
echo.

timeout /t 2 /nobreak >nul
