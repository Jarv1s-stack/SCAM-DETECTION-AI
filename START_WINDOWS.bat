@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ==============================================
echo   Scam Detection AI Lab - Windows launcher
echo ==============================================

echo Checking Python 3.13...
py -3.13 --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.13 is not installed.
    echo Run this in PowerShell: py install 3.13
    echo Then close and reopen the terminal and run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.13 virtual environment...
    py -3.13 -m venv .venv
    if errorlevel 1 goto :error
)

echo Installing application and audio dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements-audio.txt
if errorlevel 1 goto :error

echo Starting local website...
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo Startup failed. Read the error above and check README.md.
pause
exit /b 1
