@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo First run START_WINDOWS.bat to create the virtual environment.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements-audio.txt
if errorlevel 1 (
    echo Audio dependencies installation failed.
    pause
    exit /b 1
)
echo Audio transcription support is installed.
pause
