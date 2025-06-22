@echo off
REM OmniParser Installation Script for Windows
REM Auto-detects platform and installs appropriate dependencies

echo 🚀 OmniParser Installation Script
echo ==================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.8+ first.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo 🐍 Python version: %python_version%

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Run the auto-detection installer
echo 🔍 Running platform-specific installer...
python install_dependencies.py

echo.
echo ✅ Installation completed!
echo.
echo To activate the environment in the future:
echo   .venv\Scripts\activate.bat
echo.
echo To run OmniParser:
echo   python batch_run_omniparser.py
echo.
pause 