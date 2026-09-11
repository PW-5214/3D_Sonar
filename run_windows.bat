@echo off
setlocal
cd /d %~dp0
if not exist .venv (
  py -3.11 -m venv .venv 2>nul || python -m venv .venv
)
if not exist .venv\Scripts\python.exe (
  echo Failed to create the Python virtual environment.
  pause
  exit /b 1
)

set "PYTHON=.venv\Scripts\python.exe"
%PYTHON% -m pip install --upgrade pip
if errorlevel 1 (
  echo Failed to upgrade pip.
  pause
  exit /b 1
)

%PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install the required packages.
  pause
  exit /b 1
)

.venv\Scripts\python.exe -m streamlit run app.py
pause
