@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=C:\Users\D E B M A L Y A\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

cd /d "%PROJECT_DIR%"

if not exist "%PYTHON_EXE%" (
  echo Python was not found at:
  echo %PYTHON_EXE%
  echo.
  echo Install Python from https://www.python.org/downloads/ and then run:
  echo python -m venv .venv
  exit /b 1
)

"%PYTHON_EXE%" -m venv .venv
".venv\Scripts\python.exe" -m pip install "setuptools<81" pytest webrtcvad-wheels
echo.
echo Virtual environment is ready.
echo To activate it, run:
echo .venv\Scripts\activate
