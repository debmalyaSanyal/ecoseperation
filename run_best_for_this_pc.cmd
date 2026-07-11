@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  call setup_venv.cmd
)

set "ASR_ENGINE_OVERRIDE=faster-whisper"
set "ASR_MODEL_OVERRIDE=base"
set "LOCAL_ASR_DEVICE=cpu"
set "LOCAL_ASR_COMPUTE_TYPE=int8"
set "CAPTURE_BACKEND=sounddevice"
set "LOCAL_ASR_MIC_DEVICE=9"
set "LOCAL_ASR_SYSTEM_DEVICE=11"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"

".venv\Scripts\python.exe" main.py
