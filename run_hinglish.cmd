@echo off
setlocal

cd /d "%~dp0"

set "ASR_ENGINE_OVERRIDE=faster-whisper"
set "ASR_MODEL_OVERRIDE=base"
set "LOCAL_ASR_DEVICE=cpu"
set "LOCAL_ASR_COMPUTE_TYPE=int8"
set "CAPTURE_BACKEND=sounddevice"
set "LOCAL_ASR_MIC_DEVICE=9"
set "LOCAL_ASR_SYSTEM_DEVICE=11"

rem Hinglish mode: keep mixed Hindi-English speech in readable Roman text.
set "LOCAL_ASR_LANGUAGE=en"
set "LOCAL_ASR_INITIAL_PROMPT="

set "LOCAL_ASR_NATIVE_DIAGNOSTICS=true"
set "LOCAL_ASR_NATIVE_CHUNK_FRAMES=2048"
set "LOCAL_ASR_UTTERANCE_LOW_PACKET_TARGET=4"
set "LOCAL_ASR_UTTERANCE_RMS_THRESHOLD=120"
set "OMP_NUM_THREADS=4"
set "MKL_NUM_THREADS=4"

".venv\Scripts\python.exe" main.py
