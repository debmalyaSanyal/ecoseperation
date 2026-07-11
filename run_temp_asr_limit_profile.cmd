@echo off
cd /d "%~dp0"

set "LOCAL_ASR_NATIVE_DIAGNOSTICS=true"
set "LOCAL_ASR_NATIVE_CHUNK_FRAMES=1024"
set "LOCAL_ASR_UTTERANCE_LOW_PACKET_TARGET=3"
set "LOCAL_ASR_UTTERANCE_RMS_THRESHOLD=150"
set "ASR_MODEL_OVERRIDE=small"

rem CUDA decode is currently blocked by missing cudnn_ops_infer64_8.dll.
rem This CPU profile was the best stable setting from the temp limit test.
set "LOCAL_ASR_DEVICE=cpu"
set "LOCAL_ASR_COMPUTE_TYPE=int8"

".venv\Scripts\python.exe" main.py
