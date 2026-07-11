@echo off
cd /d "%~dp0"

if not exist "work" mkdir "work"

echo Generating Hinglish test audio...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Rate = -1; $synth.Volume = 100; $text = Get-Content -LiteralPath 'work\hinglish_test_prompt.txt' -Raw; $synth.SetOutputToWaveFile('work\hinglish_test_prompt.wav'); $synth.Speak($text); $synth.SetOutputToNull(); $synth.Dispose()"
if errorlevel 1 exit /b %errorlevel%

echo Running robust ASR matrix...
".venv\Scripts\python.exe" robust_asr_temp_benchmark.py
