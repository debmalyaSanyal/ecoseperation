# Final Project Benchmark and Setup Report

Date: 2026-06-09

Project path: `C:\Users\D E B M A L Y A\OneDrive\Desktop\Alpha\Final Project`

## 1. Goal

The goal was to test the local offline meeting assistant project and tune it for the current Windows system. The focus was practical runtime stability first, then best usable settings for local transcription.

## 2. System Profile Detected

| Item | Result |
|---|---|
| OS | Windows 11 |
| Architecture | AMD64 |
| Python in venv | Python 3.12.13 |
| CPU threads | 12 |
| RAM | 7.65 GB |
| Free disk | About 135 GB |
| GPU detected | NVIDIA GeForce RTX 2050 |
| GPU VRAM | 4 GB |
| CUDA-capable GPU | Yes, hardware detected |

Important finding: the GPU exists, but the installed PyTorch build in the virtual environment is CPU-only. For this reason, the stable tuned profile uses CPU `int8` instead of GPU.

## 3. Environment Fixes Applied

| Check / Fix | Result |
|---|---|
| Virtual environment | Created successfully at `.venv` |
| Python command issue | Fixed by using available Python instead of unavailable `python3.11` |
| `pytest` | Installed |
| `webrtcvad` import | Fixed by installing `webrtcvad-wheels` |
| `pkg_resources` / CTranslate2 issue | Fixed by pinning `setuptools<81` |
| Faster-Whisper import | Working |
| Sounddevice import | Working |
| Silero VAD import | Working |

## 4. Benchmarks and Tests Run

| Benchmark / Test | Command Type | Result |
|---|---|---|
| Python version check | `.venv\Scripts\python.exe --version` | Passed: Python 3.12.13 |
| Code compile check | Python compile-all | Passed |
| Runtime test suite | Pytest | Passed: 29 tests |
| System profiler | Project profiler | Passed: detected Windows 11, 12 CPU threads, 7.65 GB RAM, RTX 2050 |
| Dependency import check | Import scan | Passed after fixes |
| Tiny model load test | Faster-Whisper CPU int8 | Passed |
| Base model load test | Faster-Whisper CPU int8 | Passed |
| Native preflight | Windows WASAPI path | Preflight passed, but full native launch needs missing helper binary |
| Sounddevice preflight before tuning | Fallback path | Failed because system audio loopback was not detected |
| Sounddevice preflight after tuning | Fallback path | Passed with Stereo Mix |
| Actual launch smoke test | Short launch run | Reached running state; no immediate startup crash before timeout stop |

Latest verified test result:

```text
29 passed, 1 warning in 1.70s
```

The warning was only about pytest cache write permission inside the OneDrive project folder, not a project failure.

## 5. Model Benchmarks

| Model | Device | Compute Type | Result | Notes |
|---|---|---|---|---|
| `tiny` | CPU | int8 | Loaded successfully | Fastest, lowest accuracy |
| `base` | CPU | int8 | Loaded successfully | Best stable default for this system |

Cached models found:

| Model | Cache Status |
|---|---|
| `tiny` | Cached |
| `base` | Cached |

Total measured model cache file size from the tested model tree was about 226 MB.

## 6. Audio Device Results

| Device Role | Selected Device |
|---|---|
| Microphone | `[9] Microphone Array (Intel Smart Sound Technology for Digital Microphones)` |
| System audio | `[11] Stereo Mix (Realtek HD Audio Stereo input)` |

Before tuning, the project did not recognize `Stereo Mix` as loopback/system audio. I updated the device resolver to accept Windows loopback names including:

```text
stereo mix
what u hear
wave out
```

## 7. Best Settings Selected

The tuned launcher is:

```text
run_best_for_this_pc.cmd
```

It sets:

| Setting | Value | Reason |
|---|---|---|
| ASR engine | `faster-whisper` | Only fully implemented local engine in this code |
| Model | `base` | Better quality than tiny, still stable on 8 GB RAM |
| Device | `cpu` | Current Python ML stack is CPU-only |
| Compute type | `int8` | Lower RAM/CPU load |
| Capture backend | `sounddevice` | Works now with Stereo Mix |
| Mic device | `9` | Detected default microphone array |
| System device | `11` | Stereo Mix input |
| CPU thread env | `4` | Conservative for real-time transcription |

Runtime config tuning:

| Setting | Old | New | Reason |
|---|---:|---:|---|
| `BUFFER_SECONDS` | 3 | 2 | Lower latency |
| `QUEUE_SIZE` | 64 | 32 | Lower memory pressure |
| `BEAM_SIZE` | 7 | 1 | Much faster decode |
| `BEST_OF` | 7 | 1 | Much faster decode |
| `PATIENCE` | 2 | 1 | Faster decode |
| `CONDITION_ON_PREVIOUS_TEXT` | True | False | Reduces repeated/laggy text in live chunks |

## 8. Recommended Run Command

From PowerShell:

```powershell
cd "C:\Users\D E B M A L Y A\OneDrive\Desktop\Alpha\Final Project"
.\run_best_for_this_pc.cmd
```

When prompted, type:

```text
Launch Transcription
```

## 9. Limitations Found

1. GPU acceleration is not active yet.
   - RTX 2050 hardware was detected.
   - PyTorch reported CPU-only.
   - Current stable profile stays on CPU `int8`.

2. Native Windows WASAPI helper is not built.
   - The project expects `native-audio-capture\windows\bin\win-x64\native-audio-streamer.exe`.
   - That file was missing.
   - `dotnet` was not installed, so the helper could not be built during testing.

3. OneDrive permissions caused a pytest cache warning.
   - The tests still passed.
   - This warning does not block running the app.

## 10. Final Recommendation

Use the `base` CPU `int8` profile now. It is the best stable setting tested for this system because it balances transcription quality, RAM usage, and startup reliability.

For a later upgrade, install the correct CUDA-enabled ML stack and .NET SDK. Then the project can be retested for GPU Faster-Whisper or native Windows WASAPI capture.
