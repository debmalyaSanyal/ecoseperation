from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model

from native_audio_pipeline import UtteranceAudioBuffer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent
WORK_DIR = PROJECT_ROOT / "work"
PROMPT_FILE = WORK_DIR / "hinglish_test_prompt.txt"
WAV_FILE = WORK_DIR / "hinglish_test_prompt.wav"
JSON_OUT = WORK_DIR / "robust_asr_temp_benchmark_results.json"
CSV_OUT = WORK_DIR / "robust_asr_temp_benchmark_results.csv"
PACKET_JSON_OUT = WORK_DIR / "robust_packet_gate_results.json"
GPU_JSON_OUT = WORK_DIR / "robust_gpu_probe_results.json"


@dataclass
class DecodeCase:
    model: str
    compute_type: str
    language: str
    beam_size: int
    vad_filter: bool
    initial_prompt: str


def wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def run_decode_case(case: DecodeCase, audio_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    model_path = download_model(
        case.model,
        local_files_only=True,
        cache_dir=str(PROJECT_ROOT / "models" / "whisper" / case.model),
    )
    model = WhisperModel(
        model_path,
        device="cpu",
        compute_type=case.compute_type,
        local_files_only=True,
        cpu_threads=4,
        num_workers=1,
    )
    load_seconds = time.perf_counter() - started

    decode_started = time.perf_counter()
    language = None if case.language == "auto" else case.language
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=case.beam_size,
        best_of=case.beam_size,
        temperature=0.0,
        vad_filter=case.vad_filter,
        condition_on_previous_text=False,
        without_timestamps=True,
        word_timestamps=False,
        language=language,
        initial_prompt=case.initial_prompt or None,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    decode_seconds = time.perf_counter() - decode_started

    duration = wav_duration_seconds(audio_path)
    return {
        **asdict(case),
        "status": "ok",
        "load_seconds": round(load_seconds, 3),
        "decode_seconds": round(decode_seconds, 3),
        "audio_seconds": round(duration, 3),
        "realtime_factor": round(decode_seconds / duration, 3),
        "detected_language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "text": text,
    }


def run_cpu_matrix(audio_path: Path) -> list[dict[str, Any]]:
    cases = [
        DecodeCase("base", "int8", "auto", 1, True, ""),
        DecodeCase("base", "int8", "en", 1, True, ""),
        DecodeCase("base", "int8", "hi", 1, True, ""),
        DecodeCase("base", "int8", "auto", 3, True, "Hinglish meeting notes with Hindi and English mixed words."),
        DecodeCase("base", "int8_float32", "auto", 1, True, ""),
        DecodeCase("base", "int8_float32", "auto", 1, False, ""),
        DecodeCase("small", "int8", "auto", 1, True, ""),
        DecodeCase("small", "int8", "en", 1, True, ""),
        DecodeCase("small", "int8", "hi", 1, True, ""),
        DecodeCase("small", "int8_float32", "auto", 1, True, ""),
        DecodeCase("small", "int8_float32", "en", 1, True, ""),
        DecodeCase("small", "int8_float32", "auto", 3, True, "Hinglish meeting notes with Hindi and English mixed words."),
        DecodeCase("small", "int8_float32", "auto", 1, False, ""),
        DecodeCase("small", "float32", "auto", 1, True, ""),
    ]

    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            result = run_decode_case(case, audio_path)
        except Exception as exc:
            result = {
                **asdict(case),
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return results


def run_packet_matrix() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for chunk_frames in [512, 1024, 2048, 4096]:
        for rms_threshold in [80, 120, 150, 220]:
            for low_packet_target in [2, 3, 4]:
                buffer = UtteranceAudioBuffer(
                    sample_rate=16000,
                    rms_threshold=rms_threshold,
                    low_packet_target=low_packet_target,
                    min_seconds=0.10,
                    max_seconds=12.0,
                )
                active = np.full(chunk_frames, 300 / 32768, dtype=np.float32)
                quiet = np.zeros(chunk_frames, dtype=np.float32)
                buffer.append(active)
                emitted_after_packet = None
                emitted_duration = None
                for packet_index in range(1, 7):
                    chunks = buffer.append(quiet)
                    if chunks:
                        emitted_after_packet = packet_index
                        emitted_duration = len(chunks[0]) / 16000
                        break
                results.append(
                    {
                        "chunk_frames": chunk_frames,
                        "chunk_ms": round(chunk_frames / 16000 * 1000, 1),
                        "rms_threshold": rms_threshold,
                        "low_packet_target": low_packet_target,
                        "emitted_after_silence_packet": emitted_after_packet,
                        "emitted_audio_seconds": round(emitted_duration, 3)
                        if emitted_duration is not None
                        else None,
                    }
                )
    return results


def run_gpu_probe() -> list[dict[str, Any]]:
    probes = []
    for model in ["base", "small"]:
        for compute_type in ["float16", "int8_float16", "int8"]:
            code = f"""
from pathlib import Path
from faster_whisper import WhisperModel
from faster_whisper.utils import download_model
model_name = {model!r}
compute_type = {compute_type!r}
model_path = download_model(model_name, local_files_only=True, cache_dir=str(Path('models/whisper') / model_name))
model = WhisperModel(model_path, device='cuda', compute_type=compute_type, local_files_only=True, cpu_threads=4, num_workers=1)
segments, info = model.transcribe('work/hinglish_test_prompt.wav', beam_size=1, best_of=1, temperature=0.0, vad_filter=True, without_timestamps=True)
print(' '.join(segment.text.strip() for segment in segments).strip())
"""
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=str(PROJECT_ROOT),
                text=True,
                capture_output=True,
                timeout=90,
            )
            probes.append(
                {
                    "model": model,
                    "device": "cuda",
                    "compute_type": compute_type,
                    "exit_code": completed.returncode,
                    "stdout": completed.stdout.strip(),
                    "stderr": completed.stderr.strip(),
                }
            )
    return probes


def write_outputs(results: list[dict[str, Any]], packet_results: list[dict[str, Any]], gpu_results: list[dict[str, Any]]) -> None:
    JSON_OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    PACKET_JSON_OUT.write_text(json.dumps(packet_results, indent=2, ensure_ascii=False), encoding="utf-8")
    GPU_JSON_OUT.write_text(json.dumps(gpu_results, indent=2, ensure_ascii=False), encoding="utf-8")

    fieldnames = sorted({key for result in results for key in result})
    with CSV_OUT.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run robust local ASR temp benchmarks.")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip crash-isolated CUDA decode probes.")
    args = parser.parse_args()

    if not WAV_FILE.exists():
        raise SystemExit(f"Missing Hinglish WAV file: {WAV_FILE}")

    results = run_cpu_matrix(WAV_FILE)
    packet_results = run_packet_matrix()
    gpu_results = [] if args.skip_gpu else run_gpu_probe()
    write_outputs(results, packet_results, gpu_results)
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {CSV_OUT}")
    print(f"Wrote {PACKET_JSON_OUT}")
    print(f"Wrote {GPU_JSON_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
