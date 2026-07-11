"""
Native macOS audio pipeline fed by the Swift ScreenCaptureKit helper.
"""

from __future__ import annotations

import asyncio
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from app_config import RuntimeConfig


NATIVE_HELPER_DIR = Path("native-audio-capture") / "macos"
NATIVE_HELPER_BINARY = (
    NATIVE_HELPER_DIR
    / ".build"
    / "x86_64-apple-macosx"
    / "debug"
    / "native-audio-streamer"
)
NATIVE_PERMISSION_GUIDANCE = (
    "Native audio capture needs Microphone and Screen & System Audio Recording "
    "permission for Terminal/iTerm/VS Code or the packaged helper. After changing "
    "permissions, quit and reopen the host app before testing again."
)


def decode_pcm16_stereo(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    usable_bytes = len(payload) - (len(payload) % 4)
    if usable_bytes <= 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty

    pcm = np.frombuffer(payload[:usable_bytes], dtype="<i2").reshape(-1, 2)
    samples = pcm.astype(np.float32) / 32768.0
    system_audio = samples[:, 0].copy()
    mic_audio = samples[:, 1].copy()
    return system_audio, mic_audio


class SlidingAudioBuffer:
    def __init__(
        self,
        sample_rate: int,
        window_seconds: float,
        overlap_seconds: float = 0.25,
    ):
        self.required_samples = int(sample_rate * window_seconds)
        self.overlap_samples = min(
            int(sample_rate * overlap_seconds),
            max(0, self.required_samples - 1),
        )
        self.step_samples = max(1, self.required_samples - self.overlap_samples)
        self.buffer = np.empty(0, dtype=np.float32)

    def append(self, audio: np.ndarray) -> list[np.ndarray]:
        if audio.size == 0:
            return []

        self.buffer = np.concatenate((self.buffer, audio.astype(np.float32, copy=False)))
        chunks = []
        while len(self.buffer) >= self.required_samples:
            chunks.append(self.buffer[: self.required_samples].copy())
            self.buffer = self.buffer[self.step_samples :]
        return chunks


class UtteranceAudioBuffer:
    def __init__(
        self,
        sample_rate: int,
        rms_threshold: float = 80.0,
        low_packet_target: int = 3,
        min_seconds: float = 0.35,
        max_seconds: float = 12.0,
    ):
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.low_packet_target = max(1, low_packet_target)
        self.min_samples = max(1, int(sample_rate * min_seconds))
        self.max_samples = max(self.min_samples, int(sample_rate * max_seconds))
        self.buffer = np.empty(0, dtype=np.float32)
        self.active = False
        self.low_packets = 0

    def append(self, audio: np.ndarray) -> list[np.ndarray]:
        if audio.size == 0:
            return []

        audio = audio.astype(np.float32, copy=False)
        rms = pcm16_rms(audio)
        is_active = rms > self.rms_threshold
        chunks = []

        if is_active:
            self.active = True
            self.low_packets = 0
            self.buffer = np.concatenate((self.buffer, audio))
        elif self.active:
            self.low_packets += 1
            self.buffer = np.concatenate((self.buffer, audio))
            if self.low_packets >= self.low_packet_target:
                chunk = self.flush()
                if chunk is not None:
                    chunks.append(chunk)

        while len(self.buffer) >= self.max_samples:
            chunks.append(self.buffer[: self.max_samples].copy())
            self.buffer = self.buffer[self.max_samples :]
            self.active = bool(self.buffer.size)
            self.low_packets = 0

        return chunks

    def flush(self) -> np.ndarray | None:
        if len(self.buffer) < self.min_samples:
            self.reset()
            return None

        chunk = self.buffer.copy()
        self.reset()
        return chunk

    def reset(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)
        self.active = False
        self.low_packets = 0


def pcm16_rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32, copy=False)))) * 32768.0)


class NativeAudioPipeline:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.task_queue: queue.Queue[tuple[str, np.ndarray]] = queue.Queue()
        self.helper_process: subprocess.Popen | None = None
        self.system_buffer: SlidingAudioBuffer | UtteranceAudioBuffer | None = None
        self.mic_buffer: SlidingAudioBuffer | UtteranceAudioBuffer | None = None
        self.worker = None

    def start(self, runtime_config: RuntimeConfig) -> None:
        runtime_config.apply_to_legacy_config()
        runtime_config.apply_native_low_latency_config()

        from config import QUEUE_SIZE, WHISPER_SAMPLE_RATE, WHISPER_WORKERS
        from logger import banner, error, initialize_logger, success
        from transcription_worker import TranscriptionWorker

        self.task_queue = queue.Queue(maxsize=QUEUE_SIZE)
        if runtime_config.native_utterance_mode and not runtime_config.asr_engine_supports_streaming:
            self.system_buffer = UtteranceAudioBuffer(
                WHISPER_SAMPLE_RATE,
                runtime_config.native_utterance_rms_threshold,
                runtime_config.native_utterance_low_packet_target,
                runtime_config.native_utterance_min_seconds,
                runtime_config.native_utterance_max_seconds,
            )
            self.mic_buffer = UtteranceAudioBuffer(
                WHISPER_SAMPLE_RATE,
                runtime_config.native_utterance_rms_threshold,
                runtime_config.native_utterance_low_packet_target,
                runtime_config.native_utterance_min_seconds,
                runtime_config.native_utterance_max_seconds,
            )
        else:
            self.system_buffer = SlidingAudioBuffer(
                WHISPER_SAMPLE_RATE,
                runtime_config.native_window_seconds,
                runtime_config.native_overlap_seconds,
            )
            self.mic_buffer = SlidingAudioBuffer(
                WHISPER_SAMPLE_RATE,
                runtime_config.native_window_seconds,
                runtime_config.native_overlap_seconds,
            )
        self.worker = TranscriptionWorker(diagnostics=runtime_config.native_diagnostics)

        def worker_loop():
            while not self.shutdown_event.is_set():
                try:
                    source, audio = self.task_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    self.worker.handle(source, audio)
                except Exception as exc:
                    error(str(exc))
                finally:
                    self.task_queue.task_done()

        def shutdown(*_):
            self.stop()
            success("Shutdown complete.")

        banner()
        initialize_logger()
        signal.signal(signal.SIGINT, shutdown)
        for _ in range(max(1, WHISPER_WORKERS)):
            threading.Thread(target=worker_loop, daemon=True).start()

        try:
            asyncio.run(self._run_server_and_helper(runtime_config))
        except KeyboardInterrupt:
            shutdown()
        except Exception as exc:
            error(str(exc))
            self.stop()

    async def _run_server_and_helper(self, runtime_config: RuntimeConfig) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError(
                "The 'websockets' package is not installed. Run: pip install websockets"
            ) from exc

        from logger import success

        host = runtime_config.native_ws_host
        port = runtime_config.native_ws_port
        async with websockets.serve(self._handle_connection, host, port):
            self._launch_helper(runtime_config)
            success(
                f"Native audio receiver listening on ws://{host}:{port}. "
                "Local transcription running."
            )
            while not self.shutdown_event.is_set():
                await asyncio.sleep(0.25)

    async def _handle_connection(self, websocket, path=None) -> None:
        from logger import info, warning

        info("Native audio helper connected.")
        async for message in websocket:
            if isinstance(message, str):
                if runtime_config_value("LOCAL_ASR_NATIVE_DIAGNOSTICS", False):
                    info(f"Native metadata ignored: {message[:200]}")
                continue
            if not isinstance(message, (bytes, bytearray)):
                continue

            system_audio, mic_audio = decode_pcm16_stereo(bytes(message))
            self._enqueue_chunks("SYSTEM", self.system_buffer, system_audio)
            self._enqueue_chunks("MIC", self.mic_buffer, mic_audio)

        warning("Native audio helper disconnected.")
        self._flush_pending_utterance("SYSTEM", self.system_buffer)
        self._flush_pending_utterance("MIC", self.mic_buffer)

    def _enqueue_chunks(
        self,
        source: str,
        audio_buffer: SlidingAudioBuffer | UtteranceAudioBuffer | None,
        audio: np.ndarray,
    ) -> None:
        if audio_buffer is None:
            return
        for chunk in audio_buffer.append(audio):
            if runtime_config_value("LOCAL_ASR_NATIVE_DIAGNOSTICS", False):
                from logger import info

                duration = len(chunk) / 16000
                info(
                    "Native ASR enqueue "
                    f"source={source} duration={duration:.2f}s "
                    f"queue_size={self.task_queue.qsize()}"
                )
            self.task_queue.put((source, chunk))

    def _flush_pending_utterance(
        self,
        source: str,
        audio_buffer: SlidingAudioBuffer | UtteranceAudioBuffer | None,
    ) -> None:
        if not isinstance(audio_buffer, UtteranceAudioBuffer):
            return
        chunk = audio_buffer.flush()
        if chunk is not None:
            self.task_queue.put((source, chunk))

    def _launch_helper(self, runtime_config: RuntimeConfig) -> None:
        from logger import info, warning

        helper_dir = runtime_config.native_helper_dir or NATIVE_HELPER_DIR
        if not helper_dir.exists():
            raise RuntimeError(f"Native helper folder not found: {helper_dir}")

        ws_url = f"ws://{runtime_config.native_ws_host}:{runtime_config.native_ws_port}"
        command = self._helper_command(runtime_config, ws_url)
        if runtime_config.native_diagnostics:
            command.append("--diagnostics")
        if not runtime_config.native_rms_logging:
            command.append("--no-rms-logging")

        env = os.environ.copy()
        swift_cache = helper_dir / ".swiftpm-cache"
        clang_cache = helper_dir / ".clang-module-cache"
        swift_cache.mkdir(parents=True, exist_ok=True)
        clang_cache.mkdir(parents=True, exist_ok=True)
        env.setdefault("XDG_CACHE_HOME", str(swift_cache))
        env.setdefault("CLANG_MODULE_CACHE_PATH", str(clang_cache))

        self.helper_process = subprocess.Popen(
            command,
            cwd=str(helper_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        info("Started native macOS audio helper.")
        threading.Thread(target=self._relay_helper_logs, daemon=True).start()
        time.sleep(0.5)
        if self.helper_process.poll() is not None:
            warning(
                "Native audio helper exited early. "
                f"{NATIVE_PERMISSION_GUIDANCE}"
            )

    def _helper_command(self, runtime_config: RuntimeConfig, ws_url: str) -> list[str]:
        helper_override = os.environ.get("LOCAL_ASR_NATIVE_HELPER_BIN")
        if helper_override:
            command = [helper_override]
        elif runtime_config.native_helper_binary and runtime_config.native_helper_binary.exists():
            command = [str(runtime_config.native_helper_binary.resolve())]
        elif runtime_config.native_helper_run_command:
            command = list(runtime_config.native_helper_run_command)
        elif NATIVE_HELPER_BINARY.exists():
            command = [str(NATIVE_HELPER_BINARY.resolve())]
        else:
            command = ["swift", "run", "native-audio-streamer"]

        command.extend(
            [
                "--ws-url",
                ws_url,
                "--call-id",
                "local-asr",
                "--session-id",
                "local-asr-session",
                "--chunk-frames",
                str(runtime_config.native_chunk_frames),
            ]
        )
        return command

    def _relay_helper_logs(self) -> None:
        from logger import info, warning

        process = self.helper_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            text = line.rstrip()
            if not text:
                continue
            if "[Native Error]" in text or "[Native Mic Warning]" in text:
                warning(text)
            else:
                info(text)

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.helper_process is not None and self.helper_process.poll() is None:
            self.helper_process.terminate()
            try:
                self.helper_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.helper_process.kill()


def runtime_config_value(env_name: str, default: bool) -> bool:
    value = os.environ.get(env_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
