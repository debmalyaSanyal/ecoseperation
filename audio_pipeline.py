"""
Runtime audio pipeline for local ASR.
"""

import queue
import signal
import threading
import time

from app_config import RuntimeConfig, apply_device_config
from device_manager import ResolvedDevices


class AudioPipeline:
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.task_queue = None
        self.mic_engine = None
        self.system_engine = None
        self.latest_system = None
        self.system_lock = threading.Lock()

    def start(self, runtime_config: RuntimeConfig, devices: ResolvedDevices) -> None:
        runtime_config.apply_to_legacy_config()
        apply_device_config(devices.mic.index, devices.system.index)

        from config import (
            BUFFER_SECONDS,
            INPUT_SAMPLE_RATE,
            LOOP_SLEEP,
            MIC_CHANNELS,
            QUEUE_SIZE,
            SYSTEM_CHANNELS,
            WHISPER_WORKERS,
        )
        from audio_engine import AudioEngine
        from audio_processor import AudioProcessor
        from logger import banner, error, initialize_logger, log, success
        from transcription_worker import TranscriptionWorker

        processor = AudioProcessor()
        transcription_worker = TranscriptionWorker()

        self.task_queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.mic_engine = AudioEngine(
            devices.mic.index,
            INPUT_SAMPLE_RATE,
            MIC_CHANNELS,
            BUFFER_SECONDS,
        )
        self.system_engine = AudioEngine(
            devices.system.index,
            INPUT_SAMPLE_RATE,
            SYSTEM_CHANNELS,
            BUFFER_SECONDS,
        )

        def mic_producer():
            while not self.shutdown_event.is_set():
                audio = self.mic_engine.get_latest_audio()
                if audio is None:
                    time.sleep(LOOP_SLEEP)
                    continue
                self.task_queue.put(("MIC", audio))

        def system_producer():
            while not self.shutdown_event.is_set():
                audio = self.system_engine.get_latest_audio()
                if audio is None:
                    time.sleep(LOOP_SLEEP)
                    continue
                with self.system_lock:
                    self.latest_system = audio
                self.task_queue.put(("SYSTEM", audio))

        def worker():
            while not self.shutdown_event.is_set():
                try:
                    source, audio = self.task_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                try:
                    if source == "MIC":
                        with self.system_lock:
                            ref = self.latest_system
                        audio = processor.process(audio, ref)
                    else:
                        audio = processor.process(audio)

                    if audio is None:
                        continue

                    transcription_worker.handle(source, audio)
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
        self.mic_engine.start()
        self.system_engine.start()
        threading.Thread(target=mic_producer, daemon=True).start()
        threading.Thread(target=system_producer, daemon=True).start()
        for _ in range(max(1, WHISPER_WORKERS)):
            threading.Thread(target=worker, daemon=True).start()

        success("Local transcription running.")
        while not self.shutdown_event.is_set():
            time.sleep(1)

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.mic_engine is not None:
            self.mic_engine.stop()
        if self.system_engine is not None:
            self.system_engine.stop()
