
"""
FINAL UPDATED MAIN.PY
Production orchestrator integrating upgraded modules.
"""

import signal
import threading
import queue
import time

from config import *
from audio_engine import AudioEngine
from audio_processor import AudioProcessor
from aec import EchoCanceller
from vad import VoiceActivityDetector
from transcriber import Transcriber
from postprocess import PostProcessor
from logger import (
    initialize_logger,
    banner,
    log,
    success,
    error,
)

shutdown_event = threading.Event()
task_queue = queue.Queue(maxsize=QUEUE_SIZE)

processor = AudioProcessor()
aec = EchoCanceller()
vad = VoiceActivityDetector()
transcriber = Transcriber()
post = PostProcessor()

mic_engine = AudioEngine(MIC_DEVICE, INPUT_SAMPLE_RATE, MIC_CHANNELS, BUFFER_SECONDS)
system_engine = AudioEngine(SYSTEM_DEVICE, INPUT_SAMPLE_RATE, SYSTEM_CHANNELS, BUFFER_SECONDS)

latest_system = None
system_lock = threading.Lock()

def mic_producer():
    while not shutdown_event.is_set():
        audio = mic_engine.get_latest_audio()
        if audio is None:
            time.sleep(LOOP_SLEEP)
            continue
        task_queue.put(("MIC", audio))

def system_producer():
    global latest_system
    while not shutdown_event.is_set():
        audio = system_engine.get_latest_audio()
        if audio is None:
            time.sleep(LOOP_SLEEP)
            continue
        with system_lock:
            latest_system = audio
        task_queue.put(("SYSTEM", audio))

def worker():
    global latest_system
    while not shutdown_event.is_set():
        try:
            source, audio = task_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            if source == "MIC":
                with system_lock:
                    ref = latest_system
                audio = processor.process(audio, ref)
            else:
                audio = processor.process(audio)

            if audio is None:
                continue

            speech = vad.process(audio)
            if speech is None:
                continue

            language, text = transcriber.process(speech)
            if text:
                text = post.process(text)
                if text:
                    log(source, language or "unk", text)
        except Exception as exc:
            error(str(exc))
        finally:
            task_queue.task_done()

def shutdown(*_):
    shutdown_event.set()
    mic_engine.stop()
    system_engine.stop()
    success("Shutdown complete.")

def main():
    banner()
    initialize_logger()
    signal.signal(signal.SIGINT, shutdown)
    mic_engine.start()
    system_engine.start()
    threading.Thread(target=mic_producer, daemon=True).start()
    threading.Thread(target=system_producer, daemon=True).start()
    for _ in range(max(1, WHISPER_WORKERS)):
        threading.Thread(target=worker, daemon=True).start()
    success("Meeting assistant running.")
    while not shutdown_event.is_set():
        time.sleep(1)

if __name__ == "__main__":
    main()
