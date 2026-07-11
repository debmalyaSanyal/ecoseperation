# realtime_transcription.py

import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
from scipy.signal import resample

# --------------------------
# CONFIG
# --------------------------

MIC_DEVICE = 9
SYSTEM_DEVICE = 11

SAMPLE_RATE = 48000
CHUNK_DURATION = 2

# --------------------------
# LOAD WHISPER
# --------------------------

print("Loading Whisper on RTX 2050...")

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

print("Whisper Ready")

# --------------------------
# QUEUES
# --------------------------

mic_queue = queue.Queue()
sys_queue = queue.Queue()

def resample_to_16k(audio):
    target_length = int(len(audio) * 16000 / 48000)
    return resample(audio, target_length)

# --------------------------
# CALLBACKS
# --------------------------

def mic_callback(indata, frames, time_info, status):
    if status:
        print(status)

    mic_queue.put(indata.copy())

def sys_callback(indata, frames, time_info, status):
    if status:
        print(status)

    sys_queue.put(indata.copy())

# --------------------------
# TRANSCRIBE
# --------------------------

def transcribe_stream(audio_queue, label):

    buffer = np.empty((0, 1), dtype=np.float32)

    while True:

        while not audio_queue.empty():
            chunk = audio_queue.get()
            buffer = np.concatenate((buffer, chunk))

        if len(buffer) >= SAMPLE_RATE * CHUNK_DURATION:

            audio = buffer[:SAMPLE_RATE * CHUNK_DURATION]
            buffer = buffer[SAMPLE_RATE * CHUNK_DURATION:]

            audio = audio.flatten()
            audio = resample_to_16k(audio)
            audio = audio.astype(np.float32)

            segments, info = model.transcribe(
                audio,
                beam_size=3,
                vad_filter=True
            )

            text = ""

            for seg in segments:
                text += seg.text

            text = text.strip()

            if text:
                print(f"\n[{label}] {text}")

# --------------------------
# THREADS
# --------------------------

threading.Thread(
    target=transcribe_stream,
    args=(mic_queue, "YOU"),
    daemon=True
).start()

threading.Thread(
    target=transcribe_stream,
    args=(sys_queue, "MEETING"),
    daemon=True
).start()

# --------------------------
# AUDIO STREAMS
# --------------------------

print("Listening...")

mic_stream = sd.InputStream(
    device=MIC_DEVICE,
    channels=1,
    samplerate=48000,
    callback=mic_callback
)

sys_stream = sd.InputStream(
    device=SYSTEM_DEVICE,
    channels=1,
    samplerate=48000,   # Stereo Mix native rate
    callback=sys_callback
)

with mic_stream, sys_stream:

    while True:
        pass