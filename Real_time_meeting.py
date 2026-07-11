import sounddevice as sd
import numpy as np
import queue
import threading
from scipy.signal import resample
from faster_whisper import WhisperModel

# =====================
# CONFIG
# =====================

MIC_DEVICE = 9
SYSTEM_DEVICE = 11

DEVICE_SAMPLE_RATE = 48000
WHISPER_SAMPLE_RATE = 16000

CHUNK_SECONDS = 3

# =====================
# LOAD MODEL
# =====================

print("Loading Whisper on RTX 2050...")

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

print("Whisper Ready")

# =====================
# QUEUES
# =====================

mic_queue = queue.Queue()
sys_queue = queue.Queue()

# =====================
# RESAMPLING
# =====================

def to_whisper_rate(audio):

    target_length = int(
        len(audio) *
        WHISPER_SAMPLE_RATE /
        DEVICE_SAMPLE_RATE
    )

    return resample(
        audio,
        target_length
    ).astype(np.float32)

# =====================
# CALLBACKS
# =====================

def mic_callback(indata, frames, time_info, status):
    if status:
        print(status)

    mic_queue.put(indata.copy())


def sys_callback(indata, frames, time_info, status):
    if status:
        print(status)

    sys_queue.put(indata.copy())

# =====================
# TRANSCRIPTION THREAD
# =====================

def transcribe_stream(q, label):

    buffer = np.empty(
        (0, 2),
        dtype=np.float32
    )

    required_samples = (
        DEVICE_SAMPLE_RATE *
        CHUNK_SECONDS
    )

    while True:

        while not q.empty():
            chunk = q.get()
            buffer = np.concatenate(
                (buffer, chunk),
                axis=0
            )

        if len(buffer) >= required_samples:

            audio = buffer[:required_samples]

            buffer = buffer[
                required_samples:
            ]

            # stereo -> mono
            audio = np.mean(
                audio,
                axis=1
            )

            # 48k -> 16k
            audio = to_whisper_rate(
                audio
            )

            try:

                segments, info = (
                    model.transcribe(
                        audio,
                        beam_size=3,
                        vad_filter=False
                    )
                )

                text = " ".join(
                    seg.text.strip()
                    for seg in segments
                )

                if text.strip():

                    print(
                        f"\n[{label}] {text}"
                    )

            except Exception as e:

                print(
                    f"\n[{label}] ERROR:",
                    e
                )

# =====================
# START THREADS
# =====================

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

# =====================
# OPEN STREAMS
# =====================

mic_stream = sd.InputStream(
    device=MIC_DEVICE,
    channels=2,
    samplerate=DEVICE_SAMPLE_RATE,
    callback=mic_callback
)

sys_stream = sd.InputStream(
    device=SYSTEM_DEVICE,
    channels=2,
    samplerate=DEVICE_SAMPLE_RATE,
    callback=sys_callback
)

print("\nListening...\n")

with mic_stream, sys_stream:

    while True:
        pass