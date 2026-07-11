from faster_whisper import WhisperModel
import time

print("Loading model...")

start = time.time()

model = WhisperModel(
    "small",
    device="cuda",
    compute_type="float16"
)

print(f"Loaded in {time.time()-start:.2f}s")
print("GPU transcription ready!")