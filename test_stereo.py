# import sounddevice as sd
# import soundfile as sf

# DEVICE_ID = 11

# print("Recording 10 seconds...")

# audio = sd.rec(
#     int(10 * 16000),
#     samplerate=16000,
#     channels=1,
#     dtype='float32',
#     device=DEVICE_ID
# )

# sd.wait()

# sf.write(
#     "stereo_test.wav",
#     audio,
#     16000
# )

# print("Saved stereo_test.wav")

import sounddevice as sd
import soundfile as sf

DEVICE_ID = 11

print("Recording...")

audio = sd.rec(
    int(10 * 48000),
    samplerate=48000,
    channels=2,
    dtype="float32",
    device=DEVICE_ID
)

sd.wait()

sf.write(
    "stereo_test.wav",
    audio,
    48000
)

print("Done")