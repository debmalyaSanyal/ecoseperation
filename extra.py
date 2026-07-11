import sounddevice as sd
import soundfile as sf
import threading

def record_mic():
    audio = sd.rec(
        int(5 * 48000),
        samplerate=48000,
        channels=2,
        dtype="float32",
        device=9
    )
    sd.wait()
    sf.write("mic.wav", audio, 48000)
    print("Mic done")

def record_system():
    audio = sd.rec(
        int(5 * 48000),
        samplerate=48000,
        channels=2,
        dtype="float32",
        device=11
    )
    sd.wait()
    sf.write("system.wav", audio, 48000)
    print("System done")

t1 = threading.Thread(target=record_mic)
t2 = threading.Thread(target=record_system)

t1.start()
t2.start()

t1.join()
t2.join()

print("Finished")