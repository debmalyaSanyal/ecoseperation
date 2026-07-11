"""
=========================================================
OFFLINE AI MEETING ASSISTANT

HIGH PERFORMANCE AUDIO ENGINE

Low Latency
Thread Safe
Overflow Protection
Adaptive Buffering
Production Version
=========================================================
"""

import threading
import queue
import sounddevice as sd
import numpy as np

from config import *


class AudioEngine:

    def __init__(
        self,
        device,
        samplerate,
        channels,
        buffer_seconds=3,
    ):

        self.device = device
        self.samplerate = samplerate
        self.channels = channels
        self.buffer_seconds = buffer_seconds

        self.required_samples = int(
            samplerate * buffer_seconds
        )

        self.audio_buffer = np.empty(
            (0, channels),
            dtype=np.float32
        )

        self.lock = threading.Lock()

        self.running = False

        self.stream = None

        self.frame_count = 0

        self.drop_count = 0

        self.max_buffer_samples = (
            samplerate * 30
        )

        self.audio_queue = queue.Queue(
            maxsize=QUEUE_SIZE
        )

    # ------------------------------------------------

    def callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):

        if status and DEBUG:
            print(status)

        if not self.running:
            return

        audio = indata.copy()

        self.frame_count += frames

        with self.lock:

            self.audio_buffer = np.concatenate(
                (
                    self.audio_buffer,
                    audio
                ),
                axis=0
            )

            if len(self.audio_buffer) > self.max_buffer_samples:

                excess = (
                    len(self.audio_buffer)
                    - self.max_buffer_samples
                )

                self.audio_buffer = self.audio_buffer[
                    excess:
                ]

                self.drop_count += excess

    # ------------------------------------------------

    def start(self):

        if self.running:
            return

        self.stream = sd.InputStream(

            device=self.device,

            samplerate=self.samplerate,

            channels=self.channels,

            dtype=AUDIO_DTYPE,

            blocksize=BLOCK_SIZE,

            latency="low",

            callback=self.callback

        )

        self.stream.start()

        self.running = True

    # ------------------------------------------------

    def stop(self):

        self.running = False

        if self.stream is not None:

            try:

                self.stream.stop()

            except:

                pass

            try:

                self.stream.close()

            except:

                pass

    # ------------------------------------------------

    def clear(self):

        with self.lock:

            self.audio_buffer = np.empty(

                (
                    0,
                    self.channels
                ),

                dtype=np.float32

            )

    # ------------------------------------------------

    def available_samples(self):

        with self.lock:

            return len(self.audio_buffer)

    # ------------------------------------------------

    def available_seconds(self):

        return (
            self.available_samples()
            / self.samplerate
        )

    # ------------------------------------------------

    def get_audio(self):

        with self.lock:

            if len(
                self.audio_buffer
            ) < self.required_samples:

                return None

            chunk = self.audio_buffer[
                :self.required_samples
            ].copy()

            self.audio_buffer = self.audio_buffer[
                self.required_samples:
            ]

        return chunk

    # ------------------------------------------------

    def get_latest_audio(self):

        """
        Sliding window.

        Returns latest chunk while keeping
        overlap for smoother transcription.
        """

        with self.lock:

            if len(
                self.audio_buffer
            ) < self.required_samples:

                return None

            chunk = self.audio_buffer[
                -self.required_samples:
            ].copy()

            overlap = int(
                self.required_samples * 0.50
            )

            self.audio_buffer = self.audio_buffer[
                -overlap:
            ]

        return chunk

    # ------------------------------------------------

    def flush(self):

        with self.lock:

            data = self.audio_buffer.copy()

            self.audio_buffer = np.empty(

                (
                    0,
                    self.channels
                ),

                dtype=np.float32

            )

        return data

    # ------------------------------------------------

    def stats(self):

        return {

            "device": self.device,

            "running": self.running,

            "samplerate": self.samplerate,

            "channels": self.channels,

            "buffer_seconds": round(
                self.available_seconds(),
                2
            ),

            "buffer_samples": self.available_samples(),

            "frames_received": self.frame_count,

            "dropped_samples": self.drop_count

        }

    # ------------------------------------------------

    def print_stats(self):

        s = self.stats()

        print()

        print("=" * 50)

        print("Audio Engine Statistics")

        print("=" * 50)

        for k, v in s.items():

            print(f"{k:20}: {v}")

        print()

    # ------------------------------------------------

    def is_running(self):

        return self.running

    # ------------------------------------------------

    def restart(self):

        self.stop()

        self.clear()

        self.start()