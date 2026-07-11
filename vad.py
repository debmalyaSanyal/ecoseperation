"""
=========================================================
OFFLINE AI MEETING ASSISTANT

SILERO VAD

Production Version

Author : Debmalya Sanyal

=========================================================
"""

import numpy as np

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps,
)


class VoiceActivityDetector:

    def __init__(self):

        self.model = load_silero_vad()

        self.sample_rate = 16000

        self.min_speech_duration_ms = 250

        self.min_silence_duration_ms = 150

        self.threshold = 0.50

    # ----------------------------------------------------

    def process(self, audio):

        if audio is None:

            return None

        if len(audio) == 0:

            return None

        audio = audio.astype(np.float32)

        timestamps = get_speech_timestamps(

            audio,

            self.model,

            sampling_rate=self.sample_rate,

            threshold=self.threshold,

            min_speech_duration_ms=self.min_speech_duration_ms,

            min_silence_duration_ms=self.min_silence_duration_ms,

        )

        if len(timestamps) == 0:

            return None

        speech = []

        for segment in timestamps:

            start = segment["start"]

            end = segment["end"]

            speech.append(

                audio[start:end]

            )

        if len(speech) == 0:

            return None

        return np.concatenate(

            speech

        ).astype(np.float32)

    # ----------------------------------------------------

    def has_speech(self, audio):

        speech = self.process(audio)

        return speech is not None

    # ----------------------------------------------------

    def speech_ratio(self, audio):

        if audio is None:

            return 0

        timestamps = get_speech_timestamps(

            audio,

            self.model,

            sampling_rate=self.sample_rate,

            threshold=self.threshold,

            min_speech_duration_ms=self.min_speech_duration_ms,

            min_silence_duration_ms=self.min_silence_duration_ms,

        )

        if len(timestamps) == 0:

            return 0

        speech_samples = 0

        for seg in timestamps:

            speech_samples += (

                seg["end"]

                - seg["start"]

            )

        return speech_samples / len(audio)

    # ----------------------------------------------------

    def print_statistics(self, audio):

        ratio = self.speech_ratio(audio)

        print()

        print("=" * 50)

        print("Voice Activity Statistics")

        print("=" * 50)

        print(

            "Speech Ratio :",

            round(

                ratio * 100,

                2

            ),

            "%"

        )

        print()

    # ----------------------------------------------------

    def split_segments(self, audio):

        timestamps = get_speech_timestamps(

            audio,

            self.model,

            sampling_rate=self.sample_rate,

            threshold=self.threshold,

            min_speech_duration_ms=self.min_speech_duration_ms,

            min_silence_duration_ms=self.min_silence_duration_ms,

        )

        segments = []

        for seg in timestamps:

            segments.append(

                audio[

                    seg["start"]:

                    seg["end"]

                ]

            )

        return segments

    # ----------------------------------------------------

    def reset(self):

        pass