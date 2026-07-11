"""
=========================================================
OFFLINE AI MEETING ASSISTANT

ADAPTIVE ECHO CANCELLER

Production Version

✓ Adaptive suppression
✓ Dynamic attenuation
✓ Double-talk protection
✓ Energy estimation
✓ Smooth gain transition
✓ Stable output

=========================================================
"""

import numpy as np

from config import *



class EchoCanceller:

    def __init__(self):

        self.speaker_threshold = 0.02

        self.mic_threshold = 0.01

        self.previous_gain = 1.0

        self.attack = 0.25

        self.release = 0.05

    # ------------------------------------------------

    def rms(self, audio):

        return np.sqrt(

            np.mean(

                audio.astype(np.float32) ** 2

            )

            + 1e-10

        )

    # ------------------------------------------------

    def stereo_to_mono(self, audio):

        if audio is None:

            return None

        if audio.ndim == 2:

            return np.mean(

                audio,

                axis=1

            )

        return audio

    # ------------------------------------------------

    def smooth_gain(

        self,

        target_gain

    ):

        if target_gain < self.previous_gain:

            gain = (

                self.attack * target_gain

                +

                (1 - self.attack)

                * self.previous_gain

            )

        else:

            gain = (

                self.release * target_gain

                +

                (1 - self.release)

                * self.previous_gain

            )

        self.previous_gain = gain

        return gain

    # ------------------------------------------------

    def process(

        self,

        mic_audio,

        speaker_audio

    ):

        if speaker_audio is None:

            return mic_audio.astype(

                np.float32

            )

        mic_audio = self.stereo_to_mono(

            mic_audio

        )

        speaker_audio = self.stereo_to_mono(

            speaker_audio

        )

        n = min(

            len(mic_audio),

            len(speaker_audio)

        )

        mic_audio = mic_audio[:n]

        speaker_audio = speaker_audio[:n]

        mic_energy = self.rms(

            mic_audio

        )

        speaker_energy = self.rms(

            speaker_audio

        )

        # --------------------------------------
        # No active speaker
        # --------------------------------------

        if speaker_energy < self.speaker_threshold:

            gain = self.smooth_gain(

                1.0

            )

            return (

                mic_audio * gain

            ).astype(np.float32)

        # --------------------------------------
        # Double-talk detection
        # --------------------------------------

        ratio = mic_energy / (

            speaker_energy + 1e-8

        )

        if ratio > 0.75:

            gain = self.smooth_gain(

                1.0

            )

            return (

                mic_audio * gain

            ).astype(np.float32)

        # --------------------------------------
        # Adaptive suppression
        # --------------------------------------

        suppression = np.interp(

            ratio,

            [0.0, 0.75],

            [0.10, 1.00]

        )

        gain = self.smooth_gain(

            suppression

        )

        output = mic_audio * gain

        return output.astype(

            np.float32

        )

    # ------------------------------------------------

    def statistics(

        self,

        mic_audio,

        speaker_audio

    ):

        mic_audio = self.stereo_to_mono(

            mic_audio

        )

        speaker_audio = self.stereo_to_mono(

            speaker_audio

        )

        return {

            "mic_rms":

                float(

                    self.rms(

                        mic_audio

                    )

                ),

            "speaker_rms":

                float(

                    self.rms(

                        speaker_audio

                    )

                ),

            "gain":

                round(

                    self.previous_gain,

                    3

                )

        }

    # ------------------------------------------------

    def reset(self):

        self.previous_gain = 1.0