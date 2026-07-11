"""
=========================================================
OFFLINE AI MEETING ASSISTANT

PRODUCTION AUDIO PROCESSOR

✓ DC Removal
✓ Highpass Filter
✓ Bandpass Filter
✓ Echo Suppression
✓ AGC
✓ Clipping Protection
✓ Silence Detection
✓ Normalization
✓ Resampling

=========================================================
"""

import numpy as np

from scipy.signal import butter
from scipy.signal import sosfilt
from scipy.signal import resample_poly

from config import *


class AudioProcessor:

    def __init__(self):

        self.highpass_sos = butter(
            2,
            HIGHPASS_CUTOFF,
            btype="highpass",
            fs=INPUT_SAMPLE_RATE,
            output="sos"
        )

        self.bandpass_sos = butter(
            4,
            [LOWCUT, HIGHCUT],
            btype="bandpass",
            fs=INPUT_SAMPLE_RATE,
            output="sos"
        )

    # --------------------------------------------------

    def stereo_to_mono(self, audio):

        if audio.ndim == 2:

            return np.mean(
                audio,
                axis=1
            )

        return audio

    # --------------------------------------------------

    def remove_dc(self, audio):

        return audio - np.mean(audio)

    # --------------------------------------------------

    def highpass(self, audio):

        return sosfilt(
            self.highpass_sos,
            audio
        )

    # --------------------------------------------------

    def bandpass(self, audio):

        return sosfilt(
            self.bandpass_sos,
            audio
        )

    # --------------------------------------------------

    def rms(self, audio):

        return np.sqrt(

            np.mean(

                audio ** 2

            ) + 1e-10

        )

    # --------------------------------------------------

    def agc(self, audio):

        rms = self.rms(audio)

        gain = TARGET_RMS / rms

        gain = np.clip(

            gain,

            MIN_GAIN,

            MAX_GAIN

        )

        audio = audio * gain

        return audio

    # --------------------------------------------------

    def clipping_protection(self, audio):

        peak = np.max(

            np.abs(audio)

        )

        if peak > 0.98:

            audio = audio / peak * 0.98

        return audio

    # --------------------------------------------------

    def normalize(self, audio):

        peak = np.max(

            np.abs(audio)

        )

        if peak > 0:

            audio = audio / peak

        return audio

    # --------------------------------------------------

    def silence(self, audio):

        return self.rms(audio) < SILENCE_THRESHOLD

    # --------------------------------------------------

    def echo_suppress(

        self,

        mic,

        speaker

    ):

        if speaker is None:

            return mic

        speaker = self.stereo_to_mono(
            speaker
        )

        n = min(

            len(mic),

            len(speaker)

        )

        mic = mic[:n]

        speaker = speaker[:n]

        speaker_energy = self.rms(
            speaker
        )

        mic_energy = self.rms(
            mic
        )

        if speaker_energy < AEC_MIN_SPEAKER_ENERGY:

            return mic

        ratio = mic_energy / (

            speaker_energy + 1e-8

        )

        ratio = np.clip(

            ratio,

            0.10,

            1.00

        )

        suppression = (

            1.0

            -

            AEC_SUPPRESSION

            *

            (1.0 - ratio)

        )

        return mic * suppression

    # --------------------------------------------------

    def resample(self, audio):

        return resample_poly(

            audio,

            up=1,

            down=3

        )

    # --------------------------------------------------

    def process(

        self,

        mic_audio,

        speaker_audio=None

    ):

        mic_audio = self.stereo_to_mono(
            mic_audio
        )

        mic_audio = mic_audio.astype(
            np.float32
        )

        # --------------------------------

        if ENABLE_DC_REMOVAL:

            mic_audio = self.remove_dc(
                mic_audio
            )

        # --------------------------------

        if ENABLE_HIGHPASS:

            mic_audio = self.highpass(
                mic_audio
            )

        # --------------------------------

        if ENABLE_BANDPASS:

            mic_audio = self.bandpass(
                mic_audio
            )

        # --------------------------------

        if ENABLE_AEC:

            mic_audio = self.echo_suppress(

                mic_audio,

                speaker_audio

            )

        # --------------------------------

        if ENABLE_AGC:

            mic_audio = self.agc(

                mic_audio

            )

        # --------------------------------

        if ENABLE_CLIPPING_PROTECTION:

            mic_audio = self.clipping_protection(

                mic_audio

            )

        # --------------------------------

        if ENABLE_NORMALIZATION:

            mic_audio = self.normalize(

                mic_audio

            )

        # --------------------------------

        if ENABLE_SILENCE_GATE:

            if self.silence(

                mic_audio

            ):

                return None

        # --------------------------------

        mic_audio = self.resample(

            mic_audio

        )

        return mic_audio.astype(

            np.float32

        )

    # --------------------------------------------------

    def statistics(self, audio):

        return {

            "duration_sec":

                len(audio)

                /

                WHISPER_SAMPLE_RATE,

            "rms":

                float(

                    self.rms(audio)

                ),

            "peak":

                float(

                    np.max(

                        np.abs(audio)

                    )

                ),

            "mean":

                float(

                    np.mean(audio)

                ),

            "std":

                float(

                    np.std(audio)

                )

        }

    # --------------------------------------------------

    def print_statistics(

        self,

        audio

    ):

        stats = self.statistics(

            audio

        )

        print()

        print("=" * 50)

        print("Audio Statistics")

        print("=" * 50)

        for k, v in stats.items():

            print(

                f"{k:20}: {v}"

            )

        print()