"""
=========================================================

OFFLINE AI MEETING ASSISTANT

UTILITY FUNCTIONS

Production Version

=========================================================
"""

import re
import numpy as np

from scipy.signal import resample_poly


# --------------------------------------------------------
# Audio Utilities
# --------------------------------------------------------

def stereo_to_mono(audio):

    if audio is None:
        return None

    if audio.ndim == 2:
        return np.mean(audio, axis=1)

    return audio


# --------------------------------------------------------

def normalize(audio):

    peak = np.max(np.abs(audio))

    if peak < 1e-8:
        return audio.astype(np.float32)

    audio = audio / peak

    return audio.astype(np.float32)


# --------------------------------------------------------

def rms(audio):

    return np.sqrt(

        np.mean(

            np.square(audio)

        )

        + 1e-10

    )


# --------------------------------------------------------

def dbfs(audio):

    value = rms(audio)

    return 20 * np.log10(

        value + 1e-10

    )


# --------------------------------------------------------

def remove_dc(audio):

    return audio - np.mean(audio)


# --------------------------------------------------------

def clipping(audio):

    return np.max(

        np.abs(audio)

    ) >= 0.99


# --------------------------------------------------------

def resample(audio):

    return resample_poly(

        audio,

        up=1,

        down=3

    ).astype(np.float32)


# --------------------------------------------------------
# Text Utilities
# --------------------------------------------------------

def clean_text(text):

    text = text.strip()

    text = re.sub(

        r"\s+",

        " ",

        text

    )

    return text


# --------------------------------------------------------

def remove_duplicate_words(text):

    words = text.split()

    if len(words) <= 1:

        return text

    output = []

    for word in words:

        if len(output) == 0:

            output.append(word)

            continue

        if output[-1].lower() != word.lower():

            output.append(word)

    return " ".join(output)


# --------------------------------------------------------

def capitalize(text):

    if len(text) == 0:

        return text

    return text[0].upper() + text[1:]


# --------------------------------------------------------

def ensure_period(text):

    text = text.strip()

    if len(text) == 0:

        return text

    if text[-1] not in ".?!":

        text += "."

    return text


# --------------------------------------------------------

def preprocess_transcript(text):

    text = clean_text(text)

    text = remove_duplicate_words(text)

    text = capitalize(text)

    text = ensure_period(text)

    return text


# --------------------------------------------------------
# Similarity
# --------------------------------------------------------

def jaccard_similarity(a, b):

    a = set(

        a.lower().split()

    )

    b = set(

        b.lower().split()

    )

    if len(a | b) == 0:

        return 0

    return len(a & b) / len(a | b)


# --------------------------------------------------------

def cosine_like_similarity(a, b):

    a = a.lower().split()

    b = b.lower().split()

    common = len(

        set(a) & set(b)

    )

    total = max(

        len(a),

        len(b)

    )

    if total == 0:

        return 0

    return common / total


# --------------------------------------------------------
# Audio Statistics
# --------------------------------------------------------

def audio_statistics(audio):

    return {

        "samples": len(audio),

        "duration":

            round(

                len(audio) / 16000,

                2

            ),

        "peak":

            float(

                np.max(

                    np.abs(audio)

                )

            ),

        "rms":

            float(

                rms(audio)

            ),

        "dbfs":

            float(

                dbfs(audio)

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


# --------------------------------------------------------

def print_audio_statistics(audio):

    stats = audio_statistics(audio)

    print()

    print("=" * 50)

    print("Audio Statistics")

    print("=" * 50)

    for key, value in stats.items():

        print(

            f"{key:15}: {value}"

        )

    print()


# --------------------------------------------------------
# Timer
# --------------------------------------------------------

class Timer:

    def __init__(self):

        import time

        self.time = time

        self.start_time = self.time.time()

    def reset(self):

        self.start_time = self.time.time()

    def elapsed(self):

        return self.time.time() - self.start_time