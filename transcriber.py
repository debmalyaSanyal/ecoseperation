"""
=========================================================
OFFLINE AI MEETING ASSISTANT

PRODUCTION TRANSCRIBER

Powered by Faster-Whisper

=========================================================
"""

import re

from faster_whisper import WhisperModel

from config import *


class Transcriber:

    def __init__(self):

        print()

        print("=" * 60)
        print("Loading Whisper Model...")
        print("=" * 60)

        self.model = WhisperModel(

            WHISPER_MODEL,

            device=WHISPER_DEVICE,

            compute_type=WHISPER_COMPUTE_TYPE,

            cpu_threads=4,

            num_workers=1

        )

        self.locked_language = None

        self.previous_text = ""

    # -------------------------------------------------

    def clean_text(self, text):

        text = text.strip()

        text = re.sub(

            r"\s+",

            " ",

            text

        )

        text = re.sub(

            r"(.)\1{5,}",

            r"\1",

            text

        )

        return text

    # -------------------------------------------------

    def remove_duplicate_words(self, text):

        words = text.split()

        if len(words) <= 1:

            return text

        cleaned = []

        for word in words:

            if len(cleaned) == 0:

                cleaned.append(word)

            elif cleaned[-1].lower() != word.lower():

                cleaned.append(word)

        return " ".join(cleaned)

    # -------------------------------------------------

    def hallucination_filter(self, text):

        text_low = text.lower()

        blocked = [

            "thank you for watching",

            "thanks for watching",

            "subscribe",

            "like and subscribe",

            "captions by",

            "subtitle",

            "subtitles",

            "music",

            "foreign",

            "www",

            "youtube",

            "facebook",

            "instagram"

        ]

        for phrase in blocked:

            if phrase in text_low:

                return ""

        return text

    # -------------------------------------------------

    def similarity(self, a, b):

        a = a.lower()

        b = b.lower()

        sa = set(a.split())

        sb = set(b.split())

        if len(sa | sb) == 0:

            return 0

        return len(sa & sb) / len(sa | sb)

    # -------------------------------------------------

    def decode(self, audio):

        language = self.locked_language

        segments, info = self.model.transcribe(

            audio,

            beam_size=BEAM_SIZE,

            best_of=BEST_OF,

            patience=PATIENCE,

            length_penalty=LENGTH_PENALTY,

            temperature=TEMPERATURE,

            vad_filter=USE_VAD_FILTER,

            condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,

            word_timestamps=WORD_TIMESTAMPS,

            without_timestamps=WITHOUT_TIMESTAMPS,

            language=language

        )

        text = ""

        for segment in segments:

            text += segment.text + " "

        text = text.strip()

        detected_language = info.language

        probability = info.language_probability

        return detected_language, probability, text

    # -------------------------------------------------

    def process(self, audio):

        try:

            language, confidence, text = self.decode(audio)

        except Exception:

            return None, None

        if AUTO_LANGUAGE:

            if LOCK_LANGUAGE_AFTER_FIRST_SEGMENT:

                if self.locked_language is None:

                    if confidence >= LANGUAGE_CONFIDENCE:

                        self.locked_language = language

                else:

                    language = self.locked_language

        text = self.clean_text(text)

        text = self.remove_duplicate_words(text)

        text = self.hallucination_filter(text)

        if len(text) < MINIMUM_CHARACTERS:

            return None, None

        if ENABLE_DUPLICATE_FILTER:

            score = self.similarity(

                text,

                self.previous_text

            )

            if score > SIMILARITY_THRESHOLD:

                return None, None

        self.previous_text = text

        return language, text

    # -------------------------------------------------

    def reset(self):

        self.previous_text = ""

        self.locked_language = None

    # -------------------------------------------------

    def get_language(self):

        return self.locked_language

    # -------------------------------------------------

    def unlock_language(self):

        self.locked_language = None

    # -------------------------------------------------

    def print_status(self):

        print()

        print("=" * 50)
        print("Whisper Status")
        print("=" * 50)
        print("Model     :", WHISPER_MODEL)
        print("Device    :", WHISPER_DEVICE)
        print("Language  :", self.locked_language)
        print("=" * 50)
        print()