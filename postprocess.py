"""
=========================================================

OFFLINE AI MEETING ASSISTANT

POST PROCESSOR

Production Version

=========================================================
"""

import re
from difflib import SequenceMatcher

from config import *


class PostProcessor:

    def __init__(self):

        self.last_sentence = ""

        self.blocked_phrases = {

            "thank you for watching",

            "thanks for watching",

            "subscribe",

            "like and subscribe",

            "captions",

            "subtitle",

            "subtitles",

            "foreign",

            "music",

            "applause"

        }

    # --------------------------------------------------

    def similarity(

        self,

        a,

        b

    ):

        return SequenceMatcher(

            None,

            a.lower(),

            b.lower()

        ).ratio()

    # --------------------------------------------------

    def remove_noise(

        self,

        text

    ):

        text = text.lower()

        for phrase in self.blocked_phrases:

            if phrase in text:

                return ""

        return text

    # --------------------------------------------------

    def remove_duplicate_words(

        self,

        text

    ):

        words = text.split()

        output = []

        for word in words:

            if len(output) == 0:

                output.append(word)

                continue

            if output[-1].lower() != word.lower():

                output.append(word)

        return " ".join(output)

    # --------------------------------------------------

    def remove_repeated_phrases(

        self,

        text

    ):

        pattern = r'\b(.+?)\s+\1\b'

        previous = None

        while previous != text:

            previous = text

            text = re.sub(

                pattern,

                r'\1',

                text,

                flags=re.IGNORECASE

            )

        return text

    # --------------------------------------------------

    def normalize_spaces(

        self,

        text

    ):

        return re.sub(

            r"\s+",

            " ",

            text

        ).strip()

    # --------------------------------------------------

    def capitalize(

        self,

        text

    ):

        if len(text) == 0:

            return text

        return text[0].upper() + text[1:]

    # --------------------------------------------------

    def punctuation(

        self,

        text

    ):

        if len(text) == 0:

            return text

        if text[-1] not in ".?!":

            text += "."

        return text

    # --------------------------------------------------

    def duplicate_sentence(

        self,

        text

    ):

        if self.last_sentence == "":

            return False

        score = self.similarity(

            text,

            self.last_sentence

        )

        return score >= SIMILARITY_THRESHOLD

    # --------------------------------------------------

    def merge(

        self,

        previous,

        current

    ):

        previous_words = previous.split()

        current_words = current.split()

        overlap = 0

        maximum = min(

            len(previous_words),

            len(current_words),

            10

        )

        for i in range(

            maximum,

            0,

            -1

        ):

            if previous_words[-i:] == current_words[:i]:

                overlap = i

                break

        merged = (

            previous_words +

            current_words[overlap:]

        )

        return " ".join(

            merged

        )

    # --------------------------------------------------

    def process(

        self,

        text

    ):

        if text is None:

            return None

        text = self.remove_noise(

            text

        )

        if text == "":

            return None

        text = self.remove_duplicate_words(

            text

        )

        text = self.remove_repeated_phrases(

            text

        )

        text = self.normalize_spaces(

            text

        )

        text = self.capitalize(

            text

        )

        text = self.punctuation(

            text

        )

        if self.duplicate_sentence(

            text

        ):

            return None

        self.last_sentence = text

        return text

    # --------------------------------------------------

    def reset(

        self

    ):

        self.last_sentence = ""

    # --------------------------------------------------

    def merge_stream(

        self,

        previous,

        current

    ):

        if previous is None:

            return current

        if current is None:

            return previous

        return self.merge(

            previous,

            current

        )