"""
=========================================================

OFFLINE AI MEETING ASSISTANT

PRODUCTION LOGGER

✓ Thread Safe
✓ Async Ready
✓ Duplicate Filter
✓ Similarity Filter
✓ Colored Console
✓ Automatic Flushing
✓ Transcript Saving

=========================================================
"""

from datetime import datetime
from threading import Lock
from difflib import SequenceMatcher

from config import *

write_lock = Lock()

_last_text = ""

_last_time = None


# ----------------------------------------------------


class Colors:

    GREEN = "\033[92m"

    BLUE = "\033[94m"

    CYAN = "\033[96m"

    YELLOW = "\033[93m"

    RED = "\033[91m"

    RESET = "\033[0m"


# ----------------------------------------------------


def initialize_logger():

    with open(

        TRANSCRIPT_FILE,

        "w",

        encoding="utf8"

    ) as f:

        f.write("")

    print()

    print("=" * 60)

    print("Transcript Initialized")

    print("=" * 60)

    print()


# ----------------------------------------------------


def timestamp():

    return datetime.now().strftime(

        LOG_TIME_FORMAT

    )


# ----------------------------------------------------


def similarity(

    a,

    b

):

    return SequenceMatcher(

        None,

        a.lower(),

        b.lower()

    ).ratio()


# ----------------------------------------------------


def is_duplicate(text):

    global _last_text

    if _last_text == "":

        return False

    score = similarity(

        text,

        _last_text

    )

    return score >= SIMILARITY_THRESHOLD


# ----------------------------------------------------


def console_color(source):

    if source == "MIC":

        return Colors.GREEN

    if source == "SYSTEM":

        return Colors.BLUE

    return Colors.CYAN


# ----------------------------------------------------


def clean_text(text):

    text = text.strip()

    while "  " in text:

        text = text.replace(

            "  ",

            " "

        )

    return text


# ----------------------------------------------------


def print_console(

    source,

    language,

    text

):

    if not CONSOLE_OUTPUT:

        return

    color = console_color(

        source

    )

    now = timestamp()

    print(

        color

        +

        f"[{now}] "

        +

        f"[{source}] "

        +

        f"[{language}] "

        +

        text

        +

        Colors.RESET

    )


# ----------------------------------------------------


def save(

    source,

    language,

    text

):

    now = timestamp()

    line = ""

    if SAVE_TIMESTAMP:

        line += f"[{now}] "

    line += f"[{source}] "

    if SAVE_LANGUAGE:

        line += f"[{language}] "

    line += text

    line += "\n"

    with open(

        TRANSCRIPT_FILE,

        "a",

        encoding="utf8"

    ) as f:

        f.write(line)

        f.flush()


# ----------------------------------------------------


def log(

    source,

    language,

    text

):

    global _last_text

    text = clean_text(

        text

    )

    if len(text) == 0:

        return

    if len(

        text.split()

    ) < MINIMUM_WORDS:

        return

    if text.lower() in IGNORE_TEXT:

        return

    if ENABLE_DUPLICATE_FILTER:

        if is_duplicate(

            text

        ):

            return

    with write_lock:

        print_console(

            source,

            language,

            text

        )

        save(

            source,

            language,

            text

        )

    _last_text = text


# ----------------------------------------------------


def separator():

    with write_lock:

        with open(

            TRANSCRIPT_FILE,

            "a",

            encoding="utf8"

        ) as f:

            f.write(

                "\n"

                +

                "=" * 60

                +

                "\n\n"

            )


# ----------------------------------------------------


def info(message):

    print(

        Colors.CYAN

        +

        "[INFO] "

        +

        message

        +

        Colors.RESET

    )


# ----------------------------------------------------


def warning(message):

    print(

        Colors.YELLOW

        +

        "[WARNING] "

        +

        message

        +

        Colors.RESET

    )


# ----------------------------------------------------


def error(message):

    print(

        Colors.RED

        +

        "[ERROR] "

        +

        message

        +

        Colors.RESET

    )


# ----------------------------------------------------


def success(message):

    print(

        Colors.GREEN

        +

        "[SUCCESS] "

        +

        message

        +

        Colors.RESET

    )


# ----------------------------------------------------


def banner():

    print()

    print("=" * 60)

    print(APP_NAME)

    print(VERSION)

    print("=" * 60)

    print()