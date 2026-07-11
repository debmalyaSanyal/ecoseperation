"""
=========================================================
OFFLINE AI MEETING ASSISTANT

CONFIGURATION FILE
Production Optimized Version
=========================================================
"""

# =========================================================
# AUDIO DEVICES
# =========================================================

# Run sounddevice.query_devices()
# to find your microphone and loopback devices.

MIC_DEVICE = 9
SYSTEM_DEVICE = 11

# =========================================================
# AUDIO SETTINGS
# =========================================================

INPUT_SAMPLE_RATE = 48000

# Whisper works at 16 kHz
WHISPER_SAMPLE_RATE = 16000

MIC_CHANNELS = 1
SYSTEM_CHANNELS = 2

AUDIO_DTYPE = "float32"

# =========================================================
# STREAM SETTINGS
# =========================================================

# Lower = lower latency
# Higher = better accuracy

BUFFER_SECONDS = 3

BLOCK_SIZE = 2048

QUEUE_SIZE = 64

LOOP_SLEEP = 0.02

# =========================================================
# AUDIO PREPROCESSING
# =========================================================

ENABLE_DC_REMOVAL = True

ENABLE_HIGHPASS = True

ENABLE_BANDPASS = False

ENABLE_NORMALIZATION = True

ENABLE_AGC = True

ENABLE_AEC = True

ENABLE_NOISE_REDUCTION = False

ENABLE_SILENCE_GATE = False

ENABLE_VAD = True

ENABLE_CLIPPING_PROTECTION = True

# =========================================================
# HIGH PASS FILTER
# =========================================================

HIGHPASS_CUTOFF = 80

# =========================================================
# BANDPASS FILTER
# =========================================================

LOWCUT = 80

HIGHCUT = 7600

# =========================================================
# AGC SETTINGS
# =========================================================

TARGET_RMS = 0.12

MIN_GAIN = 0.6

MAX_GAIN = 3.5

ATTACK_TIME = 0.02

RELEASE_TIME = 0.20

# =========================================================
# SILENCE DETECTION
# =========================================================

SILENCE_THRESHOLD = 0.008

MIN_SPEECH_SECONDS = 0.50

MAX_SILENCE_SECONDS = 0.80

# =========================================================
# VOICE ACTIVITY DETECTION
# =========================================================

# 0-3
# 3 = most aggressive

WEBRTC_VAD_MODE = 3

# =========================================================
# ECHO CANCELLATION
# =========================================================

AEC_SUPPRESSION = 0.85

AEC_MIN_SPEAKER_ENERGY = 0.02

# =========================================================
# WHISPER SETTINGS
# =========================================================

WHISPER_MODEL = "large-v3-turbo"

WHISPER_DEVICE = "cuda"

WHISPER_COMPUTE_TYPE = "int8_float16"

BEAM_SIZE = 7

BEST_OF = 7

TEMPERATURE = 0.0

PATIENCE = 2

LENGTH_PENALTY = 1.0

USE_VAD_FILTER = True

CONDITION_ON_PREVIOUS_TEXT = True

WITHOUT_TIMESTAMPS = True

WORD_TIMESTAMPS = False

# =========================================================
# LANGUAGE SETTINGS
# =========================================================

AUTO_LANGUAGE = True

LOCK_LANGUAGE_AFTER_FIRST_SEGMENT = True

DEFAULT_LANGUAGE = None

LANGUAGE_CONFIDENCE = 0.95

# =========================================================
# DUPLICATE FILTER
# =========================================================

ENABLE_DUPLICATE_FILTER = True

SIMILARITY_THRESHOLD = 0.92

# =========================================================
# TEXT FILTER
# =========================================================

MINIMUM_WORDS = 2

MINIMUM_CHARACTERS = 4

IGNORE_TEXT = {

    "thank you",

    "thanks",

    "bye",

    "goodbye",

    "music",

    "subtitle",

    "subtitles",

    "captions",

    "caption",

    "foreign",

    "applause",

    "cheering"

}

# =========================================================
# LOGGING
# =========================================================

TRANSCRIPT_FILE = "transcript.txt"

SAVE_TIMESTAMP = True

SAVE_LANGUAGE = True

CONSOLE_OUTPUT = True

LOG_TIME_FORMAT = "%H:%M:%S"

# =========================================================
# THREADING
# =========================================================

MIC_QUEUE_SIZE = 32

SYSTEM_QUEUE_SIZE = 32

WHISPER_WORKERS = 1

ENABLE_ASYNC_LOGGER = True

# =========================================================
# PERFORMANCE
# =========================================================

USE_FP16 = True

USE_CUDA = True

PIN_MEMORY = True

GPU_BATCH_SIZE = 8

# =========================================================
# FUTURE FEATURES
# =========================================================

ENABLE_SPEAKER_DIARIZATION = False

ENABLE_TRANSLATION = False

ENABLE_MEETING_SUMMARY = False

ENABLE_ACTION_ITEMS = False

ENABLE_KEYWORD_EXTRACTION = False

ENABLE_NAMED_ENTITY_RECOGNITION = False

ENABLE_SENTIMENT_ANALYSIS = False

ENABLE_VECTOR_DATABASE = False

ENABLE_REALTIME_SEARCH = False

ENABLE_PDF_EXPORT = False

ENABLE_DOCX_EXPORT = False

# =========================================================
# DEBUG
# =========================================================

DEBUG = False

PRINT_AUDIO_STATS = False

PRINT_QUEUE_STATS = False

PRINT_GPU_STATS = False

PRINT_PROCESSING_TIME = False

PRINT_WHISPER_TIME = False

# =========================================================
# VERSION
# =========================================================

APP_NAME = "Offline AI Meeting Assistant"

VERSION = "2.0 Production"

AUTHOR = "Debmalya Sanyal"