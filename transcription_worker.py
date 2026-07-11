"""
Shared VAD, ASR, post-processing, and transcript logging.
"""

from transcript_dedupe import CrossSourceDedupe
import time


class TranscriptionWorker:
    def __init__(self, diagnostics: bool = False, asr_engine=None):
        from postprocess import PostProcessor
        from asr_engines import FasterWhisperEngine
        from vad import VoiceActivityDetector

        self.diagnostics = diagnostics
        self.vad = VoiceActivityDetector()
        self.asr_engine = asr_engine or FasterWhisperEngine().prepare()
        self.post = PostProcessor()
        self.dedupe = CrossSourceDedupe()

    def handle(self, source: str, audio):
        from logger import info, log

        if audio is None:
            return

        started = time.perf_counter()
        speech = self.vad.process(audio)
        if speech is None:
            if self.diagnostics:
                elapsed = time.perf_counter() - started
                info(f"ASR skipped source={source} reason=no_speech elapsed={elapsed:.2f}s")
            return

        event = self.asr_engine.transcribe_chunk(speech, source, sample_rate=16000)
        language, text = event.language, event.text
        if not text:
            if self.diagnostics:
                elapsed = time.perf_counter() - started
                info(f"ASR skipped source={source} reason=no_text elapsed={elapsed:.2f}s")
            return

        text = self.post.process(text)
        if text and self.dedupe.should_emit(source, text):
            log(source, language or "unk", text)
            if self.diagnostics:
                elapsed = time.perf_counter() - started
                info(
                    "ASR emitted "
                    f"source={source} duration={len(audio) / 16000:.2f}s "
                    f"elapsed={elapsed:.2f}s"
                )
