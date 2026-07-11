"""
Common ASR engine interface and current Faster-Whisper adapter.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptEvent:
    language: str | None
    text: str | None


class ASREngine(ABC):
    supports_streaming: bool = False

    @abstractmethod
    def prepare(self):
        raise NotImplementedError

    @abstractmethod
    def transcribe_chunk(self, audio, source: str, sample_rate: int = 16000) -> TranscriptEvent:
        raise NotImplementedError


class FasterWhisperEngine(ASREngine):
    supports_streaming = False

    def __init__(self, transcriber=None):
        self.transcriber = transcriber

    def prepare(self):
        if self.transcriber is None:
            from transcriber import Transcriber

            self.transcriber = Transcriber()
        return self

    def transcribe_chunk(self, audio, source: str, sample_rate: int = 16000) -> TranscriptEvent:
        self.prepare()
        language, text = self.transcriber.process(audio)
        return TranscriptEvent(language=language, text=text)


class PendingASREngine(ASREngine):
    def __init__(self, engine_key: str, setup_hint: str, supports_streaming: bool = False):
        self.engine_key = engine_key
        self.setup_hint = setup_hint
        self.supports_streaming = supports_streaming

    def prepare(self):
        raise NotImplementedError(
            f"ASR engine '{self.engine_key}' is selected by policy but is not implemented. "
            f"{self.setup_hint}"
        )

    def transcribe_chunk(self, audio, source: str, sample_rate: int = 16000) -> TranscriptEvent:
        self.prepare()
        return TranscriptEvent(language=None, text=None)


class CloudASREngine(PendingASREngine):
    pass
