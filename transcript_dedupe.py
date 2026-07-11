"""
Transcript-level duplicate suppression across audio sources.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
import time


@dataclass
class TranscriptEvent:
    source: str
    text: str
    timestamp: float


class CrossSourceDedupe:
    def __init__(self, threshold: float = 0.88, window_seconds: float = 8.0):
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.events: list[TranscriptEvent] = []

    def should_emit(self, source: str, text: str) -> bool:
        now = time.time()
        self.events = [
            event
            for event in self.events
            if now - event.timestamp <= self.window_seconds
        ]

        normalized = self._normalize(text)
        for event in self.events:
            if event.source == source:
                continue
            if self._similarity(normalized, self._normalize(event.text)) >= self.threshold:
                return False

        self.events.append(TranscriptEvent(source=source, text=text, timestamp=now))
        return True

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        return SequenceMatcher(None, left, right).ratio()
