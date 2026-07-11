"""
Configurable ASR model catalog.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ASRModelSpec:
    key: str
    provider_model: str
    compatible_engines: tuple[str, ...]
    min_ram_gb: float
    min_disk_gb: float
    preferred_compute_type: str
    quality: str
    latency: str


class ASRModelCatalog:
    def __init__(self, models: dict[str, ASRModelSpec] | None = None):
        self.models = models or DEFAULT_ASR_MODELS

    def get(self, key: str) -> ASRModelSpec:
        if key not in self.models:
            raise KeyError(f"Unknown ASR model tier: {key}")
        return self.models[key]

    def has(self, key: str) -> bool:
        return key in self.models


DEFAULT_ASR_MODELS = {
    "tiny": ASRModelSpec(
        key="tiny",
        provider_model="tiny",
        compatible_engines=("faster-whisper", "whisper.cpp", "whisperkit"),
        min_ram_gb=4,
        min_disk_gb=1,
        preferred_compute_type="int8",
        quality="low",
        latency="lowest",
    ),
    "base": ASRModelSpec(
        key="base",
        provider_model="base",
        compatible_engines=("faster-whisper", "whisper.cpp", "whisperkit"),
        min_ram_gb=8,
        min_disk_gb=2,
        preferred_compute_type="int8",
        quality="basic",
        latency="low",
    ),
    "small": ASRModelSpec(
        key="small",
        provider_model="small",
        compatible_engines=("faster-whisper", "whisper.cpp", "whisperkit"),
        min_ram_gb=16,
        min_disk_gb=4,
        preferred_compute_type="int8",
        quality="good",
        latency="medium",
    ),
    "medium": ASRModelSpec(
        key="medium",
        provider_model="medium",
        compatible_engines=("faster-whisper", "whisper.cpp", "whisperkit"),
        min_ram_gb=24,
        min_disk_gb=8,
        preferred_compute_type="int8",
        quality="high",
        latency="higher",
    ),
    "large-v3-turbo": ASRModelSpec(
        key="large-v3-turbo",
        provider_model="large-v3-turbo",
        compatible_engines=("faster-whisper", "whisper.cpp", "whisperkit"),
        min_ram_gb=24,
        min_disk_gb=10,
        preferred_compute_type="int8_float16",
        quality="very high",
        latency="gpu preferred",
    ),
    "large-v3": ASRModelSpec(
        key="large-v3",
        provider_model="large-v3",
        compatible_engines=("faster-whisper", "whisper.cpp", "whisperkit"),
        min_ram_gb=32,
        min_disk_gb=12,
        preferred_compute_type="float16",
        quality="best",
        latency="gpu required",
    ),
    "nemo-streaming-small": ASRModelSpec(
        key="nemo-streaming-small",
        provider_model="nvidia/stt_en_fastconformer_transducer_large",
        compatible_engines=("nemo",),
        min_ram_gb=16,
        min_disk_gb=8,
        preferred_compute_type="float16",
        quality="good",
        latency="true streaming",
    ),
    "nemo-streaming-large": ASRModelSpec(
        key="nemo-streaming-large",
        provider_model="nvidia/parakeet-tdt-0.6b-v2",
        compatible_engines=("nemo",),
        min_ram_gb=24,
        min_disk_gb=12,
        preferred_compute_type="float16",
        quality="very high",
        latency="true streaming",
    ),
    "cloud": ASRModelSpec(
        key="cloud",
        provider_model="cloud",
        compatible_engines=("cloud",),
        min_ram_gb=0,
        min_disk_gb=0,
        preferred_compute_type="cloud",
        quality="service configured",
        latency="network dependent",
    ),
}
