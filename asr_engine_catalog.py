"""
Configurable ASR engine catalog.
"""

from dataclasses import dataclass


ENGINE_FASTER_WHISPER = "faster-whisper"
ENGINE_WHISPER_CPP = "whisper.cpp"
ENGINE_WHISPERKIT = "whisperkit"
ENGINE_NEMO = "nemo"
ENGINE_CLOUD = "cloud"


@dataclass(frozen=True)
class ASREngineSpec:
    key: str
    label: str
    local: bool
    implemented: bool
    supports_streaming: bool
    recommended_for: str
    setup_hint: str


class ASREngineCatalog:
    def __init__(self, engines: dict[str, ASREngineSpec] | None = None):
        self.engines = engines or DEFAULT_ASR_ENGINES

    def get(self, key: str) -> ASREngineSpec:
        if key not in self.engines:
            raise KeyError(f"Unknown ASR engine: {key}")
        return self.engines[key]

    def has(self, key: str) -> bool:
        return key in self.engines


DEFAULT_ASR_ENGINES = {
    ENGINE_FASTER_WHISPER: ASREngineSpec(
        key=ENGINE_FASTER_WHISPER,
        label="Faster-Whisper",
        local=True,
        implemented=True,
        supports_streaming=False,
        recommended_for="current Python local ASR path on CPU/CUDA",
        setup_hint="Install faster-whisper/CTranslate2 dependencies.",
    ),
    ENGINE_WHISPER_CPP: ASREngineSpec(
        key=ENGINE_WHISPER_CPP,
        label="whisper.cpp",
        local=True,
        implemented=False,
        supports_streaming=False,
        recommended_for="macOS Apple Silicon Metal and non-NVIDIA local Whisper paths",
        setup_hint="Bundle/build whisper.cpp and ggml model files before enabling.",
    ),
    ENGINE_WHISPERKIT: ASREngineSpec(
        key=ENGINE_WHISPERKIT,
        label="WhisperKit",
        local=True,
        implemented=False,
        supports_streaming=False,
        recommended_for="future native macOS Apple Silicon CoreML path",
        setup_hint="Add a Swift/CoreML WhisperKit adapter before enabling.",
    ),
    ENGINE_NEMO: ASREngineSpec(
        key=ENGINE_NEMO,
        label="NVIDIA NeMo",
        local=True,
        implemented=False,
        supports_streaming=True,
        recommended_for="Windows machines with NVIDIA CUDA GPUs",
        setup_hint="Install/bundle NeMo, PyTorch CUDA, and a supported streaming checkpoint.",
    ),
    ENGINE_CLOUD: ASREngineSpec(
        key=ENGINE_CLOUD,
        label="Cloud ASR",
        local=False,
        implemented=False,
        supports_streaming=True,
        recommended_for="weak local hardware or policy fallback",
        setup_hint="Wire the hosted ASR client before enabling cloud mode.",
    ),
}
