"""
Hardware-aware ASR model and mode selection.
"""

from dataclasses import dataclass
import os

from asr_engine_catalog import (
    ASREngineCatalog,
    ASREngineSpec,
    ENGINE_CLOUD,
    ENGINE_FASTER_WHISPER,
    ENGINE_NEMO,
    ENGINE_WHISPER_CPP,
    ENGINE_WHISPERKIT,
)
from asr_model_catalog import ASRModelCatalog, ASRModelSpec
from system_profiler import SystemProfile


@dataclass(frozen=True)
class ASRDecision:
    mode: str
    engine_key: str
    engine: ASREngineSpec
    model_key: str
    model: ASRModelSpec
    device: str
    compute_type: str
    reason: str


class ASRDecisionEngine:
    def __init__(
        self,
        catalog: ASRModelCatalog | None = None,
        engine_catalog: ASREngineCatalog | None = None,
    ):
        self.catalog = catalog or ASRModelCatalog()
        self.engine_catalog = engine_catalog or ASREngineCatalog()

    def decide(self, profile: SystemProfile, override: str | None = None) -> ASRDecision:
        mode = os.environ.get("ASR_MODEL_MODE", "auto").strip().lower()
        engine_mode = os.environ.get("ASR_ENGINE_MODE", "auto").strip().lower()
        engine_override = os.environ.get("ASR_ENGINE_OVERRIDE")
        override = override or os.environ.get("ASR_MODEL_OVERRIDE") or os.environ.get("LOCAL_ASR_MODEL")
        cloud_fallback = self._bool_env("ASR_CLOUD_FALLBACK", True)

        if engine_mode == "manual" and not engine_override:
            raise ValueError("ASR_ENGINE_MODE=manual requires ASR_ENGINE_OVERRIDE.")

        if engine_override:
            engine_key = engine_override.strip()
            engine = self.engine_catalog.get(engine_key)
            key = override.strip() if override else self._default_model_for_engine(engine_key, profile)
            if engine_key == ENGINE_CLOUD:
                key = "cloud"
            model = self.catalog.get(key)
            self._validate_engine_model(engine_key, model)
            selected_mode = "cloud" if engine_key == ENGINE_CLOUD else "local"
            return self._decision(
                selected_mode,
                engine_key,
                engine,
                key,
                model,
                profile,
                f"manual engine override selected {engine_key}"
                + (f" with model {key}" if override else ""),
            )

        if override:
            key = override.strip()
            model = self.catalog.get(key)
            engine_key = self._engine_for_model_override(model, profile)
            engine = self.engine_catalog.get(engine_key)
            selected_mode = "cloud" if engine_key == ENGINE_CLOUD else "local"
            return self._decision(
                selected_mode,
                engine_key,
                engine,
                key,
                model,
                profile,
                f"manual model override selected {key}",
            )

        if mode == "cloud":
            model = self.catalog.get("cloud")
            engine = self.engine_catalog.get(ENGINE_CLOUD)
            return self._decision(
                "cloud",
                ENGINE_CLOUD,
                engine,
                "cloud",
                model,
                profile,
                "ASR_MODEL_MODE=cloud",
            )

        engine_key, key = self._auto_engine_and_model(profile, cloud_fallback)
        selected_mode = "cloud" if engine_key == ENGINE_CLOUD else "local"
        engine = self.engine_catalog.get(engine_key)
        model = self.catalog.get(key)
        return self._decision(
            selected_mode,
            engine_key,
            engine,
            key,
            model,
            profile,
            self._reason_for(profile, key),
        )

    def _auto_engine_and_model(self, profile: SystemProfile, cloud_fallback: bool) -> tuple[str, str]:
        if profile.total_ram_gb and profile.total_ram_gb < 8:
            return (ENGINE_CLOUD, "cloud") if cloud_fallback else (ENGINE_FASTER_WHISPER, "tiny")
        if profile.available_disk_gb and profile.available_disk_gb < 2:
            return (ENGINE_CLOUD, "cloud") if cloud_fallback else (ENGINE_FASTER_WHISPER, "tiny")

        if profile.os_name == "Windows" and profile.max_cuda_vram_gb >= 10:
            return ENGINE_NEMO, "nemo-streaming-large"
        if profile.os_name == "Windows" and profile.max_cuda_vram_gb >= 6:
            return ENGINE_NEMO, "nemo-streaming-small"

        if profile.os_name == "Darwin" and profile.is_apple_silicon:
            if profile.total_ram_gb >= 24 and profile.cpu_count >= 8:
                return ENGINE_WHISPER_CPP, "medium"
            if profile.total_ram_gb >= 16:
                return ENGINE_WHISPER_CPP, "small"
            return ENGINE_WHISPER_CPP, "base"

        if profile.os_name == "Windows":
            if profile.total_ram_gb >= 16:
                return ENGINE_WHISPER_CPP, "small"
            return ENGINE_WHISPER_CPP, "base"

        if profile.total_ram_gb >= 24 and profile.cpu_count >= 8:
            return ENGINE_FASTER_WHISPER, "medium"
        if profile.total_ram_gb >= 16:
            return ENGINE_FASTER_WHISPER, "small"
        if profile.total_ram_gb >= 8:
            return ENGINE_FASTER_WHISPER, "base"
        return ENGINE_FASTER_WHISPER, "tiny"

    def _decision(
        self,
        mode: str,
        engine_key: str,
        engine: ASREngineSpec,
        key: str,
        model: ASRModelSpec,
        profile: SystemProfile,
        reason: str,
    ) -> ASRDecision:
        if mode == "cloud":
            return ASRDecision(mode, engine_key, engine, key, model, "cloud", "cloud", reason)
        device = self._device_for(engine_key, profile)
        compute_type = model.preferred_compute_type if device == "cuda" else "int8"
        if engine_key == ENGINE_WHISPER_CPP and profile.os_name == "Darwin" and profile.is_apple_silicon:
            device = "metal"
            compute_type = "ggml"
        if engine_key == ENGINE_WHISPERKIT:
            device = "coreml"
            compute_type = "coreml"
        return ASRDecision(mode, engine_key, engine, key, model, device, compute_type, reason)

    def _reason_for(self, profile: SystemProfile, key: str) -> str:
        return (
            f"auto selected {key} for os={profile.os_name} "
            f"ram={profile.total_ram_gb:.1f}GB cpu={profile.cpu_count} "
            f"cuda_vram={profile.max_cuda_vram_gb:.1f}GB"
        )

    def _device_for(self, engine_key: str, profile: SystemProfile) -> str:
        if engine_key in {ENGINE_NEMO, ENGINE_FASTER_WHISPER}:
            if profile.os_name == "Windows" and profile.has_cuda_gpu:
                return "cuda"
        return "cpu"

    def _default_model_for_engine(self, engine_key: str, profile: SystemProfile) -> str:
        if engine_key == ENGINE_CLOUD:
            return "cloud"
        if engine_key == ENGINE_NEMO:
            return "nemo-streaming-large" if profile.max_cuda_vram_gb >= 10 else "nemo-streaming-small"
        if profile.total_ram_gb >= 24 and profile.cpu_count >= 8:
            return "medium"
        if profile.total_ram_gb >= 16:
            return "small"
        if profile.total_ram_gb >= 8:
            return "base"
        return "tiny"

    def _engine_for_model_override(self, model: ASRModelSpec, profile: SystemProfile) -> str:
        if model.key == "cloud":
            return ENGINE_CLOUD
        if model.compatible_engines == (ENGINE_NEMO,):
            return ENGINE_NEMO
        if profile.os_name == "Darwin" and profile.is_apple_silicon and ENGINE_WHISPER_CPP in model.compatible_engines:
            return ENGINE_WHISPER_CPP
        if profile.os_name == "Windows" and not profile.has_cuda_gpu and ENGINE_WHISPER_CPP in model.compatible_engines:
            return ENGINE_WHISPER_CPP
        return ENGINE_FASTER_WHISPER

    def _validate_engine_model(self, engine_key: str, model: ASRModelSpec) -> None:
        if engine_key not in model.compatible_engines:
            raise ValueError(f"Model '{model.key}' is not compatible with ASR engine '{engine_key}'.")

    @staticmethod
    def _bool_env(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
