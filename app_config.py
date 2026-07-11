"""
Runtime configuration resolution for the local ASR CLI.
"""

from dataclasses import dataclass
from pathlib import Path
import os
import platform
import sys

from asr_decision_engine import ASRDecisionEngine
from capture_backend import CaptureBackendResolver
from system_profiler import SystemProfile, SystemProfiler


DEFAULT_MODEL = "small"
QUALITY_MODEL = "large-v3-turbo"
MODEL_CACHE_DIR = Path("models") / "whisper"
CAPTURE_MODE_NATIVE = "native"
CAPTURE_MODE_SOUNDDEVICE = "sounddevice"


@dataclass(frozen=True)
class RuntimeConfig:
    os_name: str
    python_version: tuple[int, int, int]
    model_name: str
    model_key: str
    asr_engine: str
    asr_engine_label: str
    asr_engine_implemented: bool
    asr_engine_supports_streaming: bool
    asr_engine_setup_hint: str
    asr_mode: str
    asr_reason: str
    model_cache_dir: Path
    whisper_device: str
    whisper_compute_type: str
    hardware_profile: SystemProfile | None = None
    input_sample_rate: int = 48000
    whisper_sample_rate: int = 16000
    mic_channels: int = 1
    system_channels: int = 2
    vad_enabled: bool = True
    echo_strategy: str = "echo mitigation: separate streams + VAD + cross-source dedupe"
    capture_mode: str = CAPTURE_MODE_SOUNDDEVICE
    capture_backend: str = "sounddevice"
    capture_backend_reason: str = ""
    native_description: str = ""
    native_helper_dir: Path | None = None
    native_helper_binary: Path | None = None
    native_helper_run_command: tuple[str, ...] = ()
    native_ws_host: str = "127.0.0.1"
    native_ws_port: int = 8089
    native_chunk_frames: int = 4096
    native_diagnostics: bool = False
    native_rms_logging: bool = True
    native_window_seconds: float = 1.25
    native_overlap_seconds: float = 0.25
    native_utterance_mode: bool = True
    native_utterance_rms_threshold: float = 80.0
    native_utterance_low_packet_target: int = 3
    native_utterance_min_seconds: float = 0.35
    native_utterance_max_seconds: float = 12.0

    @classmethod
    def resolve(
        cls,
        os_name: str | None = None,
        selected_model: str | None = None,
        profile: SystemProfile | None = None,
    ) -> "RuntimeConfig":
        resolved_profile = profile or SystemProfiler.profile_current()
        if os_name is not None and os_name != resolved_profile.os_name:
            resolved_profile = SystemProfile(
                os_name=os_name,
                os_version=resolved_profile.os_version,
                architecture=resolved_profile.architecture,
                cpu_count=resolved_profile.cpu_count,
                total_ram_gb=resolved_profile.total_ram_gb,
                available_disk_gb=resolved_profile.available_disk_gb,
                is_apple_silicon=os_name == "Darwin" and resolved_profile.is_apple_silicon,
                gpus=resolved_profile.gpus,
            )

        capture_backend = CaptureBackendResolver().resolve(resolved_profile)
        asr_decision = ASRDecisionEngine().decide(resolved_profile, selected_model)

        return cls(
            os_name=resolved_profile.os_name,
            python_version=sys.version_info[:3],
            model_name=asr_decision.model.provider_model,
            model_key=asr_decision.model_key,
            asr_engine=asr_decision.engine_key,
            asr_engine_label=asr_decision.engine.label,
            asr_engine_implemented=asr_decision.engine.implemented,
            asr_engine_supports_streaming=asr_decision.engine.supports_streaming,
            asr_engine_setup_hint=asr_decision.engine.setup_hint,
            asr_mode=asr_decision.mode,
            asr_reason=asr_decision.reason,
            model_cache_dir=MODEL_CACHE_DIR,
            whisper_device=os.environ.get("LOCAL_ASR_DEVICE", asr_decision.device),
            whisper_compute_type=os.environ.get(
                "LOCAL_ASR_COMPUTE_TYPE",
                asr_decision.compute_type,
            ),
            hardware_profile=resolved_profile,
            capture_mode=capture_backend.capture_mode,
            capture_backend=capture_backend.backend,
            capture_backend_reason=capture_backend.reason,
            native_description=capture_backend.description,
            native_helper_dir=capture_backend.helper_dir,
            native_helper_binary=capture_backend.helper_binary,
            native_helper_run_command=capture_backend.helper_run_command,
            native_ws_host=os.environ.get("LOCAL_ASR_NATIVE_WS_HOST", "127.0.0.1"),
            native_ws_port=cls._int_env("LOCAL_ASR_NATIVE_WS_PORT", 8089),
            native_chunk_frames=cls._int_env("LOCAL_ASR_NATIVE_CHUNK_FRAMES", 4096),
            native_diagnostics=cls._bool_env("LOCAL_ASR_NATIVE_DIAGNOSTICS", False),
            native_rms_logging=cls._bool_env("LOCAL_ASR_NATIVE_RMS_LOGGING", True),
            native_window_seconds=cls._float_env(
                "LOCAL_ASR_NATIVE_WINDOW_SECONDS",
                1.25,
            ),
            native_overlap_seconds=cls._float_env(
                "LOCAL_ASR_NATIVE_OVERLAP_SECONDS",
                0.25,
            ),
            native_utterance_mode=cls._bool_env(
                "LOCAL_ASR_UTTERANCE_MODE",
                True,
            ),
            native_utterance_rms_threshold=cls._float_env(
                "LOCAL_ASR_UTTERANCE_RMS_THRESHOLD",
                80.0,
            ),
            native_utterance_low_packet_target=cls._int_env(
                "LOCAL_ASR_UTTERANCE_LOW_PACKET_TARGET",
                3,
            ),
            native_utterance_min_seconds=cls._float_env(
                "LOCAL_ASR_UTTERANCE_MIN_SECONDS",
                0.35,
            ),
            native_utterance_max_seconds=cls._float_env(
                "LOCAL_ASR_UTTERANCE_MAX_SECONDS",
                12.0,
            ),
        )

    @staticmethod
    def _int_env(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _bool_env(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _float_env(name: str, default: float) -> float:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default

    @property
    def python_supported(self) -> bool:
        major, minor, _ = self.python_version
        return major == 3 and 10 <= minor <= 12

    @property
    def python_warning(self) -> str | None:
        if self.python_supported:
            return None
        major, minor, patch = self.python_version
        return (
            f"Python {major}.{minor}.{patch} detected. This project recommends "
            "Python 3.10-3.12 because audio/ML wheels are more reliable there."
        )

    def apply_to_legacy_config(self) -> None:
        import config

        config.MIC_DEVICE = None
        config.SYSTEM_DEVICE = None
        config.WHISPER_MODEL = self.model_name
        config.WHISPER_DEVICE = self.whisper_device
        config.WHISPER_COMPUTE_TYPE = self.whisper_compute_type
        config.ENABLE_VAD = self.vad_enabled
        config.USE_CUDA = self.whisper_device == "cuda"
        config.USE_FP16 = self.whisper_compute_type in {"float16", "int8_float16"}

    def apply_native_low_latency_config(self) -> None:
        import config

        config.BEAM_SIZE = 1
        config.BEST_OF = 1
        config.PATIENCE = 1
        config.CONDITION_ON_PREVIOUS_TEXT = False
        config.USE_VAD_FILTER = True


def apply_device_config(mic_device: int, system_device: int) -> None:
    import config

    config.MIC_DEVICE = mic_device
    config.SYSTEM_DEVICE = system_device
