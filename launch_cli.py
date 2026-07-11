"""
Terminal preflight and launch flow.
"""

from dataclasses import replace

from app_config import RuntimeConfig
from audio_pipeline import AudioPipeline
from device_manager import DeviceManager, DeviceResolutionError
from model_manager import ModelManager, ModelSetupError
from native_audio_pipeline import NativeAudioPipeline


LAUNCH_PROMPT = 'Type "Launch Transcription" to start:'
ACCEPTED_LAUNCH_INPUTS = {"launch transcription", "luanch transcription"}


def is_launch_command(value: str) -> bool:
    return " ".join(value.lower().strip().split()) in ACCEPTED_LAUNCH_INPUTS


class LaunchCLI:
    def __init__(
        self,
        device_manager: DeviceManager | None = None,
        pipeline: AudioPipeline | None = None,
        native_pipeline: NativeAudioPipeline | None = None,
    ):
        self.device_manager = device_manager or DeviceManager()
        self.pipeline = pipeline or AudioPipeline()
        self.native_pipeline = native_pipeline or NativeAudioPipeline()

    def run(self) -> int:
        runtime_config = RuntimeConfig.resolve()
        print()
        print("Local ASR Preflight")
        print("=" * 60)
        print(f"OS              : {runtime_config.os_name}")
        print(
            "Python          : "
            f"{runtime_config.python_version[0]}."
            f"{runtime_config.python_version[1]}."
            f"{runtime_config.python_version[2]}"
        )
        if runtime_config.python_warning:
            print(f"Warning         : {runtime_config.python_warning}")
        if runtime_config.hardware_profile is not None:
            profile = runtime_config.hardware_profile
            print(
                "Hardware        : "
                f"arch={profile.architecture} cpu={profile.cpu_count} "
                f"ram={profile.total_ram_gb:.1f}GB "
                f"disk_free={profile.available_disk_gb:.1f}GB"
            )
        print(f"Capture backend : {runtime_config.capture_backend}")
        print(f"Capture reason  : {runtime_config.capture_backend_reason}")
        print(f"ASR mode        : {runtime_config.asr_mode}")
        print(f"ASR engine      : {runtime_config.asr_engine} ({runtime_config.asr_engine_label})")
        print(
            "ASR engine impl : "
            f"{'ready' if runtime_config.asr_engine_implemented else 'pending'}"
            f", streaming={'yes' if runtime_config.asr_engine_supports_streaming else 'no'}"
        )
        print(f"ASR selection   : {runtime_config.model_key}")
        print(f"ASR reason      : {runtime_config.asr_reason}")

        devices = None
        if runtime_config.capture_mode == "sounddevice":
            try:
                devices = self.device_manager.resolve_required_devices()
            except DeviceResolutionError as exc:
                print(f"Device error    : {exc}")
                if exc.guidance:
                    print(exc.guidance)
                return 1

        model_info = None
        if not runtime_config.asr_engine_implemented:
            print(
                "ASR engine note : "
                f"{runtime_config.asr_engine} is selected by policy, but this CLI prototype "
                "has not wired that engine yet."
            )
            print(f"Setup guidance  : {runtime_config.asr_engine_setup_hint}")
            return 1

        if runtime_config.asr_mode == "local":
            model_manager = ModelManager(runtime_config.model_cache_dir)
            try:
                model_info = model_manager.ensure_model(
                    runtime_config.model_name,
                    device=runtime_config.whisper_device,
                    compute_type=runtime_config.whisper_compute_type,
                )
            except ModelSetupError as exc:
                print(f"Model error     : {exc}")
                if runtime_config.capture_mode != "sounddevice":
                    print("Cloud fallback is selected by policy but cloud ASR is not wired in this CLI prototype.")
                return 1

        if devices is not None:
            print(f"Mic input       : [{devices.mic.index}] {devices.mic.name}")
            print(f"System input    : [{devices.system.index}] {devices.system.name}")
        else:
            print(f"Native audio    : {runtime_config.native_description}")
            print(
                "Native receiver : "
                f"ws://{runtime_config.native_ws_host}:{runtime_config.native_ws_port}"
            )
            print(
                "Native ASR      : "
                f"window={runtime_config.native_window_seconds:.2f}s "
                f"overlap={runtime_config.native_overlap_seconds:.2f}s"
            )
            print(
                "Native RMS logs : "
                f"{'enabled' if runtime_config.native_rms_logging else 'disabled'}"
            )
            if runtime_config.native_utterance_mode and not runtime_config.asr_engine_supports_streaming:
                print(
                    "Utterance gate  : "
                    f"rms>{runtime_config.native_utterance_rms_threshold:.0f}, "
                    f"low_packets={runtime_config.native_utterance_low_packet_target}, "
                    f"max={runtime_config.native_utterance_max_seconds:.1f}s"
                )
        if model_info is not None:
            print(f"Model           : {model_info.name}")
            print(f"Model cache     : {model_info.cache_path}")
            print(f"Model downloaded: {model_info.downloaded}")
            print(f"Whisper device  : {model_info.device}")
            print(f"Compute type    : {model_info.compute_type}")
        else:
            print("Model           : cloud")
        print(f"VAD             : {'enabled' if runtime_config.vad_enabled else 'disabled'}")
        print(f"Echo handling   : {runtime_config.echo_strategy}")
        print("=" * 60)

        if runtime_config.asr_mode == "cloud":
            print("Cloud ASR was selected, but the cloud ASR client is not implemented in this CLI prototype.")
            return 1

        launch_value = input(f'{LAUNCH_PROMPT} ')
        if not is_launch_command(launch_value):
            print("Launch cancelled.")
            return 0

        local_runtime_config = replace(runtime_config, model_name=str(model_info.cache_path))
        if local_runtime_config.capture_mode == "native":
            self.native_pipeline.start(local_runtime_config)
        else:
            self.pipeline.start(local_runtime_config, devices)
        return 0
