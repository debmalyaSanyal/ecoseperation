import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from asr_decision_engine import ASRDecisionEngine
from app_config import RuntimeConfig
from capture_backend import CaptureBackendResolver
from device_manager import DeviceManager, DeviceResolutionError
from launch_cli import is_launch_command
from model_manager import ModelManager
from native_audio_pipeline import SlidingAudioBuffer, UtteranceAudioBuffer, decode_pcm16_stereo, pcm16_rms
from system_profiler import GpuInfo, SystemProfile
from transcript_dedupe import CrossSourceDedupe


def profile(
    os_name="Darwin",
    os_version="14.0",
    architecture="x86_64",
    cpu_count=8,
    total_ram_gb=16,
    available_disk_gb=100,
    gpus=(),
):
    return SystemProfile(
        os_name=os_name,
        os_version=os_version,
        architecture=architecture,
        cpu_count=cpu_count,
        total_ram_gb=total_ram_gb,
        available_disk_gb=available_disk_gb,
        is_apple_silicon=os_name == "Darwin" and architecture == "arm64",
        gpus=tuple(gpus),
    )


class RuntimeConfigTests(unittest.TestCase):
    def test_macos_uses_cpu_int8(self):
        config = RuntimeConfig.resolve(selected_model="small", profile=profile())

        self.assertEqual(config.whisper_device, "cpu")
        self.assertEqual(config.whisper_compute_type, "int8")
        self.assertEqual(config.model_name, "small")
        self.assertEqual(config.capture_mode, "native")
        self.assertEqual(config.capture_backend, "macos-native")

    def test_sounddevice_capture_mode_override(self):
        with patch.dict("os.environ", {"CAPTURE_BACKEND": "sounddevice"}):
            config = RuntimeConfig.resolve(selected_model="small", profile=profile())

        self.assertEqual(config.capture_mode, "sounddevice")

    def test_native_window_overlap_env_overrides(self):
        with patch.dict(
            "os.environ",
            {
                "LOCAL_ASR_NATIVE_WINDOW_SECONDS": "1.5",
                "LOCAL_ASR_NATIVE_OVERLAP_SECONDS": "0.2",
                "LOCAL_ASR_NATIVE_RMS_LOGGING": "false",
            },
        ):
            config = RuntimeConfig.resolve(selected_model="small", profile=profile())

        self.assertEqual(config.native_window_seconds, 1.5)
        self.assertEqual(config.native_overlap_seconds, 0.2)
        self.assertFalse(config.native_rms_logging)

    def test_native_utterance_env_overrides(self):
        with patch.dict(
            "os.environ",
            {
                "LOCAL_ASR_UTTERANCE_MODE": "false",
                "LOCAL_ASR_UTTERANCE_RMS_THRESHOLD": "120",
                "LOCAL_ASR_UTTERANCE_LOW_PACKET_TARGET": "4",
                "LOCAL_ASR_UTTERANCE_MIN_SECONDS": "0.5",
                "LOCAL_ASR_UTTERANCE_MAX_SECONDS": "10",
            },
        ):
            config = RuntimeConfig.resolve(selected_model="small", profile=profile())

        self.assertFalse(config.native_utterance_mode)
        self.assertEqual(config.native_utterance_rms_threshold, 120)
        self.assertEqual(config.native_utterance_low_packet_target, 4)
        self.assertEqual(config.native_utterance_min_seconds, 0.5)
        self.assertEqual(config.native_utterance_max_seconds, 10)

    def test_native_low_latency_config_applies_fast_decode_settings(self):
        import config as legacy_config

        runtime_config = RuntimeConfig.resolve(selected_model="small", profile=profile())
        original = (
            legacy_config.BEAM_SIZE,
            legacy_config.BEST_OF,
            legacy_config.PATIENCE,
            legacy_config.CONDITION_ON_PREVIOUS_TEXT,
        )
        try:
            runtime_config.apply_native_low_latency_config()

            self.assertEqual(legacy_config.BEAM_SIZE, 1)
            self.assertEqual(legacy_config.BEST_OF, 1)
            self.assertEqual(legacy_config.PATIENCE, 1)
            self.assertFalse(legacy_config.CONDITION_ON_PREVIOUS_TEXT)
        finally:
            (
                legacy_config.BEAM_SIZE,
                legacy_config.BEST_OF,
                legacy_config.PATIENCE,
                legacy_config.CONDITION_ON_PREVIOUS_TEXT,
            ) = original


class CaptureBackendResolverTests(unittest.TestCase):
    def test_resolver_chooses_macos_native_on_modern_macos(self):
        with patch.dict("os.environ", {}, clear=True):
            backend = CaptureBackendResolver().resolve(profile(os_name="Darwin", os_version="14.0"))

        self.assertEqual(backend.backend, "macos-native")
        self.assertEqual(backend.capture_mode, "native")

    def test_resolver_chooses_windows_wasapi_on_windows(self):
        with patch.dict("os.environ", {}, clear=True):
            backend = CaptureBackendResolver().resolve(profile(os_name="Windows", os_version="11"))

        self.assertEqual(backend.backend, "windows-wasapi")
        self.assertEqual(backend.capture_mode, "native")

    def test_old_macos_falls_back_to_sounddevice(self):
        with patch.dict("os.environ", {}, clear=True):
            backend = CaptureBackendResolver().resolve(profile(os_name="Darwin", os_version="12.6"))

        self.assertEqual(backend.backend, "sounddevice")


class ASRDecisionEngineTests(unittest.TestCase):
    def test_weak_hardware_selects_cloud_with_fallback(self):
        with patch.dict("os.environ", {"ASR_CLOUD_FALLBACK": "true"}, clear=True):
            decision = ASRDecisionEngine().decide(profile(total_ram_gb=4))

        self.assertEqual(decision.mode, "cloud")
        self.assertEqual(decision.model_key, "cloud")

    def test_manual_model_override_wins(self):
        with patch.dict("os.environ", {}, clear=True):
            decision = ASRDecisionEngine().decide(profile(total_ram_gb=4), override="tiny")

        self.assertEqual(decision.mode, "local")
        self.assertEqual(decision.model_key, "tiny")

    def test_standard_hardware_selects_small(self):
        with patch.dict("os.environ", {}, clear=True):
            decision = ASRDecisionEngine().decide(profile(total_ram_gb=16, cpu_count=8))

        self.assertEqual(decision.model_key, "small")
        self.assertEqual(decision.engine_key, "faster-whisper")

    def test_macos_apple_silicon_selects_whisper_cpp(self):
        with patch.dict("os.environ", {}, clear=True):
            decision = ASRDecisionEngine().decide(
                profile(architecture="arm64", total_ram_gb=16, cpu_count=8)
            )

        self.assertEqual(decision.engine_key, "whisper.cpp")
        self.assertEqual(decision.model_key, "small")
        self.assertEqual(decision.device, "metal")

    def test_windows_without_nvidia_selects_whisper_cpp(self):
        with patch.dict("os.environ", {}, clear=True):
            decision = ASRDecisionEngine().decide(
                profile(os_name="Windows", total_ram_gb=16, cpu_count=8)
            )

        self.assertEqual(decision.engine_key, "whisper.cpp")
        self.assertEqual(decision.model_key, "small")

    def test_windows_cuda_6gb_selects_large_v3_turbo(self):
        with patch.dict("os.environ", {}, clear=True):
            decision = ASRDecisionEngine().decide(
                profile(
                    os_name="Windows",
                    total_ram_gb=32,
                    gpus=(GpuInfo("NVIDIA", "RTX", 8, supports_cuda=True),),
                )
            )

        self.assertEqual(decision.engine_key, "nemo")
        self.assertEqual(decision.model_key, "nemo-streaming-small")
        self.assertEqual(decision.device, "cuda")

    def test_windows_cuda_10gb_selects_large_v3(self):
        with patch.dict("os.environ", {}, clear=True):
            decision = ASRDecisionEngine().decide(
                profile(
                    os_name="Windows",
                    total_ram_gb=32,
                    gpus=(GpuInfo("NVIDIA", "RTX", 12, supports_cuda=True),),
                )
            )

        self.assertEqual(decision.engine_key, "nemo")
        self.assertEqual(decision.model_key, "nemo-streaming-large")

    def test_manual_engine_override_wins(self):
        with patch.dict("os.environ", {"ASR_ENGINE_OVERRIDE": "faster-whisper"}, clear=True):
            decision = ASRDecisionEngine().decide(
                profile(architecture="arm64", total_ram_gb=16, cpu_count=8)
            )

        self.assertEqual(decision.engine_key, "faster-whisper")
        self.assertEqual(decision.model_key, "small")

    def test_manual_model_override_wins_with_selected_engine(self):
        with patch.dict(
            "os.environ",
            {
                "ASR_ENGINE_OVERRIDE": "nemo",
                "ASR_MODEL_OVERRIDE": "nemo-streaming-small",
            },
            clear=True,
        ):
            decision = ASRDecisionEngine().decide(
                profile(
                    os_name="Windows",
                    total_ram_gb=32,
                    gpus=(GpuInfo("NVIDIA", "RTX", 12, supports_cuda=True),),
                )
            )

        self.assertEqual(decision.engine_key, "nemo")
        self.assertEqual(decision.model_key, "nemo-streaming-small")


class FakeSoundDevice:
    def __init__(self, devices):
        self.devices = devices

    def query_devices(self):
        return self.devices


class DeviceManagerTests(unittest.TestCase):
    def test_rejects_missing_mic(self):
        manager = DeviceManager(FakeSoundDevice([]))

        with self.assertRaises(DeviceResolutionError):
            manager.resolve_required_devices()

    def test_rejects_missing_loopback_with_guidance(self):
        manager = DeviceManager(
            FakeSoundDevice(
                [
                    {
                        "name": "MacBook Microphone",
                        "max_input_channels": 1,
                        "default_samplerate": 48000,
                    }
                ]
            )
        )

        with self.assertRaises(DeviceResolutionError) as ctx:
            manager.resolve_required_devices()

        self.assertIn("loopback", str(ctx.exception).lower())
        self.assertIn("BlackHole", ctx.exception.guidance)


class ModelManagerTests(unittest.TestCase):
    def test_skips_download_when_cache_exists(self):
        calls = []

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            model_cache = cache_dir / "small"
            model_cache.mkdir()
            (model_cache / "cached.bin").write_text("cached", encoding="utf8")

            manager = ModelManager(cache_dir, model_cls=FakeWhisperModel)
            info = manager.ensure_model("small")

        self.assertFalse(info.downloaded)
        self.assertEqual(info.cache_path.name, "small")
        self.assertTrue(calls[0][1]["local_files_only"])


class LaunchPromptTests(unittest.TestCase):
    def test_launch_prompt_accepts_expected_values(self):
        self.assertTrue(is_launch_command("Launch Transcription"))
        self.assertTrue(is_launch_command("launch transcription"))
        self.assertTrue(is_launch_command("Luanch Transcription"))
        self.assertFalse(is_launch_command("start"))


class NativeAudioTests(unittest.TestCase):
    def test_decode_pcm16_stereo_splits_system_and_mic(self):
        payload = (
            int(32767).to_bytes(2, "little", signed=True)
            + int(-32768).to_bytes(2, "little", signed=True)
            + int(0).to_bytes(2, "little", signed=True)
            + int(16384).to_bytes(2, "little", signed=True)
        )

        system_audio, mic_audio = decode_pcm16_stereo(payload)

        self.assertAlmostEqual(float(system_audio[0]), 32767 / 32768, places=5)
        self.assertAlmostEqual(float(mic_audio[0]), -1.0, places=5)
        self.assertAlmostEqual(float(system_audio[1]), 0.0, places=5)
        self.assertAlmostEqual(float(mic_audio[1]), 0.5, places=5)

    def test_sliding_audio_buffer_emits_windows_after_enough_frames(self):
        audio_buffer = SlidingAudioBuffer(sample_rate=4, window_seconds=1, overlap_seconds=0.5)

        self.assertEqual(audio_buffer.append(np.array([1, 2], dtype=np.float32)), [])
        chunks = audio_buffer.append(np.array([3, 4], dtype=np.float32))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(audio_buffer.buffer.tolist(), [3, 4])

    def test_sliding_audio_buffer_advances_by_window_minus_overlap(self):
        audio_buffer = SlidingAudioBuffer(sample_rate=4, window_seconds=1, overlap_seconds=0.25)

        chunks = audio_buffer.append(np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32))

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].tolist(), [1, 2, 3, 4])
        self.assertEqual(chunks[1].tolist(), [4, 5, 6, 7])
        self.assertEqual(audio_buffer.buffer.tolist(), [7])

    def test_pcm16_rms_uses_pcm_scale(self):
        audio = np.array([80 / 32768, -80 / 32768], dtype=np.float32)

        self.assertAlmostEqual(pcm16_rms(audio), 80, places=3)

    def test_utterance_buffer_flushes_after_low_energy_packets(self):
        audio_buffer = UtteranceAudioBuffer(
            sample_rate=4,
            rms_threshold=80,
            low_packet_target=3,
            min_seconds=0.25,
            max_seconds=10,
        )
        active = np.full(2, 100 / 32768, dtype=np.float32)
        quiet = np.zeros(2, dtype=np.float32)

        self.assertEqual(audio_buffer.append(active), [])
        self.assertEqual(audio_buffer.append(quiet), [])
        self.assertEqual(audio_buffer.append(quiet), [])
        chunks = audio_buffer.append(quiet)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 8)
        self.assertFalse(audio_buffer.active)

    def test_utterance_buffer_ignores_quiet_before_activity(self):
        audio_buffer = UtteranceAudioBuffer(
            sample_rate=4,
            rms_threshold=80,
            low_packet_target=3,
            min_seconds=0.25,
            max_seconds=10,
        )

        self.assertEqual(audio_buffer.append(np.zeros(4, dtype=np.float32)), [])
        self.assertEqual(len(audio_buffer.buffer), 0)

    def test_utterance_buffer_force_flushes_at_max_duration(self):
        audio_buffer = UtteranceAudioBuffer(
            sample_rate=4,
            rms_threshold=80,
            low_packet_target=3,
            min_seconds=0.25,
            max_seconds=1,
        )
        active = np.full(5, 100 / 32768, dtype=np.float32)

        chunks = audio_buffer.append(active)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(chunks[0]), 4)
        self.assertEqual(len(audio_buffer.buffer), 1)


class CrossSourceDedupeTests(unittest.TestCase):
    def test_suppresses_near_identical_cross_source_text(self):
        dedupe = CrossSourceDedupe(threshold=0.80, window_seconds=30)

        with patch("transcript_dedupe.time.time", side_effect=[100.0, 101.0]):
            self.assertTrue(dedupe.should_emit("SYSTEM", "Hello from the meeting"))
            self.assertFalse(dedupe.should_emit("MIC", "hello from the meeting."))


if __name__ == "__main__":
    unittest.main()
