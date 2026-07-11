"""
OS-aware native audio capture backend resolution.
"""

from dataclasses import dataclass
from pathlib import Path
import os

from system_profiler import SystemProfile


BACKEND_AUTO = "auto"
BACKEND_MACOS_NATIVE = "macos-native"
BACKEND_WINDOWS_WASAPI = "windows-wasapi"
BACKEND_SOUNDDEVICE = "sounddevice"


@dataclass(frozen=True)
class CaptureBackend:
    backend: str
    capture_mode: str
    helper_dir: Path | None
    helper_binary: Path | None
    helper_run_command: tuple[str, ...]
    description: str
    reason: str


class CaptureBackendResolver:
    def resolve(self, profile: SystemProfile) -> CaptureBackend:
        requested = (
            os.environ.get("CAPTURE_BACKEND")
            or os.environ.get("LOCAL_ASR_CAPTURE_MODE")
            or BACKEND_AUTO
        ).strip().lower()

        if requested in {"native", BACKEND_AUTO}:
            requested = BACKEND_AUTO
        if requested == BACKEND_SOUNDDEVICE:
            return self._sounddevice("manual fallback selected")
        if requested == BACKEND_MACOS_NATIVE:
            return self._macos(profile, "manual macOS native selected")
        if requested == BACKEND_WINDOWS_WASAPI:
            return self._windows(profile, "manual Windows WASAPI selected")

        if profile.os_name == "Darwin":
            return self._macos(profile, "auto selected macOS native capture")
        if profile.os_name == "Windows":
            return self._windows(profile, "auto selected Windows WASAPI capture")
        return self._sounddevice(f"unsupported native OS {profile.os_name}; using fallback")

    def _macos(self, profile: SystemProfile, reason: str) -> CaptureBackend:
        major = self._major_version(profile.os_version)
        if major is not None and major < 13:
            return self._sounddevice(
                f"macOS {profile.os_version} is below ScreenCaptureKit system-audio target; using fallback"
            )
        helper_dir = Path("native-audio-capture") / "macos"
        helper_binary = (
            helper_dir
            / ".build"
            / "x86_64-apple-macosx"
            / "debug"
            / "native-audio-streamer"
        )
        return CaptureBackend(
            backend=BACKEND_MACOS_NATIVE,
            capture_mode="native",
            helper_dir=helper_dir,
            helper_binary=helper_binary,
            helper_run_command=("swift", "run", "native-audio-streamer"),
            description="macOS ScreenCaptureKit system audio + AVFoundation microphone",
            reason=reason,
        )

    @staticmethod
    def _major_version(version: str) -> int | None:
        try:
            return int(version.split(".", 1)[0])
        except (AttributeError, ValueError):
            return None

    def _windows(self, profile: SystemProfile, reason: str) -> CaptureBackend:
        helper_dir = Path("native-audio-capture") / "windows"
        arch = "win-arm64" if profile.architecture.lower() in {"arm64", "aarch64"} else "win-x64"
        helper_binary = helper_dir / "bin" / arch / "native-audio-streamer.exe"
        return CaptureBackend(
            backend=BACKEND_WINDOWS_WASAPI,
            capture_mode="native",
            helper_dir=helper_dir,
            helper_binary=helper_binary,
            helper_run_command=(str(helper_binary),),
            description="Windows WASAPI loopback system audio + default microphone",
            reason=reason,
        )

    def _sounddevice(self, reason: str) -> CaptureBackend:
        return CaptureBackend(
            backend=BACKEND_SOUNDDEVICE,
            capture_mode="sounddevice",
            helper_dir=None,
            helper_binary=None,
            helper_run_command=(),
            description="developer sounddevice fallback",
            reason=reason,
        )
