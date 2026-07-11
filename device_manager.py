"""
Audio device discovery and validation.
"""

from dataclasses import dataclass
import os
from typing import Any


LOOPBACK_KEYWORDS = (
    "blackhole",
    "loopback",
    "stereo mix",
    "what u hear",
    "wave out",
    "soundflower",
    "vb-cable",
    "vb audio",
    "virtual",
    "aggregate",
)


LOOPBACK_GUIDANCE = """
System audio loopback input was not found.

On Windows, enable Stereo Mix or install a virtual loopback device such as VB-CABLE.
On macOS, install and configure BlackHole 2ch, then route output through it.
"""


@dataclass(frozen=True)
class AudioDevice:
    index: int
    name: str
    max_input_channels: int
    default_samplerate: float

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_loopback(self) -> bool:
        name = self.name.lower()
        return any(keyword in name for keyword in LOOPBACK_KEYWORDS)


@dataclass(frozen=True)
class ResolvedDevices:
    mic: AudioDevice
    system: AudioDevice


class DeviceResolutionError(RuntimeError):
    def __init__(self, message: str, guidance: str | None = None):
        super().__init__(message)
        self.guidance = guidance


class DeviceManager:
    def __init__(self, sounddevice_module: Any | None = None):
        self.sd = sounddevice_module

    def list_input_devices(self) -> list[AudioDevice]:
        if self.sd is None:
            try:
                import sounddevice as sd
            except ImportError as exc:
                raise DeviceResolutionError(
                    "The 'sounddevice' package is not installed. "
                    "Install project dependencies before running preflight."
                ) from exc
            self.sd = sd

        devices = []
        for index, raw in enumerate(self.sd.query_devices()):
            max_input_channels = int(raw.get("max_input_channels", 0))
            if max_input_channels <= 0:
                continue
            devices.append(
                AudioDevice(
                    index=index,
                    name=str(raw.get("name", f"Device {index}")),
                    max_input_channels=max_input_channels,
                    default_samplerate=float(raw.get("default_samplerate", 0)),
                )
            )
        return devices

    def resolve_required_devices(self) -> ResolvedDevices:
        devices = self.list_input_devices()
        if not devices:
            raise DeviceResolutionError("No microphone/input devices were found.")

        mic = self._device_from_env("LOCAL_ASR_MIC_DEVICE", devices)
        system = self._device_from_env("LOCAL_ASR_SYSTEM_DEVICE", devices)

        if mic is None:
            mic = next((device for device in devices if not device.is_loopback), None)
        if mic is None:
            raise DeviceResolutionError("No non-loopback microphone input was found.")

        if system is None:
            system = next((device for device in devices if device.is_loopback), None)
        if system is None:
            raise DeviceResolutionError(
                "No system-audio loopback input was found.",
                guidance=LOOPBACK_GUIDANCE,
            )

        return ResolvedDevices(mic=mic, system=system)

    def _device_from_env(
        self,
        env_name: str,
        devices: list[AudioDevice],
    ) -> AudioDevice | None:
        value = os.environ.get(env_name)
        if value is None or value.strip() == "":
            return None
        try:
            index = int(value)
        except ValueError as exc:
            raise DeviceResolutionError(f"{env_name} must be a device index.") from exc

        for device in devices:
            if device.index == index:
                return device
        raise DeviceResolutionError(f"{env_name}={index} was not found in input devices.")
