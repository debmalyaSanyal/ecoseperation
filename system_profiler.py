"""
System hardware and OS profiling for ASR/capture decisions.
"""

from dataclasses import dataclass
import os
import platform
import shutil
import subprocess


GB = 1024 ** 3


@dataclass(frozen=True)
class GpuInfo:
    vendor: str
    name: str
    memory_gb: float
    supports_cuda: bool = False


@dataclass(frozen=True)
class SystemProfile:
    os_name: str
    os_version: str
    architecture: str
    cpu_count: int
    total_ram_gb: float
    available_disk_gb: float
    is_apple_silicon: bool
    gpus: tuple[GpuInfo, ...] = ()

    @property
    def has_cuda_gpu(self) -> bool:
        return any(gpu.supports_cuda for gpu in self.gpus)

    @property
    def max_cuda_vram_gb(self) -> float:
        values = [gpu.memory_gb for gpu in self.gpus if gpu.supports_cuda]
        return max(values) if values else 0.0


class SystemProfiler:
    @classmethod
    def profile_current(cls) -> SystemProfile:
        os_name = platform.system()
        os_version = platform.mac_ver()[0] if os_name == "Darwin" else platform.release()
        arch = platform.machine()
        cpu_count = os.cpu_count() or 1
        total_ram_gb = cls._total_ram_gb(os_name)
        available_disk_gb = shutil.disk_usage(".").free / GB
        return SystemProfile(
            os_name=os_name,
            os_version=os_version,
            architecture=arch,
            cpu_count=cpu_count,
            total_ram_gb=total_ram_gb,
            available_disk_gb=available_disk_gb,
            is_apple_silicon=os_name == "Darwin" and arch in {"arm64", "aarch64"},
            gpus=cls._gpu_info(os_name),
        )

    @staticmethod
    def _total_ram_gb(os_name: str) -> float:
        try:
            if os_name == "Darwin":
                output = subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                return int(output.strip()) / GB
            if os_name == "Windows":
                import ctypes

                class MemoryStatusEx(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                status = MemoryStatusEx()
                status.dwLength = ctypes.sizeof(MemoryStatusEx)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
                return status.ullTotalPhys / GB
        except Exception:
            pass

        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / GB
        except Exception:
            return 0.0

    @staticmethod
    def _gpu_info(os_name: str) -> tuple[GpuInfo, ...]:
        if os_name != "Windows":
            return ()
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
        except Exception:
            return ()

        gpus = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            name, memory_mb = parts
            try:
                memory_gb = float(memory_mb) / 1024
            except ValueError:
                memory_gb = 0.0
            gpus.append(
                GpuInfo(
                    vendor="NVIDIA",
                    name=name,
                    memory_gb=memory_gb,
                    supports_cuda=True,
                )
            )
        return tuple(gpus)
