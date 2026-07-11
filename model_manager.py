"""
Local Faster-Whisper model setup.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LocalModelInfo:
    name: str
    cache_path: Path
    downloaded: bool
    device: str
    compute_type: str


class ModelSetupError(RuntimeError):
    pass


class ModelManager:
    def __init__(
        self,
        cache_dir: Path,
        model_cls: Any | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.model_cls = model_cls

    def ensure_model(
        self,
        model_name: str,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> LocalModelInfo:
        model_cache_path = self._model_cache_path(model_name)
        try:
            model_cache_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ModelSetupError(
                f"Could not create model cache directory '{model_cache_path}': {exc}"
            ) from exc

        had_cache = self._has_cached_files(model_cache_path)

        if self.model_cls is not None:
            return self._ensure_with_injected_model_cls(
                model_name,
                model_cache_path,
                had_cache,
                device,
                compute_type,
            )

        model_cls, download_model = self._real_model_dependencies()
        try:
            resolved_model_path = download_model(
                model_name,
                local_files_only=had_cache,
                cache_dir=str(model_cache_path),
            )
            model = model_cls(
                resolved_model_path,
                device=device,
                compute_type=compute_type,
                local_files_only=True,
                cpu_threads=4,
                num_workers=1,
            )
            del model
        except Exception as exc:
            if had_cache:
                raise ModelSetupError(
                    f"Cached model '{model_name}' exists but could not be loaded: {exc}"
                ) from exc
            raise ModelSetupError(
                f"Could not download or load local model '{model_name}': {exc}"
            ) from exc

        return LocalModelInfo(
            name=model_name,
            cache_path=Path(resolved_model_path),
            downloaded=not had_cache,
            device=device,
            compute_type=compute_type,
        )

    def _ensure_with_injected_model_cls(
        self,
        model_name: str,
        model_cache_path: Path,
        had_cache: bool,
        device: str,
        compute_type: str,
    ) -> LocalModelInfo:
        try:
            model = self.model_cls(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(model_cache_path),
                local_files_only=had_cache,
                cpu_threads=4,
                num_workers=1,
            )
            del model
        except Exception as exc:
            raise ModelSetupError(
                f"Could not download or load local model '{model_name}': {exc}"
            ) from exc

        return LocalModelInfo(
            name=model_name,
            cache_path=model_cache_path,
            downloaded=not had_cache,
            device=device,
            compute_type=compute_type,
        )

    def _model_cache_path(self, model_name: str) -> Path:
        safe_name = model_name.replace("/", "__")
        return self.cache_dir / safe_name

    def _has_cached_files(self, cache_path: Path) -> bool:
        return cache_path.exists() and any(cache_path.iterdir())

    def _real_model_dependencies(self) -> tuple[Any, Any]:
        try:
            from faster_whisper import WhisperModel
            from faster_whisper.utils import download_model
        except ImportError as exc:
            raise ModelSetupError(
                "The 'faster-whisper' package is not installed. "
                "Install project dependencies before model setup."
            ) from exc
        return WhisperModel, download_model
