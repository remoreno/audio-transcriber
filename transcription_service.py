"""Transcription service wrapping faster-whisper."""

import ctypes
import importlib.util
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import ctranslate2
from faster_whisper import WhisperModel

from config import DEFAULT_MODEL_SIZE
from models import (
    ProgressUpdate,
    Segment,
    Stage,
    TranscriptionRequest,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

_CUDA_ERROR_KEYWORDS = ("cublas", "cudnn", "cuda", "cufft", "cusparse", "nvcuda")


def _is_cuda_runtime_error(exc: BaseException | None) -> bool:
    """Return True if *exc* looks like a CUDA library/driver failure."""
    if exc is None:
        return False
    msg = str(exc).lower()
    return any(kw in msg for kw in _CUDA_ERROR_KEYWORDS)


def _register_cuda_dll_dirs() -> None:
    """Register CUDA Toolkit DLL directories with the Python DLL loader.

    Since Python 3.8 on Windows, ``LoadLibraryEx`` no longer searches
    ``PATH`` for DLLs.  This function discovers CUDA runtime DLL
    directories and calls :func:`os.add_dll_directory` so that libraries
    like ``cublas64_12.dll`` are found at runtime.
    """
    if sys.platform != "win32":
        return
    candidate_dirs: list[Path] = []
    cuda_path = os.environ.get("CUDA_PATH", "")
    if not cuda_path:
        cuda_path = os.environ.get("CUDA_HOME", "")
    if not cuda_path:
        base = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "NVIDIA GPU Computing Toolkit",
            "CUDA",
        )
        if os.path.isdir(base):
            versions = sorted(os.listdir(base), reverse=True)
            # Filter for CUDA 12.x versions only
            cuda12_versions = [v for v in versions if v.startswith('v12.')]
            if cuda12_versions:
                cuda_path = os.path.join(base, cuda12_versions[0])
    if cuda_path:
        candidate_dirs.append(Path(cuda_path) / "bin")

    candidate_dirs.append(Path(ctranslate2.__file__).resolve().parent)

    nvidia_spec = importlib.util.find_spec("nvidia")
    if nvidia_spec and nvidia_spec.submodule_search_locations:
        for root in nvidia_spec.submodule_search_locations:
            candidate_dirs.extend(Path(root).glob("*/bin"))

    for bin_dir in candidate_dirs:
        if not bin_dir.is_dir():
            continue
        try:
            os.add_dll_directory(str(bin_dir))
            logger.debug("Registered CUDA runtime DLL directory: %s", bin_dir)
        except OSError:
            logger.debug("Failed to register CUDA runtime DLL directory: %s", bin_dir)


def _cuda_is_available() -> bool:
    """Check whether CUDA devices exist and essential libraries are loadable."""
    try:
        if ctranslate2.get_cuda_device_count() == 0:
            return False
    except Exception:
        return False
    try:
        if sys.platform == "win32":
            ctypes.WinDLL("cublas64_12.dll")
        else:
            ctypes.CDLL("libcublas.so.12")
    except OSError:
        logger.info("CUDA device found but cublas library is not loadable")
        return False
    return True


class ModelLoadError(Exception):
    """Raised when the Whisper model fails to load."""


class TranscriptionError(Exception):
    """Raised when transcription of a single file fails."""


class TranscriptionService:
    """Manages a single faster-whisper model and transcribes files.

    The model is loaded once in ``__init__`` and reused for every call to
    :meth:`transcribe`.  This class has **no** threading awareness — the
    caller is responsible for running it off the main thread.

    Args:
        model_size: One of the model size strings accepted by
                    ``faster_whisper.WhisperModel`` (e.g. ``"medium"``).
        device:     ``"cpu"`` or ``"cuda"``.  Defaults to ``"auto"``.
        compute_type: Quantisation type (e.g. ``"int8"``, ``"float16"``).
                      Defaults to ``"int8"`` which is safe on all hardware.
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        *,
        device: str = "auto",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: WhisperModel | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Whisper model.  Raises :class:`ModelLoadError` on failure.

        If the requested device is not ``"cpu"`` and CUDA is unavailable or
        loading fails, the method falls back to ``device="cpu"`` with
        ``compute_type="int8"`` so the application never hangs.
        """
        if self.device != "cpu":
            _register_cuda_dll_dirs()
        if self.device != "cpu" and not _cuda_is_available():
            logger.warning(
                "CUDA is not usable (device=%s) — falling back to CPU",
                self.device,
            )
            self.device = "cpu"
            self.compute_type = "int8"

        logger.info(
            "Loading model '%s' (device=%s, compute_type=%s)",
            self.model_size,
            self.device,
            self.compute_type,
        )
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            if self.device == "cpu":
                logger.exception("Failed to load Whisper model on CPU")
                raise ModelLoadError(
                    f"Could not load model '{self.model_size}': {exc}"
                ) from exc
            logger.warning(
                "Failed to load model with device=%s: %s — falling back to CPU",
                self.device,
                exc,
            )
            self._fallback_to_cpu()
        logger.info(
            "Model loaded successfully (device=%s, compute_type=%s)",
            self.device,
            self.compute_type,
        )

    def _fallback_to_cpu(self) -> None:
        """Reload the model on CPU.  Raises :class:`ModelLoadError` on failure."""
        self.device = "cpu"
        self.compute_type = "int8"
        logger.info(
            "Reloading model '%s' on CPU (compute_type=%s)",
            self.model_size,
            self.compute_type,
        )
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as cpu_exc:
            logger.exception("Failed to load Whisper model on CPU fallback")
            raise ModelLoadError(
                f"Could not load model '{self.model_size}' (CPU fallback): {cpu_exc}"
            ) from cpu_exc

    def transcribe(
        self,
        request: TranscriptionRequest,
        *,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> TranscriptionResult:
        """Transcribe a single audio file.

        Args:
            request:           What to transcribe and how.
            progress_callback: Optional callable invoked after each segment
                               with a :class:`ProgressUpdate` carrying the
                               current percent and message.

        Returns:
            A :class:`TranscriptionResult` with all segments and metadata.

        Raises:
            TranscriptionError: On any failure during transcription
                                (corrupt file, ffmpeg error, etc.).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded — TranscriptionService was not initialised correctly")

        try:
            return self._transcribe_once(request, progress_callback=progress_callback)
        except TranscriptionError as exc:
            if self.device == "cpu" or not _is_cuda_runtime_error(exc.__cause__):
                raise
            logger.warning(
                "CUDA runtime error during transcription — falling back to CPU: %s",
                exc.__cause__,
            )
            self._fallback_to_cpu()
            return self._transcribe_once(request, progress_callback=progress_callback)

    def _transcribe_once(
        self,
        request: TranscriptionRequest,
        *,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
    ) -> TranscriptionResult:
        """Run a single transcription attempt.  Raises :class:`TranscriptionError` on failure."""
        source = request.source_path
        logger.info("Transcribing: %s (language=%s)", source, request.language)
        start_time = time.perf_counter()

        try:
            segments_iter, info = self._model.transcribe(
                str(source),
                language=request.language,
                beam_size=5,
                vad_filter=True,
            )
        except Exception as exc:
            logger.exception("Transcription failed for %s", source)
            raise TranscriptionError(
                f"Transcription failed for '{source.name}': {exc}"
            ) from exc

        detected_language = info.language
        detected_probability = info.language_probability
        duration = info.duration

        logger.debug(
            "Detected language: %s (prob=%.2f), duration=%.1fs",
            detected_language,
            detected_probability,
            duration,
        )

        collected: list[Segment] = []
        try:
            for idx, raw_seg in enumerate(segments_iter, start=1):
                collected.append(
                    Segment(
                        index=idx,
                        start=raw_seg.start,
                        end=raw_seg.end,
                        text=raw_seg.text,
                    )
                )
                if progress_callback is not None and duration > 0:
                    pct = min(raw_seg.end / duration, 1.0)
                    progress_callback(
                        ProgressUpdate(
                            stage=Stage.TRANSCRIBING,
                            file_path=source,
                            percent=pct,
                            message=f"Segment {idx} ({pct:.0%})",
                        )
                    )
        except Exception as exc:
            logger.exception("Error iterating segments for %s", source)
            raise TranscriptionError(
                f"Error reading segments for '{source.name}': {exc}"
            ) from exc

        elapsed = time.perf_counter() - start_time

        if not collected:
            logger.warning("No speech segments detected in %s", source)

        result = TranscriptionResult(
            source_path=source,
            language=detected_language,
            detected_language_probability=detected_probability,
            segments=tuple(collected),
            duration_seconds=duration,
            processing_time=elapsed,
            model_name=self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )

        logger.info(
            "Finished %s: %d segments in %.1fs (audio=%.1fs)",
            source.name,
            len(collected),
            elapsed,
            duration,
        )
        return result
