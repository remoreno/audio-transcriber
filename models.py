"""Shared dataclasses used across all modules."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Stage(Enum):
    """Discrete stages reported by the worker thread."""

    LOADING_MODEL = "loading_model"
    TRANSCRIBING = "transcribing"
    WRITING_OUTPUT = "writing_output"
    DONE = "done"
    ERROR = "error"
    BATCH_COMPLETE = "batch_complete"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TranscriptionRequest:
    """A single file to transcribe, with all settings resolved."""

    source_path: Path
    language: str | None
    output_dir: Path | None


@dataclass(frozen=True)
class Segment:
    """One timed text segment from the transcription."""

    index: int
    start: float   # seconds
    end: float      # seconds
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    """Complete result for one file."""

    source_path: Path
    language: str
    detected_language_probability: float | None
    segments: tuple[Segment, ...]
    duration_seconds: float
    processing_time: float
    model_name: str
    device: str
    compute_type: str


@dataclass(frozen=True)
class ProgressUpdate:
    """Message sent from the worker thread to the UI via a queue.

    ``stage`` drives UI state transitions.  ``percent`` is 0.0–1.0 for the
    current file.  ``message`` is a short human-readable status string.
    """

    stage: Stage
    file_path: Path | None = None
    file_index: int = 0
    total_files: int = 0
    percent: float = 0.0
    message: str = ""
