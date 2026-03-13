"""Centralized constants and configuration for the audio transcriber."""

from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".ogg", ".mp3", ".wav", ".m4a", ".mp4",
})

OUTPUT_FORMATS: tuple[str, ...] = (".txt", ".srt", ".json")

DEFAULT_LANGUAGE: str = "es"

DEFAULT_MODEL_SIZE: str = "medium"

AVAILABLE_MODEL_SIZES: tuple[str, ...] = (
    "tiny", "base", "small", "medium", "large-v2", "large-v3",
)

LOG_FILENAME: str = "audio_transcriber.log"
LOG_MAX_BYTES: int = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT: int = 3

QUEUE_POLL_MS: int = 100


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime settings resolved at startup."""

    model_size: str = DEFAULT_MODEL_SIZE
    language: str | None = DEFAULT_LANGUAGE
    output_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.model_size not in AVAILABLE_MODEL_SIZES:
            raise ValueError(
                f"Unsupported model size '{self.model_size}'. "
                f"Choose from: {', '.join(AVAILABLE_MODEL_SIZES)}"
            )
