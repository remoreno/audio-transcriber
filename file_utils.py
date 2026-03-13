"""File validation, ffmpeg detection, and safe output-path generation."""

import logging
import subprocess
import sys
from pathlib import Path

from config import SUPPORTED_EXTENSIONS

logger = logging.getLogger(__name__)


class UnsupportedFileError(Exception):
    """Raised when a file has an unsupported extension."""


def validate_input_file(path: Path) -> Path:
    """Verify that *path* exists and has a supported extension.

    Returns the resolved ``Path`` on success.

    Raises:
        FileNotFoundError:    If the file does not exist.
        UnsupportedFileError: If the extension is not in
                              :data:`config.SUPPORTED_EXTENSIONS`.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return path


def resolve_output_path(
    source: Path,
    suffix: str,
    output_dir: Path | None = None,
) -> Path:
    """Build a non-colliding output path for *source* with the given *suffix*.

    If *output_dir* is ``None`` the output file is placed next to *source*.
    When a file already exists at the candidate path, a numeric suffix
    (``_1``, ``_2``, …) is appended to the stem until a free name is found.

    Args:
        source:     Original input file path.
        suffix:     Desired extension including the dot (e.g. ``".srt"``).
        output_dir: Optional override directory for output files.

    Returns:
        A ``Path`` that does not yet exist on disk.
    """
    directory = output_dir if output_dir is not None else source.parent
    if not directory.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {directory}")
    stem = source.stem
    candidate = directory / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def check_ffmpeg() -> bool:
    """Return ``True`` if ``ffmpeg`` is reachable on the system PATH.

    Logs the detected version at DEBUG level on success, or an error-level
    message on failure.
    """
    try:
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            **kwargs,
        )
        if result.returncode == 0:
            first_line = result.stdout.split("\n", 1)[0]
            logger.debug("ffmpeg detected: %s", first_line)
            return True
        logger.error("ffmpeg returned non-zero exit code %d", result.returncode)
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found on PATH")
        return False
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg version check timed out")
        return False
    except OSError as exc:
        logger.error("Failed to run ffmpeg: %s", exc)
        return False
