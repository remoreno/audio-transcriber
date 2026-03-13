"""Writers for TXT, SRT, and JSON output formats."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from models import TranscriptionResult

logger = logging.getLogger(__name__)


def _format_srt_timestamp(seconds: float) -> str:
    """Convert a float seconds value to SRT timestamp ``HH:MM:SS,mmm``."""
    if seconds < 0:
        seconds = 0.0
    hours, remainder = divmod(int(seconds * 1000), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_txt(result: TranscriptionResult, path: Path) -> None:
    """Write a plain-text transcript, one segment per line.

    Args:
        result: The transcription result to serialise.
        path:   Destination file path (should not already exist).
    """
    with open(path, "w", encoding="utf-8") as fh:
        for seg in result.segments:
            fh.write(seg.text.strip() + "\n")
    logger.info("Wrote TXT: %s", path)


def write_srt(result: TranscriptionResult, path: Path) -> None:
    """Write an SRT subtitle file.

    Format per block::

        1
        00:00:00,000 --> 00:00:02,500
        Segment text

    Args:
        result: The transcription result to serialise.
        path:   Destination file path (should not already exist).
    """
    with open(path, "w", encoding="utf-8") as fh:
        for seg in result.segments:
            fh.write(f"{seg.index}\n")
            fh.write(
                f"{_format_srt_timestamp(seg.start)} --> "
                f"{_format_srt_timestamp(seg.end)}\n"
            )
            fh.write(seg.text.strip() + "\n\n")
    logger.info("Wrote SRT: %s", path)


def write_json(result: TranscriptionResult, path: Path) -> None:
    """Write a JSON transcript with metadata and segments.

    Structure::

        {
          "source_file": "...",
          "output_created_at": "...",
          "model_name": "...",
          "device": "...",
          "compute_type": "...",
          "language": "...",
          "detected_language_probability": ...,
          "segments": [
            {"start": 0.0, "end": 1.5, "text": "..."},
            ...
          ]
        }

    Args:
        result: The transcription result to serialise.
        path:   Destination file path (should not already exist).
    """
    payload: dict = {
        "source_file": str(result.source_path),
        "output_created_at": datetime.now(timezone.utc).isoformat(),
        "model_name": result.model_name,
        "device": result.device,
        "compute_type": result.compute_type,
        "language": result.language,
        "detected_language_probability": result.detected_language_probability,
        "segments": [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in result.segments
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info("Wrote JSON: %s", path)
