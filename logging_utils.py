"""Logging configuration for the audio transcriber."""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOG_FILENAME, LOG_MAX_BYTES, LOG_BACKUP_COUNT

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(*, log_dir: Path | None = None, level: int = logging.DEBUG) -> None:
    """Configure the root logger with a rotating file handler and stderr output.

    Args:
        log_dir: Directory for the log file.  Defaults to the directory
                 containing this module (i.e. the project root at runtime).
        level:   Minimum logging level.  ``DEBUG`` by default so the log
                 file captures everything; stderr uses ``INFO``.
    """
    root = logging.getLogger()
    if root.handlers:
        return

    if log_dir is None:
        log_dir = Path(__file__).resolve().parent

    log_path = log_dir / LOG_FILENAME

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.DEBUG)
    stderr_handler.setFormatter(formatter)

    root.setLevel(level)
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)
