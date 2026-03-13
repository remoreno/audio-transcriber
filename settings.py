"""Persistent UI settings backed by a JSON file."""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"


@dataclass
class UISettings:
    """Serialisable snapshot of every user-facing setting."""

    model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "Spanish (forced)"
    fmt_txt: bool = True
    fmt_srt: bool = True
    fmt_json: bool = True
    output_dir: str | None = None
    window_x: int | None = None
    window_y: int | None = None
    window_width: int | None = None
    window_height: int | None = None


def load_settings() -> UISettings:
    """Load settings from disk.  Returns defaults on any failure."""
    if not _SETTINGS_PATH.is_file():
        return UISettings()
    try:
        raw: dict[str, Any] = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
        known_fields = {f.name for f in UISettings.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known_fields}
        return UISettings(**filtered)
    except Exception:
        logger.warning("Failed to load settings from %s; using defaults", _SETTINGS_PATH, exc_info=True)
        return UISettings()


def save_settings(settings: UISettings) -> None:
    """Persist settings to disk.  Logs and swallows errors."""
    try:
        _SETTINGS_PATH.write_text(
            json.dumps(asdict(settings), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        logger.warning("Failed to save settings to %s", _SETTINGS_PATH, exc_info=True)
