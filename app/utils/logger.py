"""Application logging, in the format the build notes specify.

    2026-08-27 10:30:15 | INFO | Prediction request received

Application logs answer "what did the service do". Prediction logs are kept
separately by app.services.prediction_log, because they are analysed later for
drift and must not be tangled up with startup noise.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.utils.config import LOGS_DIR

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    # 5 MB per file, three generations kept - enough to cover a demo without
    # silently filling the disk.
    file_handler = RotatingFileHandler(
        LOGS_DIR / "application.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    root = logging.getLogger("hr_ai")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the shared hr_ai root."""
    _configure_root()
    return logging.getLogger(f"hr_ai.{name}")
