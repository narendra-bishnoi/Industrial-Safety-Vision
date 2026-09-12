"""Logging setup shared by every module.

One console handler (for the human running the CLI) and one rotating file
handler (for post-run debugging) — configured once, reused via
``logging.getLogger(__name__)`` everywhere else.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(log_dir: str | Path, level: int = logging.INFO) -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("safety_vision")
    logger.setLevel(level)
    logger.handlers.clear()  # avoid duplicate handlers if called more than once (e.g. in tests)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "run.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
