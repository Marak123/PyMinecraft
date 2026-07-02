"""Central logging setup.

Every subsystem grabs its own named logger via :func:`get_logger` so log
lines always carry a subsystem identifier, per the architecture spec.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-12s | %(message)s"
_configured = False


def init_logging(log_dir: Path | None = None, level: int = logging.INFO) -> None:
    """Configure root logging once: console + optional rotating file."""
    global _configured
    if _configured:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_dir / "engine.log", mode="w", encoding="utf-8")
        )
    logging.basicConfig(level=level, format=_FORMAT, handlers=handlers)
    _configured = True


def get_logger(subsystem: str) -> logging.Logger:
    return logging.getLogger(subsystem)
