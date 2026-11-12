"""Uniform logging so that every pipeline stage leaves an auditable trace."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import LOGS

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, logfile: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(stream)
    target = Path(logfile) if logfile else LOGS / f"{name.split('.')[-1]}.log"
    fileh = logging.FileHandler(target, encoding="utf-8")
    fileh.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(fileh)
    logger.propagate = False
    return logger
