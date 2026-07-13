# -*- coding: utf-8 -*-
"""Developer-facing file logging for runtime errors."""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path


def _log_dir():
    return Path(os.environ.get("AUGUR_LOG_DIR", "logs"))


def _log_path(now=None):
    now = now or datetime.now()
    return _log_dir() / f"augur_{now:%Y-%m-%d}.log"


def _logger():
    now = datetime.now()
    path = _log_path(now)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("augur.system")
    logger.setLevel(logging.INFO)
    target = str(path.resolve())
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == target for h in logger.handlers):
        handler = logging.FileHandler(target, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_exception(module, function, exc, params=None, run_context=None):
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module": str(module or ""),
        "function": str(function or ""),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "params": params or {},
        "run_context": run_context or {},
    }
    _logger().error(json.dumps(entry, ensure_ascii=False, default=str))
    return str(_log_path().resolve())


def log_message(module, function, level, message, params=None, run_context=None):
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "module": str(module or ""),
        "function": str(function or ""),
        "level": str(level or "INFO").upper(),
        "message": str(message),
        "params": params or {},
        "run_context": run_context or {},
    }
    _logger().info(json.dumps(entry, ensure_ascii=False, default=str))
    return str(_log_path().resolve())
