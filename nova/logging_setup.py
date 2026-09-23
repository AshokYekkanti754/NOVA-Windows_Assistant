"""
nova.logging_setup
-------------------
Structured, per-stage logging for the NOVA pipeline.

Every pipeline stage (audio, wake_word, vad, stt, brain, tools, tts) gets its
own named logger via get_logger(stage_name), so log lines are tagged with the
stage that produced them. That's what lets you measure latency per stage later
(Milestone 8) instead of grepping one undifferentiated log stream.

Usage:
    from nova.logging_setup import setup_logging, get_logger
    setup_logging()                      # call once, at startup
    log = get_logger("wake_word")
    log.info("Listening for 'hey nova'...")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from nova.config.loader import get_config

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging() -> None:
    """
    Configure the root 'nova' logger once, according to config.yaml's
    `logging` section. Safe to call more than once (subsequent calls are no-ops).
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = get_config()
    log_cfg = cfg.get("logging", {})

    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("nova")
    root.setLevel(level)
    root.propagate = False  # don't double-log through the bare root logger

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if log_cfg.get("log_to_console", True):
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    if log_cfg.get("log_to_file", True):
        log_dir = Path(log_cfg.get("log_dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "nova.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(stage_name: str) -> logging.Logger:
    """
    Return a logger namespaced under 'nova.<stage_name>', e.g. 'nova.wake_word'.
    Call setup_logging() once at startup before using loggers returned here.
    """
    return logging.getLogger(f"nova.{stage_name}")
