"""
nova.config.loader
-------------------
Loads config/config.yaml once and exposes it as a cached, dict-like object.

Usage:
    from nova.config.loader import get_config
    cfg = get_config()
    print(cfg["audio"]["sample_rate"])
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_CACHE: Dict[str, Any] | None = None
_LOCK = threading.Lock()

# Repo layout: <repo_root>/config/config.yaml, this file at <repo_root>/nova/config/loader.py
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"NOVA config file not found at {path}. "
            "Did you rename or move config/config.yaml?"
        )
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a dictionary.")
    return data


def get_config(path: str | os.PathLike | None = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Return the parsed config dict, loading it from disk on first call
    and caching it afterward. Thread-safe.

    Args:
        path: optional override path to a config yaml file (mainly for tests).
        force_reload: bypass the cache and re-read the file from disk.
    """
    global _CONFIG_CACHE

    with _LOCK:
        if _CONFIG_CACHE is not None and not force_reload and path is None:
            return _CONFIG_CACHE

        config_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
        data = _load_yaml(config_path)

        if path is None:
            _CONFIG_CACHE = data
        return data


def reset_cache() -> None:
    """Clear the cached config. Mostly useful in tests."""
    global _CONFIG_CACHE
    with _LOCK:
        _CONFIG_CACHE = None
