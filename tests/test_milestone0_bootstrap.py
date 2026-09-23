"""
Smoke tests for Milestone 0: config loads, logging configures, and the
config values main.py depends on are actually present.

Run with: pytest tests/test_milestone0_bootstrap.py -v
"""

from nova.config.loader import get_config, reset_cache
from nova.logging_setup import setup_logging, get_logger


def test_config_loads_and_has_required_keys():
    reset_cache()
    cfg = get_config()

    assert "app" in cfg
    assert cfg["app"]["name"] == "NOVA"
    assert "wake_phrase" in cfg["app"]

    assert "audio" in cfg
    assert cfg["audio"]["sample_rate"] == 16000

    assert "logging" in cfg


def test_config_is_cached_between_calls():
    reset_cache()
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2  # same object -> cache hit, no re-parse


def test_logging_setup_is_idempotent():
    # Calling setup_logging() twice should not raise or duplicate handlers endlessly.
    setup_logging()
    handlers_after_first = len(get_logger("test").parent.handlers) if get_logger("test").parent else 0
    setup_logging()
    handlers_after_second = len(get_logger("test").parent.handlers) if get_logger("test").parent else 0
    assert handlers_after_first == handlers_after_second


def test_get_logger_namespaces_under_nova():
    log = get_logger("wake_word")
    assert log.name == "nova.wake_word"
