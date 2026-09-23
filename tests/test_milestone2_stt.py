"""
Milestone 2 tests: Transcriber's segment-joining and empty-audio handling,
exercised with a fake WhisperModel so no faster-whisper weights need to
be downloaded.

Run with: pytest tests/test_milestone2_stt.py -v
(or run this file's functions directly with plain python -- see bottom)
"""

import numpy as np

from nova.audio.transcriber import Transcriber
from nova.config.loader import get_config


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeSegment:
    """Stands in for faster_whisper's Segment object -- only `.text` is used by Transcriber."""

    def __init__(self, text):
        self.text = text


class FakeInfo:
    """Stands in for faster_whisper's TranscriptionInfo -- only `.language` is used."""

    def __init__(self, language="en"):
        self.language = language


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel. Returns fixed segments/info regardless of input."""

    def __init__(self, segment_texts, language="en"):
        self._segment_texts = segment_texts
        self._language = language

    def transcribe(self, audio, language=None, vad_filter=False):
        segments = [FakeSegment(t) for t in self._segment_texts]
        return segments, FakeInfo(self._language)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_transcriber_joins_segments_and_strips_whitespace():
    cfg = get_config()
    transcriber = Transcriber(config=cfg, model=FakeWhisperModel([" hello ", " world "]))

    audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence, content is irrelevant here
    text = transcriber.transcribe(audio)

    assert text == "hello world"


def test_transcriber_drops_empty_segments():
    cfg = get_config()
    transcriber = Transcriber(config=cfg, model=FakeWhisperModel(["hello", "  ", "there"]))

    text = transcriber.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "hello there"


def test_transcriber_returns_empty_string_for_empty_audio():
    cfg = get_config()
    transcriber = Transcriber(config=cfg, model=FakeWhisperModel(["should not be reached"]))

    text = transcriber.transcribe(np.array([], dtype=np.float32))
    assert text == ""


def test_transcriber_handles_no_speech_detected():
    cfg = get_config()
    transcriber = Transcriber(config=cfg, model=FakeWhisperModel([]))  # model found nothing

    text = transcriber.transcribe(np.zeros(8000, dtype=np.float32))
    assert text == ""


if __name__ == "__main__":
    # Allows `python tests/test_milestone2_stt.py` without pytest installed.
    tests = [
        test_transcriber_joins_segments_and_strips_whitespace,
        test_transcriber_drops_empty_segments,
        test_transcriber_returns_empty_string_for_empty_audio,
        test_transcriber_handles_no_speech_detected,
    ]
    for t in tests:
        t()
        print(f"{t.__name__}: PASS")
