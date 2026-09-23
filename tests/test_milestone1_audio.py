"""
Milestone 1 tests: wake word triggering, VAD speech segmentation, and the
full pipeline state machine -- all exercised with fake/injected models so
no microphone, no openWakeWord weights, and no Silero model are needed.

Run with: pytest tests/test_milestone1_audio.py -v
(or run this file's functions directly with plain python -- see bottom)
"""

import numpy as np

from nova.audio.pipeline import AudioPipeline
from nova.audio.vad import SpeechSegmenter, VadEvent
from nova.audio.wake_word import WakeWordDetector
from nova.config.loader import get_config

FRAME = np.zeros(480, dtype=np.float32)  # 30ms of silence at 16kHz -- content doesn't matter, only shape


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeWakeWordModel:
    """Stands in for openwakeword.model.Model. Returns scores from a fixed list, one per call."""

    def __init__(self, scores, model_name="hey_jarvis"):
        self._scores = iter(scores)
        self._model_name = model_name

    def predict(self, frame_int16):
        return {self._model_name: next(self._scores, 0.0)}


class FakeCapture:
    """Stands in for AudioCapture: replays a fixed list of frames instead of reading a mic."""

    def __init__(self, frames):
        self._frames = frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def frames(self):
        yield from self._frames


# ---------------------------------------------------------------------------
# Wake word tests
# ---------------------------------------------------------------------------

def test_wake_word_triggers_above_threshold():
    cfg = get_config()
    detector = WakeWordDetector(config=cfg, model=FakeWakeWordModel([0.1, 0.9]))

    triggered1, score1 = detector.process(FRAME)
    triggered2, score2 = detector.process(FRAME)

    assert not triggered1
    assert score1 == 0.1
    assert triggered2
    assert score2 == 0.9


def test_wake_word_does_not_trigger_below_threshold():
    cfg = get_config()
    detector = WakeWordDetector(config=cfg, model=FakeWakeWordModel([0.2, 0.3, 0.49]))

    results = [detector.process(FRAME)[0] for _ in range(3)]
    assert results == [False, False, False]


# ---------------------------------------------------------------------------
# VAD / SpeechSegmenter tests
# ---------------------------------------------------------------------------

_SMALL_CFG = {
    "audio": {"sample_rate": 16000, "frame_ms": 30},
    "vad": {"min_silence_ms": 90, "min_speech_ms": 60},  # 3 frames silence, 2 frames speech
}


def test_speech_segmenter_full_start_to_end_cycle():
    # Pattern: 2 non-speech (ignored) -> 3 speech (triggers start on 2nd) -> 3 non-speech (ends on 3rd)
    pattern = iter([False, False, True, True, True, False, False, False])

    def fake_is_speech(frame, sample_rate):
        return next(pattern, False)

    segmenter = SpeechSegmenter(is_speech_fn=fake_is_speech, config=_SMALL_CFG)

    events = []
    utterance = None
    for _ in range(8):
        event, audio = segmenter.process_frame(FRAME)
        events.append(event)
        if event == VadEvent.SPEECH_ENDED:
            utterance = audio

    assert VadEvent.SPEECH_STARTED in events
    assert VadEvent.SPEECH_ENDED in events
    assert utterance is not None
    assert utterance.shape[0] > 0


def test_speech_segmenter_ignores_short_blips():
    # Only 1 speech frame (below min_speech_frames=2) -- should never start an utterance.
    pattern = iter([False, True, False, False, False])

    def fake_is_speech(frame, sample_rate):
        return next(pattern, False)

    segmenter = SpeechSegmenter(is_speech_fn=fake_is_speech, config=_SMALL_CFG)

    events = [segmenter.process_frame(FRAME)[0] for _ in range(5)]
    assert VadEvent.SPEECH_STARTED not in events
    assert VadEvent.SPEECH_ENDED not in events


def test_speech_segmenter_resets_after_utterance():
    pattern = iter([True, True, False, False, False])

    def fake_is_speech(frame, sample_rate):
        return next(pattern, False)

    segmenter = SpeechSegmenter(is_speech_fn=fake_is_speech, config=_SMALL_CFG)
    for _ in range(5):
        segmenter.process_frame(FRAME)

    assert segmenter._in_speech is False
    assert segmenter._buffer == []


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------

def test_pipeline_end_to_end_with_fakes():
    cfg = get_config()
    frames = [FRAME] * 10

    # Wake word fires on the 2nd frame; never fires again afterward.
    wake_word = WakeWordDetector(config=cfg, model=FakeWakeWordModel([0.1, 0.95] + [0.0] * 8))

    # Once listening, VAD sees 2 speech frames then goes silent -> ends the utterance quickly.
    call_count = {"n": 0}

    def fake_is_speech(frame, sample_rate):
        call_count["n"] += 1
        return call_count["n"] <= 2

    small_cfg = {
        "audio": {"sample_rate": 16000, "frame_ms": 30},
        "vad": {"min_silence_ms": 30, "min_speech_ms": 30},  # 1 frame each way
    }
    segmenter = SpeechSegmenter(is_speech_fn=fake_is_speech, config=small_cfg)

    captured = []
    pipeline = AudioPipeline(
        capture=FakeCapture(frames),
        wake_word=wake_word,
        segmenter=segmenter,
        on_utterance=lambda audio: captured.append(audio),
    )

    pipeline.run_forever()

    assert len(captured) == 1
    assert captured[0].shape[0] > 0


if __name__ == "__main__":
    # Allows `python tests/test_milestone1_audio.py` without pytest installed.
    tests = [
        test_wake_word_triggers_above_threshold,
        test_wake_word_does_not_trigger_below_threshold,
        test_speech_segmenter_full_start_to_end_cycle,
        test_speech_segmenter_ignores_short_blips,
        test_speech_segmenter_resets_after_utterance,
        test_pipeline_end_to_end_with_fakes,
    ]
    for t in tests:
        t()
        print(f"{t.__name__}: PASS")
