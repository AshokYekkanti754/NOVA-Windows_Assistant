"""
nova.audio.vad
---------------
Two things live here:

1. `load_silero_vad()` — loads the real Silero VAD model and wraps it as
   a simple `(frame, sample_rate) -> is_speech: bool` function.
2. `SpeechSegmenter` — a small state machine that consumes frames one at
   a time (after the wake word has fired) and tells you when an
   utterance has started and ended, buffering the audio for you.

`SpeechSegmenter` takes the speech-detector as an injected function
(`is_speech_fn`) rather than importing Silero directly, so it can be
unit-tested with a fake detector and no model weights or audio hardware.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable, List, Optional, Tuple

import numpy as np

from nova.config.loader import get_config
from nova.logging_setup import get_logger

log = get_logger("audio.vad")

FrameVadFn = Callable[[np.ndarray, int], bool]


class VadEvent(Enum):
    NONE = auto()
    SPEECH_STARTED = auto()
    SPEECH_ENDED = auto()


def load_silero_vad() -> FrameVadFn:
    """Loads the real Silero VAD model via torch.hub and returns a per-frame is_speech function."""
    import torch  # heavy import, deferred

    model, _utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )

    def _is_speech(frame: np.ndarray, sample_rate: int) -> bool:
        tensor = torch.from_numpy(frame.astype(np.float32))
        prob = model(tensor, sample_rate).item()
        return prob > 0.5

    return _is_speech


class SpeechSegmenter:
    def __init__(self, is_speech_fn: FrameVadFn, config: Optional[dict] = None):
        cfg = config or get_config()
        vad_cfg = cfg["vad"]
        audio_cfg = cfg["audio"]

        self.sample_rate: int = audio_cfg["sample_rate"]
        self.frame_ms: int = audio_cfg["frame_ms"]
        self.min_silence_frames = max(1, vad_cfg["min_silence_ms"] // self.frame_ms)
        self.min_speech_frames = max(1, vad_cfg["min_speech_ms"] // self.frame_ms)

        self._is_speech_fn = is_speech_fn
        self._buffer: List[np.ndarray] = []
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._in_speech = False

    def reset(self) -> None:
        self._buffer.clear()
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._in_speech = False

    def process_frame(self, frame: np.ndarray) -> Tuple[VadEvent, Optional[np.ndarray]]:
        """
        Feed one frame. Returns (event, utterance_audio).
        utterance_audio is only populated on a SPEECH_ENDED event.
        """
        is_speech = self._is_speech_fn(frame, self.sample_rate)

        log.debug(
            "VAD: %s",
            "SPEECH" if is_speech else "SILENCE",
        )

        if not self._in_speech:
            if is_speech:
                self._speech_frame_count += 1
                self._buffer.append(frame)
                if self._speech_frame_count >= self.min_speech_frames:
                    self._in_speech = True
                    self._silence_frame_count = 0
                    log.info("Speech started")
                    return VadEvent.SPEECH_STARTED, None
                return VadEvent.NONE, None
            else:
                # Not enough consecutive speech frames yet -- treat as noise, drop it.
                self._speech_frame_count = 0
                self._buffer.clear()
                return VadEvent.NONE, None

        # Currently inside an utterance.
        self._buffer.append(frame)
        if is_speech:
            self._silence_frame_count = 0
            return VadEvent.NONE, None

        self._silence_frame_count += 1
        if self._silence_frame_count >= self.min_silence_frames:
            utterance = np.concatenate(self._buffer)
            log.info("Speech ended (%.2fs captured)", len(utterance) / self.sample_rate)
            self.reset()
            return VadEvent.SPEECH_ENDED, utterance

        return VadEvent.NONE, None
