"""
nova.audio.pipeline
--------------------
NOVA's real-time audio pipeline:

    IDLE
      ↓
    Wake Word Detection
      ↓
    LISTENING
      ↓
    Silero VAD + SpeechSegmenter
      ↓
    SPEECH_ENDED
      ↓
    on_utterance(audio)

The pipeline is dependency-injected so it can be unit-tested without
microphone hardware or real ML models.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from nova.audio.capture import AudioCapture
from nova.audio.vad import (
    SpeechSegmenter,
    VadEvent,
    load_silero_vad,
)
from nova.audio.wake_word import WakeWordDetector
from nova.logging_setup import get_logger

log = get_logger("audio.pipeline")


class PipelineState(Enum):
    IDLE = auto()
    LISTENING = auto()


class AudioPipeline:
    def __init__(
        self,
        capture: Optional[AudioCapture] = None,
        wake_word: Optional[WakeWordDetector] = None,
        segmenter: Optional[SpeechSegmenter] = None,
        on_utterance: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.capture = capture or AudioCapture()
        self.wake_word = wake_word or WakeWordDetector()
        self._segmenter = segmenter
        self.on_utterance = on_utterance
        self.state = PipelineState.IDLE

    def _ensure_segmenter(self) -> SpeechSegmenter:
        if self._segmenter is None:
            self._segmenter = SpeechSegmenter(
                is_speech_fn=load_silero_vad()
            )

        return self._segmenter

    def run_forever(
        self,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        """
        Runs the pipeline until the audio source ends or `should_stop()`
        returns True (checked between frames). `should_stop` is a zero-arg
        callable -- typically a lambda reading a shutdown flag set by a
        signal handler.
        """
        segmenter = self._ensure_segmenter()

        log.info(
            "Audio pipeline running. Say the wake word to begin."
        )

        with self.capture:
            # Only pass should_stop when given, so injected fakes with a
            # zero-arg frames() (tests) remain compatible.
            if should_stop is not None:
                frames_iter = self.capture.frames(should_stop=should_stop)
            else:
                frames_iter = self.capture.frames()

            for frame in frames_iter:

                if should_stop is not None and should_stop():
                    log.info("Shutdown requested. Stopping audio pipeline.")
                    return

                # ---------------------------------------------
                # IDLE: waiting for wake word
                # ---------------------------------------------

                if self.state == PipelineState.IDLE:

                    triggered, score = self.wake_word.process(frame)

                    log.debug(
                        "Wake score: %.3f",
                        score,
                    )

                    if triggered:
                        log.info("Wake word detected.")

                        self.state = PipelineState.LISTENING
                        segmenter.reset()
                        self.wake_word.reset()

                # ---------------------------------------------
                # LISTENING: collecting speech
                # ---------------------------------------------

                elif self.state == PipelineState.LISTENING:

                    event, utterance = segmenter.process_frame(frame)

                    if event == VadEvent.SPEECH_STARTED:
                        log.info("Speech started.")

                    elif (
                        event == VadEvent.SPEECH_ENDED
                        and utterance is not None
                    ):
                        log.info(
                            "Speech ended. Captured %.2f seconds.",
                            len(utterance)
                            / segmenter.sample_rate,
                        )

                        # Return to wake-word mode first.
                        self.state = PipelineState.IDLE

                        if self.on_utterance is not None:
                            self.on_utterance(utterance)

                        segmenter.reset()