"""
nova.audio.wake_word
---------------------
Wrapper around openWakeWord with buffering so that microphone frames
are large enough for wake-word inference.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from nova.config.loader import get_config
from nova.logging_setup import get_logger

log = get_logger("audio.wake_word")


class WakeWordDetector:
    def __init__(self, config: Optional[dict] = None, model=None):
        cfg = config or get_config()
        ww_cfg = cfg["wake_word"]
        audio_cfg = cfg["audio"]

        self.model_name: str = ww_cfg["model_name"]
        self.threshold: float = ww_cfg["threshold"]

        self.sample_rate: int = audio_cfg["sample_rate"]

        # openWakeWord normally expects 16 kHz audio.
        # 1280 samples = 80 ms at 16 kHz.
        self.required_samples = 1280

        self._buffer = np.array([], dtype=np.float32)

        self._model = model

    def _ensure_model(self):
        if self._model is None:
            from openwakeword.model import Model

            self._model = Model(
                wakeword_models=[self.model_name],
                inference_framework="onnx",
            )

            log.info(
                "Loaded openWakeWord model '%s' using ONNX",
                self.model_name,
            )

        return self._model

    def reset(self) -> None:
        """Clear buffered audio."""
        self._buffer = np.array([], dtype=np.float32)

    def process(self, frame_float32: np.ndarray) -> Tuple[bool, float]:
        """
        Add a microphone frame to the buffer and run wake-word
        inference when enough samples are available.
        """

        # Add new microphone audio to our buffer.
        self._buffer = np.concatenate(
            [self._buffer, frame_float32]
        )

        # Not enough audio yet.
        if len(self._buffer) < self.required_samples:
            return False, 0.0

        # Take exactly one chunk for openWakeWord.
        chunk = self._buffer[:self.required_samples]

        # Keep remaining samples for the next inference.
        self._buffer = self._buffer[self.required_samples:]

        model = self._ensure_model()

        # float32 [-1, 1] → int16
        frame_int16 = (
            np.clip(chunk, -1.0, 1.0) * 32767
        ).astype(np.int16)

        predictions = model.predict(frame_int16)

        score = float(
            predictions.get(self.model_name, 0.0)
        )

        triggered = score >= self.threshold

        log.debug(
            "Wake word '%s' score=%.3f",
            self.model_name,
            score,
        )

        if triggered:
            log.info(
                "Wake word '%s' detected (score=%.2f)",
                self.model_name,
                score,
            )

        return triggered, score