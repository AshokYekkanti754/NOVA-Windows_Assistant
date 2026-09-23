"""
nova.audio.wake_word
---------------------
Thin wrapper around openWakeWord so the rest of the pipeline depends on
NOVA's own interface, not directly on the third-party library's API shape.

The real model is loaded lazily on first use (it's a heavy import and
needs no hardware, just weights) and can be dependency-injected for
tests via the `model` constructor argument.
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

        self.model_name: str = ww_cfg["model_name"]
        self.threshold: float = ww_cfg["threshold"]
        self._model = model  # injected fake in tests; real openwakeword.Model otherwise

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

    def process(self, frame_float32: np.ndarray) -> Tuple[bool, float]:
        """
        Feed one frame of mono float32 audio (range -1..1).
        Returns (triggered, score) for the configured wake word.
        """
        model = self._ensure_model()
        frame_int16 = (np.clip(frame_float32, -1.0, 1.0) * 32767).astype(np.int16)
        predictions = model.predict(frame_int16)
        score = float(predictions.get(self.model_name, 0.0))
        triggered = score >= self.threshold

        if triggered:
            log.info("Wake word '%s' detected (score=%.2f)", self.model_name, score)

        return triggered, score
