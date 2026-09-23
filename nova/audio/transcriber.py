"""
nova.audio.transcriber
------------------------
Wraps faster-whisper so the rest of NOVA just calls `.transcribe(audio)`
and gets plain text back, instead of dealing with segment iterators.

The model is loaded lazily (it's a multi-hundred-MB download/load) and
can be dependency-injected for tests via the `model` constructor argument.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from nova.config.loader import get_config
from nova.logging_setup import get_logger

log = get_logger("audio.stt")


class Transcriber:
    def __init__(self, config: Optional[dict] = None, model=None):
        cfg = config or get_config()
        stt_cfg = cfg["stt"]

        self.model_size: str = stt_cfg["model_size"]
        self.device: str = stt_cfg["device"]
        self.compute_type: str = stt_cfg["compute_type"]
        self._model = model  # injected fake in tests; real faster_whisper.WhisperModel otherwise

    def _ensure_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # heavy import, deferred

            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            log.info(
                "Loaded faster-whisper model '%s' (device=%s, compute_type=%s)",
                self.model_size,
                self.device,
                self.compute_type,
            )
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        audio: mono float32 numpy array in range [-1, 1] at `sample_rate` Hz
               (exactly what SpeechSegmenter hands back on SPEECH_ENDED).
        Returns the concatenated transcript text, or "" for empty/silent audio.
        """
        if audio.size == 0:
            return ""

        model = self._ensure_model()
        segments, info = model.transcribe(audio, language=None, vad_filter=False)
        text_parts = [seg.text.strip() for seg in segments]
        transcript = " ".join(part for part in text_parts if part)

        log.info(
            "Transcribed %.2fs of audio -> %r (language=%s)",
            len(audio) / sample_rate,
            transcript,
            getattr(info, "language", "unknown"),
        )
        return transcript
