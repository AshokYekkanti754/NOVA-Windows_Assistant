"""
nova.audio.capture
-------------------
Wraps sounddevice's callback-based InputStream behind a simple
producer/consumer interface: start a stream, then pull frames off a
queue at your own pace via `.frames()`.

The audio callback itself does nothing but push into the queue -- no
model inference happens on the real-time audio thread, which is what
keeps capture glitch-free.
"""

from __future__ import annotations

import queue
from typing import Callable, Iterator, Optional

import numpy as np

from nova.config.loader import get_config
from nova.logging_setup import get_logger

log = get_logger("audio.capture")


class AudioCapture:
    def __init__(self, config: Optional[dict] = None):
        cfg = config or get_config()
        audio_cfg = cfg["audio"]

        self.sample_rate: int = audio_cfg["sample_rate"]
        self.channels: int = audio_cfg["channels"]
        self.frame_ms: int = audio_cfg["frame_ms"]
        self.frame_samples: int = int(self.sample_rate * self.frame_ms / 1000)
        self.device_index = audio_cfg.get("input_device_index")

        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream = None  # type: ignore[assignment]  # sounddevice.InputStream, imported lazily

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.warning("Audio stream status: %s", status)

        audio = indata[:, 0].copy()

        self._queue.put(audio)

    def start(self) -> None:
        import sounddevice as sd  # imported lazily: heavy + requires real audio hardware

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            blocksize=self.frame_samples,
            dtype="float32",
            device=self.device_index,
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            "Audio capture started (sample_rate=%d, frame_ms=%d, device=%s)",
            self.sample_rate,
            self.frame_ms,
            self.device_index if self.device_index is not None else "default",
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            log.info("Audio capture stopped")

    def frames(
        self,
        should_stop: Optional[Callable[[], bool]] = None,
        poll_interval: float = 0.1,
    ) -> Iterator[np.ndarray]:
        """
        Generator that yields one frame (float32 numpy array) at a time.

        If `should_stop` is given it is checked between frames (and while
        waiting), letting the consumer shut down cleanly. This also uses a
        timed queue.get(): an untimed get() blocks on a lock acquire that
        Windows Ctrl+C cannot interrupt.
        """
        while True:
            if should_stop is not None and should_stop():
                return

            try:
                yield self._queue.get(timeout=poll_interval)
            except queue.Empty:
                continue

    def __enter__(self) -> "AudioCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @staticmethod
    def list_input_devices() -> None:
        """Prints available microphones and their device index, for config.yaml."""
        import sounddevice as sd

        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                print(i, dev["name"])
