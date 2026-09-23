"""
Milestone 1 deliverable: run this on your Windows machine with a mic
attached. It prints (and logs) "Wake word detected" and "Speech
started"/"Speech ended" in real time -- no STT yet, that's Milestone 2.

Run:
    python scripts/run_milestone1.py

Say "hey nova" (or whatever `wake_word.model_name` you configure), then
speak; when you go quiet again you'll see the captured utterance length.

Note: this script needs real audio hardware and the Milestone 1 deps
(sounddevice, openwakeword, torch) installed -- it will not run inside
a sandbox with no microphone.
"""

import sys

from nova.audio.pipeline import AudioPipeline
from nova.config.loader import get_config
from nova.logging_setup import get_logger, setup_logging


def main() -> int:
    get_config()
    setup_logging()
    log = get_logger("milestone1")

    def on_utterance(audio):
        seconds = len(audio) / 16000
        print(f"[Milestone 1] Captured utterance: {seconds:.2f}s of audio")

    pipeline = AudioPipeline(on_utterance=on_utterance)

    try:
        pipeline.run_forever()
    except KeyboardInterrupt:
        log.info("Stopped by user.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
