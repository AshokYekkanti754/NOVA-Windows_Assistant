"""
Milestone 2 deliverable: say "hey nova, what time is it" (or anything)
on your own machine and see the transcribed text printed to the console.
This is the full Milestone 1 + 2 loop -- mic -> wake word -> VAD ->
faster-whisper -> printed text.

Run:
    python scripts/run_milestone2.py

Requires the Milestone 1 deps (sounddevice, openwakeword, torch) plus
faster-whisper, and real audio hardware -- will not run in a sandbox.
"""

import sys

from nova.audio.pipeline import AudioPipeline
from nova.audio.transcriber import Transcriber
from nova.config.loader import get_config
from nova.logging_setup import get_logger, setup_logging


def main() -> int:
    get_config()
    setup_logging()
    log = get_logger("milestone2")

    transcriber = Transcriber()

    def on_utterance(audio):
        text = transcriber.transcribe(audio)
        if text:
            print(f"[Milestone 2] You said: {text!r}")
        else:
            print("[Milestone 2] (heard silence / nothing recognizable)")

    pipeline = AudioPipeline(on_utterance=on_utterance)

    try:
        pipeline.run_forever()
    except KeyboardInterrupt:
        log.info("Stopped by user.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
