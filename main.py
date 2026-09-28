"""
NOVA — entry point.

Starts the integrated audio pipeline:

    Microphone
        ↓
    Wake Word
        ↓
    VAD
        ↓
    Speech Segment
        ↓
    Faster-Whisper
        ↓
    NovaBrain
        ↓
    Response / Tool Execution
"""

from __future__ import annotations

import signal
import sys
import types

from nova.audio.pipeline import AudioPipeline
from nova.audio.transcriber import Transcriber
from nova.brain.brain import NovaBrain
from nova.config.loader import get_config
from nova.logging_setup import setup_logging, get_logger


_shutdown_requested = False


def _handle_shutdown_signal(
    signum: int,
    frame: types.FrameType | None,
) -> None:
    global _shutdown_requested

    if _shutdown_requested:
        raise KeyboardInterrupt

    _shutdown_requested = True
    print(
        "\nShutdown requested — finishing current work. "
        "Press Ctrl+C again to force quit."
    )


def main() -> int:
    global _shutdown_requested

    # 1. Load configuration
    cfg = get_config()

    # 2. Setup logging
    setup_logging()
    log = get_logger("main")

    # 3. Read app configuration
    app_name = cfg["app"]["name"]
    app_version = cfg["app"]["version"]
    wake_phrase = cfg["app"]["wake_phrase"]

    # 4. Setup shutdown signals
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    log.info("Booting %s v%s", app_name, app_version)
    log.info("Wake phrase configured as: '%s'", wake_phrase)

    # 5. Create the Faster-Whisper transcriber
    transcriber = Transcriber(config=cfg)

    # 6. Create NOVA's brain
    brain = NovaBrain()
    session_id = "live-session"

    # 7. Handle completed speech from AudioPipeline
    def handle_utterance(audio):
        log.info(
            "Transcribing %.2f seconds of audio...",
            len(audio) / cfg["audio"]["sample_rate"],
        )

        transcript = transcriber.transcribe(
            audio,
            sample_rate=cfg["audio"]["sample_rate"],
        )

        if not transcript:
            log.info("No speech detected in utterance.")
            return

        print(f"\n🗣️ You: {transcript}")
        log.info("NOVA heard: %s", transcript)

        # Send the transcript to NovaBrain
        response = brain.think(
            session_id=session_id,
            user_text=transcript,
        )

        print(f"🤖 NOVA: {response}")

    # 8. Create the complete audio pipeline
    pipeline = AudioPipeline(
        on_utterance=handle_utterance
    )

    log.info("%s ready.", app_name)

    # 9. Start listening
    try:
        pipeline.run_forever(
            should_stop=lambda: _shutdown_requested
        )

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received.")

    finally:
        log.info("%s stopped cleanly. Goodbye.", app_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())