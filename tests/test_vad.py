import numpy as np

from nova.audio.capture import AudioCapture
from nova.audio.vad import SpeechSegmenter, VadEvent, load_silero_vad


vad_fn = load_silero_vad()
segmenter = SpeechSegmenter(vad_fn)

capture = AudioCapture()

try:
    capture.start()

    print("🎤 VAD test started")
    print("Speak normally, then stop speaking.")
    print("Press Ctrl+C to stop.")

    for frame in capture.frames():
        event, utterance = segmenter.process_frame(frame)

        if event == VadEvent.SPEECH_STARTED:
            print("\n🗣️ Speech started")

        elif event == VadEvent.SPEECH_ENDED:
            duration = len(utterance) / capture.sample_rate

            print(
                f"\n🔇 Speech ended "
                f"(duration={duration:.2f}s, "
                f"samples={len(utterance)})"
            )

except KeyboardInterrupt:
    print("\nStopping VAD...")

finally:
    capture.stop()