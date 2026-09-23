from nova.audio.capture import AudioCapture
from nova.audio.wake_word import WakeWordDetector


capture = AudioCapture()
detector = WakeWordDetector()

try:
    capture.start()

    print("🎤 NOVA is listening for the wake word...")
    print("Press Ctrl+C to stop.")

    for frame in capture.frames():
        triggered, score = detector.process(frame)

        print(f"Wake score: {score:.3f}", end="\r")

        if triggered:
            print("\n🔥 WAKE WORD DETECTED!")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    capture.stop()