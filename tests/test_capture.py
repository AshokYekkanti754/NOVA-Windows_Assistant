from nova.audio.capture import AudioCapture

capture = AudioCapture()

try:
    capture.start()

    print("🎤 NOVA microphone is listening...")
    print("Press Ctrl+C to stop.")

    for frame in capture.frames():
        print(
            f"samples={len(frame)}, "
            f"dtype={frame.dtype}, "
            f"min={frame.min():.3f}, "
            f"max={frame.max():.3f}"
        )

except KeyboardInterrupt:
    print("\nStopping microphone...")

finally:
    capture.stop()