# NOVA — Windows Voice Assistant

Modular, open-source, AI-powered voice assistant for Windows.
Speak → Command → Get Things Done.

This repo is being built milestone by milestone (see `NOVA_Milestones_Roadmap.md`
from our planning session). **This commit completes Milestone 0: Project Foundations.**

## What Milestone 0 delivers

- Python project skeleton laid out for every future pipeline stage
- Central, typed config in `config/config.yaml` — every module reads settings
  from here instead of hardcoding values
- Structured, per-stage logging (`nova/logging_setup.py`) that tags every log
  line with the stage that produced it (`nova.wake_word`, `nova.stt`, etc.) —
  this is what makes per-stage latency profiling possible later (Milestone 8)
- `main.py`: boots the app, loads config, sets up logging, logs "NOVA ready",
  and shuts down cleanly on Ctrl+C / SIGTERM
- A smoke-test suite proving the above actually works

Nothing here touches audio, models, or tools yet — that starts at Milestone 1.

## Project layout

```
nova/
├── main.py                      # entry point — boots the whole app
├── pyproject.toml               # dependencies, grouped per milestone (uv-managed)
├── config/
│   └── config.yaml               # single source of truth for all settings
├── logs/                         # rotating app logs land here (git-ignored)
├── nova/                         # the actual Python package
│   ├── config/
│   │   └── loader.py              # get_config() — cached YAML loader
│   ├── logging_setup.py          # setup_logging(), get_logger(stage)
│   ├── audio/                    # Milestone 1: capture, wake word, VAD
│   ├── brain/                    # Milestone 3-4: LLM, agent, memory/RAG
│   ├── tools/                    # Milestone 5-6: Windows + browser tools
│   ├── memory/                   # Milestone 4: ChromaDB-backed memory
│   └── tts/                      # Milestone 7: Piper text-to-speech
└── tests/
    └── test_milestone0_bootstrap.py
```

## Setup (uv)

Dependencies live in `pyproject.toml`, grouped into optional extras per
milestone (`audio`, `stt`, `brain`, `windows`, `browser`, `tts`, `dev`) so you
never install more than the milestone you're on needs.

```bash
uv sync --extra dev
```

That installs just the core (`pyyaml`) plus `pytest` — enough for everything
in this Milestone 0 README below. `uv sync` creates `.venv/` and a `uv.lock`
automatically; you don't need to create or activate a venv by hand.

When you get to later milestones, add their extras the same way, e.g.:
```bash
uv sync --extra audio --extra stt     # Milestones 1-2
uv sync --extra all                   # everything, for Milestone 8
```

## Run it

```bash
uv run python main.py
```

Expected output:

```
2026-09-19 10:00:00 | INFO     | nova.main    | Booting NOVA v0.0.1
2026-09-19 10:00:00 | INFO     | nova.main    | Wake phrase configured as: 'hey nova'
2026-09-19 10:00:00 | INFO     | nova.main    | NOVA ready.
```

Press `Ctrl+C` — it should log a clean shutdown message and exit with code 0,
not a stack trace.

## Test it

```bash
uv run pytest tests/ -v
```

All 4 smoke tests should pass, confirming:
- config loads and contains the keys `main.py` depends on
- config is cached (not re-parsed on every call)
- logging setup is idempotent (safe to call more than once)
- loggers are correctly namespaced per pipeline stage

## Milestone 1 — Audio Input Pipeline (mic → wake word → VAD)

Adds:
- `nova/audio/capture.py` — `AudioCapture`, a `sounddevice`-backed producer/consumer stream
- `nova/audio/wake_word.py` — `WakeWordDetector`, wraps `openWakeWord`
- `nova/audio/vad.py` — `SpeechSegmenter`, a state machine that buffers an utterance and tells you when it starts/ends, backed by Silero VAD
- `nova/audio/pipeline.py` — `AudioPipeline`, wires the three together: IDLE (listening for wake word) → LISTENING (buffering until VAD says speech ended)
- `scripts/run_milestone1.py` — run this on your machine with a mic; prints "Wake word detected" / "Speech started" / "Speech ended" live
- `tests/test_milestone1_audio.py` — 6 tests, all using fake models (no mic, no model weights needed)

Every real model (`AudioCapture`'s `sounddevice.InputStream`, `WakeWordDetector`'s `openwakeword.Model`, `SpeechSegmenter`'s Silero VAD via `load_silero_vad()`) is dependency-injectable, which is how the tests run without hardware or downloaded weights — and it's also what lets you swap in a different wake-word engine later without touching `AudioPipeline`.

Run it (on your machine, mic required):
```bash
uv sync --extra audio
uv run python scripts/run_milestone1.py
```

## Milestone 2 — Speech-to-Text

Adds:
- `nova/audio/transcriber.py` — `Transcriber`, wraps `faster-whisper`, takes the utterance audio `SpeechSegmenter` produces and returns plain text
- `scripts/run_milestone2.py` — the full Milestone 1+2 loop: mic → wake word → VAD → transcribed text printed to console
- `tests/test_milestone2_stt.py` — 4 tests using a fake Whisper model (no model download needed)

Run it (on your machine, mic required):
```bash
uv sync --extra audio --extra stt
uv run python scripts/run_milestone2.py
```

## Running the tests (all milestones)

```bash
uv run pytest tests/ -v
```
14 tests total (4 Milestone 0 + 6 Milestone 1 + 4 Milestone 2), all passing without a microphone or any downloaded model weights, since every model dependency is injected.

## Next: Milestone 3

The NOVA Brain — local LLM via Ollama (llama3.1), an agent/planning loop, and tool-selection routing. Say the word and I'll build it the same way.
