# NOVA — Full System Architecture

This document maps every box in the architecture diagram to a concrete module,
data structure, and technology choice. It's the reference to come back to as
we build each stage — the roadmap says *when*, this says *what's actually
inside each box*.

---

## 1. The Pipeline at a Glance

```
┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────┐   ┌─────────────┐
│1.Microph.│──▶│2.Wake Word│──▶│ 3. VAD   │──▶│4.Speech-to-  │──▶│   5. NOVA BRAIN   │──▶│6.Tool Modules│──▶│7.Final Answer│──▶│8.Text-to-    │──▶│9.Audio Output│
│sounddevice│  │openWakeWord│  │Silero VAD│   │Text          │   │Planner+Memory+    │   │Windows/Browser│  │(NL generation│   │Speech (Piper)│   │(sounddevice) │
│          │   │"Hey Nova" │   │          │   │faster-whisper│   │Tool Router        │   │/General tools │  │for TTS)      │   │              │   │              │
└──────────┘   └───────────┘   └──────────┘   └──────────────┘   └──────────────────┘   └──────────────┘   └─────────────┘   └──────────────┘   └─────────────┘
   Stage 1        Stage 2         Stage 3          Stage 4              Stage 5               Stage 6            Stage 7           Stage 8            Stage 9
 Milestone 0-1  Milestone 1    Milestone 1      Milestone 2          Milestone 3-4        Milestone 5-6      Milestone 7        Milestone 7       Milestone 7
```

Every arrow is a **plain Python object handoff**, not a network call — the
whole pipeline runs in one process (multi-threaded for the audio callback,
otherwise sequential). This matters: it's why latency is dominated by model
inference time, not I/O, and why Milestone 8's profiling targets per-stage
*compute* time.

---

## 2. Stage-by-Stage Detail

### Stage 1 — Microphone
| | |
|---|---|
| **Module** | `nova/audio/capture.py` → `AudioCapture` |
| **Tech** | `sounddevice.InputStream`, callback-based |
| **Input** | Raw analog audio from the OS default (or configured) input device |
| **Output** | A continuous stream of 30ms float32 frames (480 samples @ 16kHz), pushed onto a `queue.Queue` |
| **Key design choice** | The audio callback thread only ever does `queue.put()` — no inference on that thread, or you get audible glitches |

### Stage 2 — Wake Word
| | |
|---|---|
| **Module** | `nova/audio/wake_word.py` → `WakeWordDetector` |
| **Tech** | `openWakeWord` (ONNX runtime under the hood) |
| **Input** | One 30ms frame at a time (converted to int16 PCM) |
| **Output** | `(triggered: bool, score: float)` — `openWakeWord` internally keeps its own sliding window across frames, so NOVA doesn't have to buffer anything at this stage |
| **Key design choice** | Runs fully offline, on-device, on every single frame all day — must be small and fast (this is why it's a purpose-trained small model, not the LLM) |

### Stage 3 — VAD (Voice Activity Detection)
| | |
|---|---|
| **Module** | `nova/audio/vad.py` → `SpeechSegmenter` |
| **Tech** | Silero VAD (via `torch.hub`) |
| **Input** | Frames, one at a time, only while in the `LISTENING` state (after wake word fires) |
| **Output** | `VadEvent.SPEECH_STARTED` / `SPEECH_ENDED`, plus the buffered utterance audio on `SPEECH_ENDED` |
| **Key design choice** | `min_speech_ms` filters short noise blips (door clicks, keyboard clacks) before committing to "listening"; `min_silence_ms` decides how long a pause has to be before NOVA assumes you're done talking |

### Stage 4 — Speech-to-Text
| | |
|---|---|
| **Module** | `nova/audio/transcriber.py` → `Transcriber` |
| **Tech** | `faster-whisper` (CTranslate2-accelerated Whisper), `model_size="small"`, `compute_type="int8"` |
| **Input** | The full utterance audio array from Stage 3 |
| **Output** | Plain transcript text, plus detected language |
| **Key design choice** | Whisper's own built-in VAD is disabled (`vad_filter=False`) — Stage 3 already found the speech boundaries; running VAD twice just risks clipping short commands |

### Stage 5 — NOVA Brain (the core reasoning layer)

This is the most complex stage, and it has its own internal architecture:

```
                         ┌───────────────────────────────────────────────┐
                         │                 5. NOVA BRAIN                  │
                         │                                                 │
   transcript ──────────▶│   ┌─────────┐   ┌───────────────┐   ┌────────┐ │
                         │   │ Planner │──▶│ Memory Manager │──▶│  Tool  │ │──▶ to Stage 6
                         │   │(break   │   │ (reads/writes  │   │ Router │ │    or Stage 7
                         │   │ down the│   │  the layer     │   │(picks  │ │
                         │   │ request)│   │  below)        │   │ tool(s)│ │
                         │   └─────────┘   └───────────────┘   └────────┘ │
                         │                         │                       │
                         │        ┌────────────────┼────────────────┐     │
                         │        ▼                ▼                ▼     │
                         │  ┌──────────┐    ┌─────────────┐   ┌──────────┐│
                         │  │Short-term│    │ Long-term   │   │   RAG    ││
                         │  │ Context  │    │  Memory     │   │Knowledge ││
                         │  │(current  │    │(preferences,│   │(PDFs,docs││
                         │  │conv,task,│    │project paths│   │code,notes││
                         │  │browser   │    │important    │   │external  ││
                         │  │ state)   │    │  facts)     │   │knowledge)││
                         │  │ SQLite   │    │  SQLite     │   │ ChromaDB ││
                         │  │          │    │             │   │+sentence-││
                         │  │          │    │             │   │transform.││
                         │  └──────────┘    └─────────────┘   └──────────┘│
                         └───────────────────────────────────────────────┘
```

| Component | Module (planned) | Tech | Purpose |
|---|---|---|---|
| **Planner** | `nova/brain/planner.py` | Ollama (llama3.1) + prompt template | Breaks the transcript into a plan: answer directly, or call tool(s) first? |
| **Memory Manager** | `nova/memory/manager.py` | Coordinates the 3 stores below | Decides *which* memory to read/write for a given turn |
| **Short-term Context** | `nova/memory/context_store.py` | SQLite | Current conversation turns, active task state, current browser tab/state — cleared or rotated per session |
| **Long-term Memory** | `nova/memory/long_term_store.py` | SQLite | User preferences, project paths, "remember that I..." facts — persists across sessions indefinitely |
| **RAG Knowledge** | `nova/memory/rag_store.py` | ChromaDB + `sentence-transformers` embeddings | Semantic search over documents/code/notes the user has fed NOVA — retrieved by similarity, not exact match |
| **Tool Router** | `nova/brain/router.py` | LLM function-calling / ReAct loop | Given the plan + retrieved memory, decides which Stage 6 tool(s) to actually invoke, and with what arguments |

**Why two separate SQLite stores instead of one?** Short-term context is
cheap to discard (session-scoped) and needs to be fast to read every turn.
Long-term memory needs to survive restarts and is written far less often.
Splitting them means clearing short-term context (e.g. "forget this
conversation") never risks touching a stored user preference.

**Why ChromaDB only for RAG Knowledge, not the other two?** Preferences and
short-term context are looked up by *key* ("what's my name", "what was I just
doing") — a plain SQL row lookup is faster and simpler than a vector search
for that. RAG Knowledge is looked up by *meaning* ("what does this codebase
do") — that's exactly what an embedding + vector similarity search is for.

### Stage 6 — Tool Modules
| Category | Module (planned) | Tech | Examples |
|---|---|---|---|
| **Windows Tools** | `nova/tools/windows_tools.py` | `pyautogui` + `pywinauto` | Open/close apps, file & folder ops, volume/brightness, screenshots, keyboard/mouse automation, lock/shutdown, clipboard, process management, app search, Windows settings |
| **Browser Tools** | `nova/tools/browser_tools.py` | `Playwright` | Open URL, search web, navigate pages, fill forms/click elements, extract information |
| **General Tools** | `nova/tools/general_tools.py` | mixed | Screenshots, media control, anything that doesn't cleanly fit Windows or Browser |

Every tool call passes through a **permission layer** (`config.yaml`'s
`tools.*.allow_list` / `require_confirmation`) before executing — this is
the "Secure: controlled tool execution & permissions" pillar from the
diagram's footer.

### Stage 7 — Final Answer
| | |
|---|---|
| **Module (planned)** | `nova/brain/responder.py` |
| **Tech** | Ollama (llama3.1), same model as the Planner but a different prompt |
| **Input** | The original transcript + whatever the Tool Router's tool calls returned |
| **Output** | A natural-language response string, ready for TTS |

### Stage 8 — Text-to-Speech
| | |
|---|---|
| **Module (planned)** | `nova/tts/synthesizer.py` |
| **Tech** | Piper (`en_US-lessac-medium` voice by default) |
| **Input** | The Stage 7 response text |
| **Output** | A raw audio waveform (numpy array / wav bytes) |

### Stage 9 — Audio Output
| | |
|---|---|
| **Module (planned)** | Reuses `sounddevice` (output stream this time, mirroring Stage 1's input stream) |
| **Tech** | `sounddevice.OutputStream` or `sd.play()` |
| **Input** | Stage 8's waveform |
| **Output** | Sound, out of your speakers |

---

## 3. Design Principles → How the Repo Enforces Them

| Diagram pillar | How it's actually implemented |
|---|---|
| **Modular & Extensible** | Every stage is its own module with a narrow interface (`process(frame)`, `transcribe(audio)`, etc.); `nova/tools/` is designed so a new tool is a new function registered with the Tool Router, not a change to core pipeline code |
| **Secure** | `config.yaml`'s `allow_list` / `require_confirmation` gate every Windows/Browser tool call; nothing executes against your OS without passing through that layer |
| **Open Source** | Every dependency (openWakeWord, Silero, faster-whisper, Ollama, ChromaDB, Playwright, Piper) is open-source and runs fully offline/local — no cloud API keys required anywhere in the pipeline |
| **Built for Windows** | `pywinauto` and the Windows Tools module are Windows-native; `sys_platform == "win32"` guards in `pyproject.toml` make that explicit |

---

## 4. Current Implementation Status

| Stage | Status | Where |
|---|---|---|
| 1. Microphone | ✅ Built & tested | `nova/audio/capture.py` |
| 2. Wake Word | ✅ Built & tested | `nova/audio/wake_word.py` |
| 3. VAD | ✅ Built & tested | `nova/audio/vad.py` |
| 4. Speech-to-Text | ✅ Built & tested | `nova/audio/transcriber.py` |
| 5. NOVA Brain | ⬜ Not started (Milestone 3-4) | — |
| 6. Tool Modules | ⬜ Not started (Milestone 5-6) | — |
| 7. Final Answer | ⬜ Not started (Milestone 7) | — |
| 8. Text-to-Speech | ⬜ Not started (Milestone 7) | — |
| 9. Audio Output | ⬜ Not started (Milestone 7) | — |

Stages 1-4 (Milestones 0-2) are the code + tests already in your
`nova_milestone2.zip` delivery, orchestrated by `nova/audio/pipeline.py`'s
`AudioPipeline` state machine (`IDLE` → `LISTENING` → transcribe → back to
`IDLE`).

## 5. What's Next

Stage 5 (the Brain) is the biggest jump in complexity — it's also where the
Planner / Memory Manager / Tool Router / three-store memory layer from
Section 2 gets built out for real, module by module.
