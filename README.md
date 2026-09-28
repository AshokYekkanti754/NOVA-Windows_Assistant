# NOVA Brain Update -- new/changed files only

This is a **delta package**, not a full project. Drop these into your
existing `nova/` folder (the one from `nova_stages1-4.zip`), preserving
paths -- everything here either creates a new file or overwrites one that
changed.

## New files
```
nova/brain/llm_client.py     -- wraps the Ollama chat API
nova/brain/planner.py        -- decides answer-directly vs call-a-tool
nova/brain/router.py         -- tool registry + executor
nova/brain/brain.py          -- NovaBrain: wires it all together (Stage 5's entry point)
nova/memory/context_store.py -- Short-term Context (SQLite)
nova/memory/long_term_store.py -- Long-term Memory (SQLite)
nova/memory/rag_store.py     -- RAG Knowledge (ChromaDB + sentence-transformers)
nova/memory/manager.py       -- Memory Manager: orchestrates the 3 stores above
scripts/run_milestone3.py    -- text chat loop with a mock tool
scripts/run_milestone4.py    -- persistent-memory demo (restart the script, it remembers)
tests/test_milestone3_brain.py   -- 10 tests, all fakes, no Ollama needed
tests/test_milestone4_memory.py  -- 13 tests, all fakes/in-memory, no ChromaDB/model needed
```

## Changed files
```
config/config.yaml  -- added brain.memory.short_term_db_path and long_term_db_path
```
If you've customized your own `config.yaml`, just add these two lines under
the existing `brain.memory:` section instead of overwriting the whole file:
```yaml
    short_term_db_path: "./data/short_term.db"
    long_term_db_path: "./data/long_term.db"
```

## After copying these in

```bash
uv sync --extra brain --extra dev
uv run pytest tests/ -v          # all 37 tests (14 old + 23 new) should pass
```

You'll also need a local Ollama server for the two demo scripts (not for
the tests -- those use fakes):
```bash
ollama pull llama3.1
uv run python scripts/run_milestone3.py
uv run python scripts/run_milestone4.py
```

See the chat response for the full explanation of how each piece works and
how they connect.
