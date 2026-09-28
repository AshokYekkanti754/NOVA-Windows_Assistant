"""
Milestone 4 deliverable: prove NOVA remembers something across sessions.

Run it once, say:
    You: remember that my favorite color is blue

Then quit (Ctrl+C or 'quit') and run this script again -- it prints
everything already remembered on startup, and you can ask:
    You: what is my favorite color

This works because LongTermMemory is backed by a SQLite file
(config.yaml's brain.memory.long_term_db_path) that persists on disk
between runs, unlike ShortTermContext which is conversation-scoped.

Requires a local Ollama server running (see run_milestone3.py's docstring).

Run:
    uv sync --extra brain
    uv run python scripts/run_milestone4.py
"""

import sys

from nova.brain.brain import NovaBrain
from nova.config.loader import get_config
from nova.logging_setup import get_logger, setup_logging


def main() -> int:
    get_config()
    setup_logging()
    log = get_logger("milestone4")

    brain = NovaBrain()
    session_id = "milestone4-demo"

    existing_facts = brain.memory.long_term.recall_all()
    if existing_facts:
        print("NOVA already remembers:")
        for fact in existing_facts:
            print(f"  - {fact}")
    else:
        print("NOVA doesn't remember anything yet. Try: 'remember that my favorite color is blue'")

    print("\nNOVA Brain -- persistent memory demo (type 'quit' to exit)")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_text:
            continue
        if user_text.lower() in ("quit", "exit"):
            break

        response = brain.think(session_id, user_text)
        print(f"NOVA: {response}")

    log.info("Milestone 4 demo stopped. Restart this script to test persistence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
