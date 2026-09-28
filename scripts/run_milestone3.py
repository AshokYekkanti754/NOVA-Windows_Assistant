"""
Milestone 3 deliverable: a text-in/text-out chat loop where the LLM can
choose to call a mock tool ("get_current_time") instead of answering
directly. This proves the Planner -> Tool Router loop works before any
audio is involved.

Requires a local Ollama server running with the configured model pulled:
    ollama pull llama3.1
    ollama serve   (usually already running as a background service)

Run:
    uv sync --extra brain
    uv run python scripts/run_milestone3.py
"""

import datetime
import sys

from nova.brain.brain import NovaBrain
from nova.brain.router import Tool, ToolRouter
from nova.config.loader import get_config
from nova.logging_setup import get_logger, setup_logging


def get_current_time() -> str:
    """The mock tool: returns the current local time as a string."""
    return datetime.datetime.now().strftime("%I:%M %p")


def main() -> int:
    get_config()
    setup_logging()
    log = get_logger("milestone3")

    router = ToolRouter()
    router.register(
        Tool(
            name="get_current_time",
            description="Returns the current local time. No arguments needed.",
            func=get_current_time,
        )
    )

    brain = NovaBrain(router=router)
    session_id = "milestone3-demo"

    print("NOVA Brain -- text chat (type 'quit' to exit)")
    print("Try asking: 'what time is it right now?'")

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

    log.info("Milestone 3 demo stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
