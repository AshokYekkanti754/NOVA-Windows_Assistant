"""
NOVA - demo for Steps 6 and 7.

Run:  python demo_steps_6_7.py
Then type commands. Two ways to test:
  1) Plain typed test of the tools (no LLM):   >> open_app notepad
  2) Full flow with your Brain's planner:      see `handle_utterance()` below.
"""
import json
import logging

from tools import LOADED, execute, execute_many, schemas_as_prompt
from final_answer import FinalAnswer

logging.basicConfig(level=logging.INFO)
fa = FinalAnswer(model="llama3.1")

# Optional: speak timer alerts through your Piper TTS
# from tools import general_tools
# general_tools.on_timer = lambda label: tts.speak(f"{label} is done.")


# ---------------------------------------------------------------------------
# HOW IT PLUGS INTO THE BRAIN (Step 5)
#
#   Planner  -> decides tool calls like:
#       [{"tool": "open_app", "args": {"name": "notepad"}},
#        {"tool": "set_volume", "args": {"level": 40}}]
#   Tool Router -> execute_many(calls)          (Step 6)
#   Final Answer -> fa.stream(...) -> Piper     (Step 7)
#
# Give the Planner the tool list with:  schemas_as_prompt()
# ---------------------------------------------------------------------------
_pending = {"calls": None}   # remembers a blocked dangerous action until user says yes


def handle_utterance(user_text: str, calls: list, history=None, direct_answer=None):
    """calls = tool calls produced by your Planner for this utterance."""
    # confirmation flow for shutdown/delete/etc.
    if _pending["calls"] and user_text.lower().strip() in ("yes", "yeah", "confirm", "do it", "go ahead"):
        results = execute_many(_pending["calls"], confirmed=True)
        _pending["calls"] = None
    elif _pending["calls"] and user_text.lower().strip() in ("no", "cancel", "stop", "never mind"):
        _pending["calls"] = None
        return iter(["Okay, cancelled."])
    else:
        results = execute_many(calls) if calls else []
        if any(r.needs_confirmation for r in results):
            _pending["calls"] = calls

    # Step 7: natural answer, sentence by sentence
    return fa.stream(user_text, results, history=history, direct_answer=direct_answer)


if __name__ == "__main__":
    print("Loaded modules:", LOADED)
    print("\nAvailable tools:\n" + schemas_as_prompt())
    print("\nType:  tool_name {json args}   e.g.  set_volume {\"level\": 30}   (blank to quit)\n")

    while True:
        line = input(">> ").strip()
        if not line:
            break
        name, _, raw = line.partition(" ")
        args = json.loads(raw) if raw.strip() else {}
        for sentence in handle_utterance(f"{name} {args}", [{"tool": name, "args": args}]):
            print("NOVA:", sentence)         # replace with tts.speak(sentence)
