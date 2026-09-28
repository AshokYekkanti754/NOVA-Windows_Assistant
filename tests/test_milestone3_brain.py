"""
Milestone 3 tests: the Planner's JSON-decision parsing, the ToolRouter's
execution, and NovaBrain's full think() loop -- all exercised with a fake
LLM (duck-typed: anything with a `.chat(messages) -> str` method) so no
Ollama server or downloaded model is needed.

Run with: pytest tests/test_milestone3_brain.py -v
(or run this file's functions directly with plain python -- see bottom)
"""

from nova.brain.brain import NovaBrain
from nova.brain.planner import Planner
from nova.brain.router import Tool, ToolRouter
from nova.memory.context_store import ShortTermContext
from nova.memory.long_term_store import LongTermMemory
from nova.memory.manager import MemoryManager
from nova.memory.rag_store import RagKnowledgeStore


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeLLM:
    """Duck-typed stand-in for LLMClient: returns canned responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        if not self._responses:
            return '{"action": "answer", "answer": "(no more fake responses)"}'
        return self._responses.pop(0)


class FakeRagStore:
    """Stands in for RagKnowledgeStore -- returns nothing by default."""

    def query(self, text, k=3):
        return []


def make_memory_manager():
    """An in-memory MemoryManager (SQLite ':memory:', fake RAG) for fast, isolated tests."""
    return MemoryManager(
        short_term=ShortTermContext(db_path=":memory:"),
        long_term=LongTermMemory(db_path=":memory:"),
        rag=FakeRagStore(),
    )


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------

def test_planner_parses_direct_answer_json():
    llm = FakeLLM(['{"action": "answer", "answer": "Hello there!"}'])
    planner = Planner(llm=llm, router=ToolRouter())

    decision = planner.decide("hi")

    assert decision.action == "answer"
    assert decision.answer == "Hello there!"


def test_planner_parses_tool_call_json():
    llm = FakeLLM(['{"action": "tool", "tool_name": "get_current_time", "tool_args": {}}'])
    planner = Planner(llm=llm, router=ToolRouter())

    decision = planner.decide("what time is it?")

    assert decision.action == "tool"
    assert decision.tool_name == "get_current_time"
    assert decision.tool_args == {}


def test_planner_falls_back_to_raw_text_on_invalid_json():
    llm = FakeLLM(["this is not json at all"])
    planner = Planner(llm=llm, router=ToolRouter())

    decision = planner.decide("hi")

    assert decision.action == "answer"
    assert decision.answer == "this is not json at all"


def test_planner_system_prompt_lists_registered_tools():
    router = ToolRouter()
    router.register(Tool(name="get_current_time", description="Tells the time.", func=lambda: "noon"))
    llm = FakeLLM(['{"action": "answer", "answer": "ok"}'])
    planner = Planner(llm=llm, router=router)

    planner.decide("hi")

    system_message = llm.calls[0][0]
    assert system_message["role"] == "system"
    assert "get_current_time" in system_message["content"]
    assert "Tells the time." in system_message["content"]


# ---------------------------------------------------------------------------
# ToolRouter tests
# ---------------------------------------------------------------------------

def test_router_executes_registered_tool():
    router = ToolRouter()
    router.register(Tool(name="add", description="Adds two numbers.", func=lambda a, b: str(a + b)))

    result = router.execute("add", {"a": 2, "b": 3})
    assert result == "5"


def test_router_handles_unknown_tool_gracefully():
    router = ToolRouter()
    result = router.execute("does_not_exist", {})
    assert "no such tool" in result


def test_router_handles_tool_exception_gracefully():
    def broken_tool():
        raise ValueError("boom")

    router = ToolRouter()
    router.register(Tool(name="broken", description="Always fails.", func=broken_tool))

    result = router.execute("broken", {})
    assert "error running tool 'broken'" in result


# ---------------------------------------------------------------------------
# Full NovaBrain loop tests
# ---------------------------------------------------------------------------

def test_brain_direct_answer_path():
    llm = FakeLLM(['{"action": "answer", "answer": "I am NOVA."}'])
    brain = NovaBrain(llm=llm, router=ToolRouter(), planner=Planner(llm=llm, router=ToolRouter()), memory=make_memory_manager())

    response = brain.think("session-1", "who are you?")

    assert response == "I am NOVA."


def test_brain_tool_call_path_executes_tool_and_asks_llm_for_final_answer():
    router = ToolRouter()
    router.register(Tool(name="get_current_time", description="Tells the time.", func=lambda: "3:00 PM"))

    llm = FakeLLM(
        [
            '{"action": "tool", "tool_name": "get_current_time", "tool_args": {}}',
            "It's currently 3:00 PM.",
        ]
    )
    planner = Planner(llm=llm, router=router)
    brain = NovaBrain(llm=llm, router=router, planner=planner, memory=make_memory_manager())

    response = brain.think("session-1", "what time is it?")

    assert response == "It's currently 3:00 PM."
    assert len(llm.calls) == 2  # one planning call, one final-answer call


def test_brain_records_conversation_in_short_term_context():
    llm = FakeLLM(['{"action": "answer", "answer": "Sure thing."}'])
    memory = make_memory_manager()
    brain = NovaBrain(llm=llm, router=ToolRouter(), planner=Planner(llm=llm, router=ToolRouter()), memory=memory)

    brain.think("session-1", "hello")

    turns = memory.short_term.get_recent("session-1")
    assert turns == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Sure thing."},
    ]


if __name__ == "__main__":
    tests = [
        test_planner_parses_direct_answer_json,
        test_planner_parses_tool_call_json,
        test_planner_falls_back_to_raw_text_on_invalid_json,
        test_planner_system_prompt_lists_registered_tools,
        test_router_executes_registered_tool,
        test_router_handles_unknown_tool_gracefully,
        test_router_handles_tool_exception_gracefully,
        test_brain_direct_answer_path,
        test_brain_tool_call_path_executes_tool_and_asks_llm_for_final_answer,
        test_brain_records_conversation_in_short_term_context,
    ]
    for t in tests:
        t()
        print(f"{t.__name__}: PASS")
