"""
nova.brain.brain
-------------------
NovaBrain is Stage 5 of the pipeline: it's what `AudioPipeline`'s
`on_utterance` callback will eventually call with the transcribed text.
It wires together everything else in `nova.brain` and `nova.memory`
behind one method: `think(session_id, user_text) -> response_text`.
"""

from __future__ import annotations

from typing import Optional

from nova.brain.llm_client import LLMClient
from nova.brain.planner import Planner
from nova.brain.router import ToolRouter
from nova.logging_setup import get_logger
from nova.memory.manager import MemoryManager

log = get_logger("brain")


class NovaBrain:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        router: Optional[ToolRouter] = None,
        planner: Optional[Planner] = None,
        memory: Optional[MemoryManager] = None,
    ):
        self.llm = llm or LLMClient()
        self.router = router or ToolRouter()
        self.planner = planner or Planner(llm=self.llm, router=self.router)
        self.memory = memory or MemoryManager()

    def think(self, session_id: str, user_text: str) -> str:
        # 1. Record what the user said (this may also trigger a long-term fact save).
        self.memory.record_turn(session_id, "user", user_text)

        # 2. Pull relevant context: known facts, RAG hits, recent conversation turns.
        context_messages = self.memory.build_context(session_id, user_text)

        # 3. Ask the Planner: answer directly, or call a tool?
        decision = self.planner.decide(user_text, context_messages=context_messages)

        if decision.action == "tool":
            tool_result = self.router.execute(decision.tool_name, decision.tool_args)
            follow_up = [
                {"role": "user", "content": user_text},
                {
                    "role": "assistant",
                    "content": f"(called tool '{decision.tool_name}', result: {tool_result})",
                },
                {
                    "role": "user",
                    "content": "Now answer the user's original request using that tool result, in plain natural language.",
                },
            ]
            response = self.llm.chat(context_messages + follow_up)
        else:
            response = decision.answer or "I'm not sure how to respond to that."

        # 4. Record NOVA's own reply, so it's part of the conversation history next turn.
        self.memory.record_turn(session_id, "assistant", response)
        return response
