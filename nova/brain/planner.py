"""
nova.brain.planner
--------------------
The Planner is a "Custom" agent framework (per config.yaml's
brain.llm.provider and the diagram's "LangChain or Custom" note) --
a single LLM call with a strict JSON output contract, rather than a
full LangChain agent. This keeps the control flow fully visible: one
prompt in, one parsed decision out, no hidden framework machinery.

The LLM is asked to reply with ONLY a JSON object shaped like:
    {"action": "answer", "answer": "<direct reply>"}
or
    {"action": "tool", "tool_name": "<name>", "tool_args": {...}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from nova.brain.llm_client import LLMClient
from nova.brain.router import ToolRouter
from nova.logging_setup import get_logger

log = get_logger("brain.planner")

_SYSTEM_PROMPT_TEMPLATE = """You are NOVA's planning module. Given the user's request \
and the conversation context, decide whether to answer directly or call a tool.

Respond with ONLY a single JSON object, no other text, in one of these two shapes:
{{"action": "answer", "answer": "<your direct reply to the user>"}}
{{"action": "tool", "tool_name": "<tool name>", "tool_args": {{"<arg>": "<value>"}}}}

Available tools:
{tool_descriptions}

If no tool is relevant, use the "answer" shape."""


@dataclass
class PlanDecision:
    action: str  # "answer" | "tool"
    answer: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)


class Planner:
    def __init__(self, llm: Optional[LLMClient] = None, router: Optional[ToolRouter] = None):
        self.llm = llm or LLMClient()
        self.router = router or ToolRouter()

    def _build_system_prompt(self) -> str:
        if self.router.tools:
            descriptions = "\n".join(
                f"- {name}: {tool.description}" for name, tool in self.router.tools.items()
            )
        else:
            descriptions = "(no tools registered)"
        return _SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=descriptions)

    def decide(
        self, user_text: str, context_messages: Optional[List[Dict[str, str]]] = None
    ) -> PlanDecision:
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        messages.extend(context_messages or [])
        messages.append({"role": "user", "content": user_text})

        raw = self.llm.chat(messages)
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> PlanDecision:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Planner LLM did not return valid JSON -- treating as a direct answer: %r", raw)
            return PlanDecision(action="answer", answer=raw.strip())

        action = data.get("action", "answer")
        if action == "tool" and data.get("tool_name"):
            return PlanDecision(
                action="tool",
                tool_name=data["tool_name"],
                tool_args=data.get("tool_args") or {},
            )
        return PlanDecision(action="answer", answer=data.get("answer", ""))
