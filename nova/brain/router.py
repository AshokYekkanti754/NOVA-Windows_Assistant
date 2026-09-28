"""
nova.brain.router
-------------------
A tool registry + executor. This is intentionally simple -- a dict of
name -> Tool -- rather than pulling in a full agent framework, so it's
easy to see exactly how a tool call turns into a Python function call.

Milestone 5/6 will register real tools here (Windows automation, browser
automation); Milestone 3 just needs one mock tool to prove the loop works.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

from nova.logging_setup import get_logger

log = get_logger("brain.router")


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., str]  # must return a plain string result


class ToolRouter:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        log.info("Registered tool '%s'", tool.name)

    def execute(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            log.warning("Unknown tool requested: %s", tool_name)
            return f"(error: no such tool '{tool_name}')"

        try:
            result = tool.func(**tool_args)
            log.info("Executed tool '%s' -> %r", tool_name, result)
            return result
        except Exception as exc:  # noqa: BLE001 -- a bad tool call must never crash NOVA
            log.exception("Tool '%s' raised an exception", tool_name)
            return f"(error running tool '{tool_name}': {exc})"
