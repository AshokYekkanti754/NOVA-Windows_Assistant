"""
NOVA - Tool registry (Step 6 core)

Every tool is a plain Python function decorated with @tool(...).
The Tool Router in your brain only needs:

    from tools import execute, get_tool_schemas
    result = execute("open_app", {"name": "notepad"})
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("nova.tools")


@dataclass
class ToolResult:
    ok: bool
    message: str                      # short, speakable summary
    data: Any = None                  # optional structured payload
    needs_confirmation: bool = False  # True when a dangerous tool was blocked

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]       # {"param": "description"}
    func: Callable[..., Any]
    category: str = "general"
    dangerous: bool = False


_REGISTRY: Dict[str, Tool] = {}


def tool(
    name: str,
    description: str,
    parameters: Optional[Dict[str, str]] = None,
    category: str = "general",
    dangerous: bool = False,
):
    """Decorator that registers a function as a NOVA tool."""

    def deco(fn: Callable[..., Any]):
        _REGISTRY[name] = Tool(name, description, parameters or {}, fn, category, dangerous)
        return fn

    return deco


def ok(message: str, data: Any = None) -> ToolResult:
    return ToolResult(True, message, data)


def fail(message: str, data: Any = None) -> ToolResult:
    return ToolResult(False, message, data)


def get_tool(name: str) -> Optional[Tool]:
    return _REGISTRY.get(name)


def list_tools(category: Optional[str] = None) -> List[Tool]:
    return [t for t in _REGISTRY.values() if category in (None, t.category)]


def get_tool_schemas(category: Optional[str] = None) -> List[dict]:
    """Schemas you can feed to the LLM / planner so it knows what exists."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
            "category": t.category,
            "dangerous": t.dangerous,
        }
        for t in list_tools(category)
    ]


def schemas_as_prompt(category: Optional[str] = None) -> str:
    """Compact text version of the tool list for a system prompt."""
    lines = []
    for t in list_tools(category):
        params = ", ".join(f"{k}: {v}" for k, v in t.parameters.items()) or "no params"
        flag = " [needs confirmation]" if t.dangerous else ""
        lines.append(f"- {t.name}({params}): {t.description}{flag}")
    return "\n".join(lines)


def execute(name: str, args: Optional[dict] = None, confirmed: bool = False) -> ToolResult:
    """
    Run a tool safely. Never raises - always returns a ToolResult.
    Dangerous tools (shutdown, delete, kill...) are blocked unless confirmed=True.
    """
    args = args or {}
    t = _REGISTRY.get(name)
    if t is None:
        return fail(f"I don't have a tool called {name}.")

    if t.dangerous and not confirmed:
        return ToolResult(
            False,
            f"This will run {name.replace('_', ' ')}. Do you want me to continue?",
            data={"tool": name, "args": args},
            needs_confirmation=True,
        )

    try:
        log.info("Running tool %s %s", name, args)
        out = t.func(**args)
        if isinstance(out, ToolResult):
            return out
        return ok(str(out) if out is not None else "Done.", out)
    except TypeError as e:
        return fail(f"Bad arguments for {name}: {e}")
    except Exception as e:  # noqa: BLE001
        log.exception("Tool %s failed", name)
        return fail(f"{name} failed: {e}")


def execute_many(calls: List[dict], confirmed: bool = False) -> List[ToolResult]:
    """calls = [{"tool": "open_app", "args": {"name": "notepad"}}, ...]"""
    results = []
    for c in calls:
        r = execute(c.get("tool", ""), c.get("args", {}), confirmed=confirmed)
        results.append(r)
        if not r.ok and not r.needs_confirmation:
            break  # stop the chain on a real failure
    return results
