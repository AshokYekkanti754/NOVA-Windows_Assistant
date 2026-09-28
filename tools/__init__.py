"""
NOVA - Tool Modules (Step 6)

    from tools import execute, execute_many, get_tool_schemas, schemas_as_prompt

Each module registers its tools on import. A missing optional dependency only disables
that one module instead of crashing NOVA.
"""
import importlib
import logging

from .base import (  # noqa: F401
    Tool,
    ToolResult,
    execute,
    execute_many,
    get_tool,
    get_tool_schemas,
    list_tools,
    schemas_as_prompt,
    tool,
)

log = logging.getLogger("nova.tools")

LOADED = {}

for _mod in ("windows_tools", "browser_tools", "knowledge_tools", "general_tools"):
    try:
        importlib.import_module(f".{_mod}", __name__)
        LOADED[_mod] = True
    except Exception as e:  # noqa: BLE001
        LOADED[_mod] = False
        log.warning("Tool module %s not loaded: %s", _mod, e)
