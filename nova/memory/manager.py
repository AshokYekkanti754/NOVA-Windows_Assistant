"""
nova.memory.manager
----------------------
Coordinates the three memory stores so the rest of the Brain only ever
talks to one object. Decides *which* store to read/write for a given
turn, so Planner/NovaBrain don't need to know these three stores exist.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from nova.logging_setup import get_logger
from nova.memory.context_store import ShortTermContext
from nova.memory.long_term_store import LongTermMemory
from nova.memory.rag_store import RagKnowledgeStore

log = get_logger("memory.manager")

# Simple heuristic for "this sentence should become a long-term fact".
# Milestone 8+ could replace this with an LLM classification call; a plain
# keyword check is enough to prove the memory layer end to end for now.
_REMEMBER_TRIGGERS = ("remember that", "remember this", "don't forget", "note that")


class MemoryManager:
    def __init__(
        self,
        short_term: Optional[ShortTermContext] = None,
        long_term: Optional[LongTermMemory] = None,
        rag: Optional[RagKnowledgeStore] = None,
    ):
        self.short_term = short_term or ShortTermContext()
        self.long_term = long_term or LongTermMemory()
        self.rag = rag or RagKnowledgeStore()

    def record_turn(self, session_id: str, role: str, content: str) -> None:
        self.short_term.add_turn(session_id, role, content)
        if role == "user":
            self._maybe_store_fact(content)

    def _maybe_store_fact(self, content: str) -> None:
        lowered = content.lower()
        for trigger in _REMEMBER_TRIGGERS:
            idx = lowered.find(trigger)
            if idx != -1:
                fact = content[idx + len(trigger) :].strip(" .")
                if fact:
                    self.long_term.remember(fact)
                return

    def build_context(
        self,
        session_id: str,
        current_message: str,
        recent_turns: int = 6,
        rag_k: int = 3,
    ) -> List[Dict[str, str]]:
        """Assembles the extra context messages to feed the LLM before the user's latest message."""
        messages: List[Dict[str, str]] = []

        facts = self.long_term.recall_all()
        if facts:
            messages.append(
                {
                    "role": "system",
                    "content": "Known facts about the user:\n" + "\n".join(f"- {f}" for f in facts),
                }
            )

        rag_hits = self.rag.query(current_message, k=rag_k)
        if rag_hits:
            messages.append(
                {
                    "role": "system",
                    "content": "Relevant knowledge:\n" + "\n".join(f"- {h}" for h in rag_hits),
                }
            )

        messages.extend(self.short_term.get_recent(session_id, limit=recent_turns))
        return messages
