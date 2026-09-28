"""
nova.brain.llm_client
-----------------------
Thin wrapper around the local Ollama server so the rest of the Brain
depends on one method -- `chat(messages) -> text` -- instead of the
`ollama` package's client shape directly.

The real client is loaded lazily and can be dependency-injected for
tests via the `client` constructor argument (anything with a matching
`.chat(model=..., messages=..., options=...)` method works).
"""

from __future__ import annotations

from typing import Dict, List, Optional

from nova.config.loader import get_config
from nova.logging_setup import get_logger

log = get_logger("brain.llm")


class LLMClient:
    def __init__(self, config: Optional[dict] = None, client=None):
        cfg = config or get_config()
        llm_cfg = cfg["brain"]["llm"]

        self.model: str = llm_cfg["model"]
        self.base_url: str = llm_cfg["base_url"]
        self.temperature: float = llm_cfg["temperature"]
        self._client = client  # injected fake in tests; real ollama.Client otherwise

    def _ensure_client(self):
        if self._client is None:
            import ollama  # heavy import, deferred; needs a running Ollama server

            self._client = ollama.Client(host=self.base_url)
            log.info("Connected to Ollama at %s (model=%s)", self.base_url, self.model)
        return self._client

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        Returns the assistant's reply text.
        """
        client = self._ensure_client()
        response = client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": self.temperature},
        )
        text = response["message"]["content"]
        log.info("LLM responded (%d chars)", len(text))
        return text
