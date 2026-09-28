"""
NOVA - Step 7: Final Answer
Turns tool results into a natural spoken reply and prepares it for Piper TTS.

    from final_answer import FinalAnswer
    fa = FinalAnswer(model="llama3.1")

    # one-shot
    text = fa.compose("open notepad and set volume to 40", results)

    # streaming, sentence by sentence (lowest latency for TTS)
    for sentence in fa.stream("what's on my clipboard", results):
        tts.speak(sentence)
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Iterator, List, Optional, Sequence, Union

try:
    import ollama
except ImportError:  # still usable in fallback mode
    ollama = None

SYSTEM_PROMPT = """You are Nova, a friendly voice assistant on the user's Windows laptop.
Your reply will be SPOKEN aloud, so:
- Use 1 to 3 short, natural sentences.
- No markdown, bullets, emojis, code, or URLs.
- Never read file paths or IDs aloud; say just the file or folder name.
- Say numbers naturally (e.g. "forty percent").
- Base your answer ONLY on the tool results provided. Never claim something succeeded if it failed.
- If a tool failed, briefly say what went wrong and suggest one fix.
- If confirmation is needed, ask a single yes/no question.
- Sound like a helpful person, not a log file."""


# ----------------------------------------------------------------------------
# Text cleanup for TTS
# ----------------------------------------------------------------------------
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\u200d\ufe0f]+", flags=re.UNICODE
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _friendly_path(m: re.Match) -> str:
    return m.group(0).replace("\\", "/").rstrip("/").split("/")[-1] or "that folder"


def prepare_for_tts(text: str, max_chars: int = 500) -> str:
    """Make text safe and pleasant for a speech engine."""
    t = text or ""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                      # code blocks
    t = re.sub(r"`([^`]*)`", r"\1", t)                                # inline code
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)                       # images
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)                    # [text](url) -> text
    t = re.sub(r"https?://\S+", "a link", t)                          # bare URLs
    t = re.sub(r"[A-Za-z]:\\[^\s,;]+", _friendly_path, t)             # C:\path\file.txt -> file.txt
    t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)                       # headings
    t = re.sub(r"(?m)^\s*[-*•]\s+", "", t)                            # bullets
    t = re.sub(r"(?m)^\s*\d+[.)]\s+", "", t)                          # numbered lists
    t = re.sub(r"[*_~>|]+", "", t)                                    # emphasis chars
    t = _EMOJI.sub("", t)
    t = (t.replace("%", " percent").replace("&", " and ").replace("°C", " degrees Celsius")
          .replace("°F", " degrees Fahrenheit").replace("→", " to ").replace("…", "."))
    t = re.sub(r"\s*\n+\s*", ". ", t)                                 # newlines -> pauses
    t = re.sub(r"\.\s*\.", ".", t)
    t = re.sub(r"\s{2,}", " ", t).strip()

    if len(t) > max_chars:                                            # cut at a sentence boundary
        cut = t[:max_chars]
        last = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        t = (cut[: last + 1] if last > 80 else cut.rsplit(" ", 1)[0] + ".")
    return t


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


# ----------------------------------------------------------------------------
# Tool-result helpers
# ----------------------------------------------------------------------------
def _as_dict(r) -> dict:
    if isinstance(r, dict):
        return r
    if hasattr(r, "to_dict"):
        return r.to_dict()
    return {"ok": True, "message": str(r)}


def _compact_results(results: Sequence) -> str:
    """Trim results so the LLM prompt stays small and fast."""
    out = []
    for r in map(_as_dict, results):
        item = {"ok": r.get("ok"), "message": r.get("message")}
        if r.get("needs_confirmation"):
            item["needs_confirmation"] = True
        data = r.get("data")
        if isinstance(data, dict) and "context" in data:               # RAG payload
            item["context"] = str(data["context"])[:1500]
        elif isinstance(data, (list, dict)):
            item["data"] = json.dumps(data, default=str)[:600]
        elif isinstance(data, str) and len(data) < 400:
            item["data"] = data
        out.append(item)
    return json.dumps(out, ensure_ascii=False)


def _simple_reply(results: Sequence) -> Optional[str]:
    """Fast path / offline fallback: no LLM needed for plain success messages."""
    rs = [_as_dict(r) for r in results]
    if not rs:
        return None
    for r in rs:
        if r.get("needs_confirmation"):
            return r["message"]
    failed = [r for r in rs if not r.get("ok")]
    if failed:
        return failed[0]["message"]
    return " ".join(r["message"] for r in rs if r.get("message"))


# ----------------------------------------------------------------------------
class FinalAnswer:
    def __init__(
        self,
        model: str = "llama3.1",
        temperature: float = 0.4,
        fast_path: bool = True,
        max_chars: int = 500,
        host: Optional[str] = None,
    ):
        """
        fast_path=True  -> skip the LLM when every tool just succeeded with a short message
                           ("Opening notepad."). Saves ~1s of latency per command.
        """
        self.model = model
        self.temperature = temperature
        self.fast_path = fast_path
        self.max_chars = max_chars
        self.client = ollama.Client(host=host) if (ollama and host) else None

    # ---- internal ---------------------------------------------------------
    def _needs_llm(self, results: Sequence) -> bool:
        if not self.fast_path:
            return True
        for r in map(_as_dict, results):
            data = r.get("data")
            if isinstance(data, dict) and "context" in data:           # RAG answers need synthesis
                return True
            if len(r.get("message", "")) > 220:
                return True
        return False

    def _messages(self, user_text: str, results: Sequence, history: Optional[List[dict]]) -> List[dict]:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            msgs += history[-6:]                                        # short-term context
        msgs.append({
            "role": "user",
            "content": (
                f'User said: "{user_text}"\n'
                f"Tool results (JSON): {_compact_results(results)}\n\n"
                "Write Nova's spoken reply."
            ),
        })
        return msgs

    def _chat(self, **kw):
        return (self.client or ollama).chat(**kw)

    # ---- public API -------------------------------------------------------
    def compose(
        self,
        user_text: str,
        results: Union[Sequence, None] = None,
        history: Optional[List[dict]] = None,
        direct_answer: Optional[str] = None,
    ) -> str:
        """
        Returns TTS-ready text.
        - direct_answer: brain already produced an answer (chit-chat, no tools) -> just clean it.
        - results: list of ToolResult / dicts from the Tool Router.
        """
        if direct_answer:
            return prepare_for_tts(direct_answer, self.max_chars)

        results = results or []
        if not results:
            return "Sorry, I didn't catch anything to do there."

        if not self._needs_llm(results):
            return prepare_for_tts(_simple_reply(results) or "Done.", self.max_chars)

        if ollama is None:
            return prepare_for_tts(_simple_reply(results) or "Done.", self.max_chars)
        try:
            resp = self._chat(
                model=self.model,
                messages=self._messages(user_text, results, history),
                options={"temperature": self.temperature, "num_predict": 160},
            )
            return prepare_for_tts(resp["message"]["content"], self.max_chars)
        except Exception:                                               # Ollama down -> still answer
            return prepare_for_tts(_simple_reply(results) or "Done.", self.max_chars)

    def stream(
        self,
        user_text: str,
        results: Union[Sequence, None] = None,
        history: Optional[List[dict]] = None,
        direct_answer: Optional[str] = None,
    ) -> Iterator[str]:
        """Yield TTS-ready sentences as soon as the LLM finishes each one."""
        results = results or []
        if direct_answer or not results or ollama is None or not self._needs_llm(results):
            for s in split_sentences(self.compose(user_text, results, history, direct_answer)):
                yield s
            return

        buf = ""
        try:
            for chunk in self._chat(
                model=self.model,
                messages=self._messages(user_text, results, history),
                options={"temperature": self.temperature, "num_predict": 160},
                stream=True,
            ):
                buf += chunk["message"]["content"]
                parts = _SENTENCE_END.split(buf)
                for sent in parts[:-1]:                                  # complete sentences only
                    clean = prepare_for_tts(sent, self.max_chars)
                    if clean:
                        yield clean
                buf = parts[-1]
            tail = prepare_for_tts(buf, self.max_chars)
            if tail:
                yield tail
        except Exception:
            yield prepare_for_tts(_simple_reply(results) or "Done.", self.max_chars)
