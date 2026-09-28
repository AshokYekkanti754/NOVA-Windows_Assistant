"""
NOVA - Browser Tools (Step 6b)
Stack: Playwright

Playwright's sync API breaks if called from an asyncio thread, and it must always be
used from the SAME thread. So every browser call runs on one dedicated worker thread.
"""
from __future__ import annotations

import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .base import tool, ok, fail, ToolResult

CATEGORY = "browser"

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nova-browser")
_state = {"pw": None, "context": None, "page": None}


def _run(fn, *args, **kwargs):
    return _executor.submit(fn, *args, **kwargs).result(timeout=60)


def _ensure_page():
    """(runs on browser thread) Start Chromium/Edge once, reuse afterwards."""
    from playwright.sync_api import sync_playwright

    ctx = _state["context"]
    if ctx is not None:
        try:
            if _state["page"] is None or _state["page"].is_closed():
                _state["page"] = ctx.new_page()
            return _state["page"]
        except Exception:
            _state.update(pw=None, context=None, page=None)  # browser was closed by user

    pw = sync_playwright().start()
    profile = str(__import__("pathlib").Path.home() / ".nova" / "browser_profile")
    kwargs = dict(user_data_dir=profile, headless=False, viewport=None, args=["--start-maximized"])
    try:
        ctx = pw.chromium.launch_persistent_context(channel="msedge", **kwargs)  # uses installed Edge
    except Exception:
        ctx = pw.chromium.launch_persistent_context(**kwargs)                    # bundled Chromium
    _state.update(pw=pw, context=ctx, page=ctx.pages[0] if ctx.pages else ctx.new_page())
    return _state["page"]


def _normalize_url(url: str) -> str:
    url = url.strip()
    if re.match(r"^https?://", url):
        return url
    return "https://" + url


# ----------------------------------------------------------------------------
@tool("open_url", "Open a website", {"url": "e.g. youtube.com"}, CATEGORY)
def open_url(url: str) -> ToolResult:
    def job():
        page = _ensure_page()
        page.goto(_normalize_url(url), wait_until="domcontentloaded")
        return page.title()
    title = _run(job)
    return ok(f"Opened {title or url}.", title)


@tool("search_web", "Search the web", {"query": "search text", "engine": "google or duckduckgo (default google)"}, CATEGORY)
def search_web(query: str, engine: str = "google") -> ToolResult:
    q = urllib.parse.quote_plus(query)
    url = f"https://duckduckgo.com/?q={q}" if engine.lower().startswith("duck") else f"https://www.google.com/search?q={q}"

    def job():
        page = _ensure_page()
        page.goto(url, wait_until="domcontentloaded")
        return page.title()
    _run(job)
    return ok(f"Searching for {query}.")


@tool("browser_back", "Go back a page", {}, CATEGORY)
def browser_back() -> ToolResult:
    _run(lambda: _ensure_page().go_back())
    return ok("Went back.")


@tool("browser_forward", "Go forward a page", {}, CATEGORY)
def browser_forward() -> ToolResult:
    _run(lambda: _ensure_page().go_forward())
    return ok("Went forward.")


@tool("browser_reload", "Reload the page", {}, CATEGORY)
def browser_reload() -> ToolResult:
    _run(lambda: _ensure_page().reload())
    return ok("Reloaded.")


@tool("browser_scroll", "Scroll the web page", {"direction": "down/up/top/bottom"}, CATEGORY)
def browser_scroll(direction: str = "down") -> ToolResult:
    js = {
        "down": "window.scrollBy(0, window.innerHeight * 0.8)",
        "up": "window.scrollBy(0, -window.innerHeight * 0.8)",
        "top": "window.scrollTo(0, 0)",
        "bottom": "window.scrollTo(0, document.body.scrollHeight)",
    }.get(direction.lower(), "window.scrollBy(0, 600)")
    _run(lambda: _ensure_page().evaluate(js))
    return ok(f"Scrolled {direction}.")


@tool("new_tab", "Open a new tab", {"url": "optional URL"}, CATEGORY)
def new_tab(url: str = "") -> ToolResult:
    def job():
        _ensure_page()
        _state["page"] = _state["context"].new_page()
        if url:
            _state["page"].goto(_normalize_url(url), wait_until="domcontentloaded")
    _run(job)
    return ok("Opened a new tab.")


@tool("click_element", "Click a link/button by its visible text or a CSS selector", {"target": "visible text or CSS selector"}, CATEGORY)
def click_element(target: str) -> ToolResult:
    def job():
        page = _ensure_page()
        for locator in (
            page.get_by_role("button", name=target),
            page.get_by_role("link", name=target),
            page.get_by_text(target, exact=False),
        ):
            if locator.count() > 0:
                locator.first.click(timeout=5000)
                return True
        try:
            page.click(target, timeout=3000)
            return True
        except Exception:
            return False
    return ok(f"Clicked {target}.") if _run(job) else fail(f"I couldn't find {target} on the page.")


@tool("fill_field", "Type into a form field", {"field": "label, placeholder or CSS selector", "value": "text to enter", "submit": "press Enter after (true/false)"}, CATEGORY)
def fill_field(field: str, value: str, submit: bool = False) -> ToolResult:
    def job():
        page = _ensure_page()
        for locator in (page.get_by_label(field), page.get_by_placeholder(field)):
            if locator.count() > 0:
                locator.first.fill(value)
                if submit:
                    locator.first.press("Enter")
                return True
        try:
            page.fill(field, value, timeout=3000)
            if submit:
                page.press(field, "Enter")
            return True
        except Exception:
            return False
    return ok(f"Filled {field}.") if _run(job) else fail(f"I couldn't find the field {field}.")


@tool("extract_text", "Read the text of the current page", {"max_chars": "default 1500"}, CATEGORY)
def extract_text(max_chars: int = 1500) -> ToolResult:
    def job():
        page = _ensure_page()
        return page.title(), page.inner_text("body")
    title, body = _run(job)
    body = re.sub(r"\n{2,}", "\n", body).strip()[: int(max_chars)]
    return ok(f"Page: {title}. {body[:300]}", {"title": title, "text": body})


@tool("extract_links", "List the main links on the current page", {"limit": "default 10"}, CATEGORY)
def extract_links(limit: int = 10) -> ToolResult:
    def job():
        page = _ensure_page()
        return page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({text: e.innerText.trim(), href: e.href})).filter(x => x.text)",
        )
    links = _run(job)[: int(limit)]
    return ok(f"I found {len(links)} links.", links)


@tool("read_search_results", "Read top results from the current Google/DuckDuckGo results page", {"limit": "default 5"}, CATEGORY)
def read_search_results(limit: int = 5) -> ToolResult:
    def job():
        page = _ensure_page()
        return page.eval_on_selector_all(
            "h3",
            "els => els.map(e => ({title: e.innerText, href: (e.closest('a')||{}).href || ''}))",
        )
    res = [r for r in _run(job) if r["title"]][: int(limit)]
    if not res:
        return fail("I don't see any search results on this page.")
    return ok("Top results: " + "; ".join(r["title"] for r in res[:3]), res)


@tool("browser_screenshot", "Screenshot the current page", {}, CATEGORY)
def browser_screenshot() -> ToolResult:
    from pathlib import Path
    import time
    folder = Path.home() / "Pictures" / "Nova Screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / time.strftime("page_%Y%m%d_%H%M%S.png")
    _run(lambda: _ensure_page().screenshot(path=str(path)))
    return ok("Saved a screenshot of the page.", str(path))


@tool("close_browser", "Close the automated browser", {}, CATEGORY)
def close_browser() -> ToolResult:
    def job():
        if _state["context"]:
            _state["context"].close()
        if _state["pw"]:
            _state["pw"].stop()
        _state.update(pw=None, context=None, page=None)
    _run(job)
    return ok("Closed the browser.")
