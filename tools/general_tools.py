"""
NOVA - General Tools (Step 6d): screenshots (see windows_tools), media, time, system status, timers.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime

import psutil
import pyautogui

from .base import tool, ok, fail, ToolResult

CATEGORY = "general"

MEDIA_KEYS = {
    "play": "playpause", "pause": "playpause", "play_pause": "playpause",
    "next": "nexttrack", "previous": "prevtrack", "prev": "prevtrack", "stop": "stop",
}


@tool("media_control", "Control media playback", {"action": "play, pause, next, previous, stop"}, CATEGORY)
def media_control(action: str) -> ToolResult:
    key = MEDIA_KEYS.get(action.lower().strip())
    if not key:
        return fail(f"I don't know the media action {action}.")
    pyautogui.press(key)
    return ok(f"Okay, {action}.")


@tool("get_datetime", "Current date and time", {}, CATEGORY)
def get_datetime() -> ToolResult:
    now = datetime.now()
    return ok(now.strftime("It's %I:%M %p on %A, %B %d, %Y.").replace(" 0", " "), now.isoformat())


@tool("system_status", "Battery, CPU, memory and disk usage", {}, CATEGORY)
def system_status() -> ToolResult:
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    disk = psutil.disk_usage("C:\\").percent
    bat = psutil.sensors_battery()
    parts = [f"CPU is at {cpu:.0f} percent", f"memory at {mem:.0f} percent", f"disk {disk:.0f} percent full"]
    if bat:
        parts.append(f"battery at {bat.percent:.0f} percent{' and charging' if bat.power_plugged else ''}")
    return ok(", ".join(parts) + ".", {"cpu": cpu, "memory": mem, "disk": disk, "battery": bat.percent if bat else None})


@tool("battery_status", "Battery level", {}, CATEGORY)
def battery_status() -> ToolResult:
    bat = psutil.sensors_battery()
    if not bat:
        return fail("I can't read a battery on this device.")
    state = "charging" if bat.power_plugged else "on battery"
    return ok(f"Battery is at {bat.percent:.0f} percent and {state}.", bat.percent)


# --- Timers -----------------------------------------------------------------
# A timer just sets a flag + callback. Wire `on_timer` to your TTS to speak the alert.
_timers = []
on_timer = None  # set from main: tools.general_tools.on_timer = lambda label: speak(f"{label} is done")


@tool("set_timer", "Set a countdown timer", {"seconds": "duration in seconds", "label": "optional name"}, CATEGORY)
def set_timer(seconds: int, label: str = "Your timer") -> ToolResult:
    seconds = int(seconds)

    def fire():
        if callable(on_timer):
            on_timer(label)

    t = threading.Timer(seconds, fire)
    t.daemon = True
    t.start()
    _timers.append(t)
    m, s = divmod(seconds, 60)
    human = f"{m} minute{'s' if m != 1 else ''}" + (f" {s} seconds" if s else "") if m else f"{s} seconds"
    return ok(f"Timer set for {human}.")


@tool("cancel_timers", "Cancel all running timers", {}, CATEGORY)
def cancel_timers() -> ToolResult:
    for t in _timers:
        t.cancel()
    n = len(_timers)
    _timers.clear()
    return ok(f"Cancelled {n} timer(s).")
