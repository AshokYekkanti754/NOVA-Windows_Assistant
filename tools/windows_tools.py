"""
NOVA - Windows Tools (Step 6a)
Stack: pyautogui + pywinauto (+ psutil, pycaw, screen-brightness-control, pyperclip)
"""
from __future__ import annotations

import ctypes
import difflib
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List

import psutil
import pyautogui
import pyperclip

from .base import tool, ok, fail, ToolResult

pyautogui.FAILSAFE = True   # slam mouse into a corner to abort automation
pyautogui.PAUSE = 0.05

CATEGORY = "windows"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
KNOWN_APPS = {
    "notepad": ("notepad.exe", "notepad.exe"),
    "calculator": ("calc.exe", "CalculatorApp.exe"),
    "calc": ("calc.exe", "CalculatorApp.exe"),
    "paint": ("mspaint.exe", "mspaint.exe"),
    "cmd": ("cmd.exe", "cmd.exe"),
    "command prompt": ("cmd.exe", "cmd.exe"),
    "terminal": ("wt.exe", "WindowsTerminal.exe"),
    "powershell": ("powershell.exe", "powershell.exe"),
    "file explorer": ("explorer.exe", "explorer.exe"),
    "explorer": ("explorer.exe", "explorer.exe"),
    "task manager": ("taskmgr.exe", "Taskmgr.exe"),
    "chrome": ("chrome.exe", "chrome.exe"),
    "edge": ("msedge.exe", "msedge.exe"),
    "firefox": ("firefox.exe", "firefox.exe"),
    "vs code": ("code", "Code.exe"),
    "vscode": ("code", "Code.exe"),
    "visual studio code": ("code", "Code.exe"),
    "word": ("winword.exe", "WINWORD.EXE"),
    "excel": ("excel.exe", "EXCEL.EXE"),
    "powerpoint": ("powerpnt.exe", "POWERPNT.EXE"),
    "spotify": ("spotify.exe", "Spotify.exe"),
    "snipping tool": ("snippingtool.exe", "SnippingTool.exe"),
}

FOLDER_ALIASES = {
    "desktop": Path.home() / "Desktop",
    "documents": Path.home() / "Documents",
    "downloads": Path.home() / "Downloads",
    "pictures": Path.home() / "Pictures",
    "music": Path.home() / "Music",
    "videos": Path.home() / "Videos",
    "home": Path.home(),
}

SETTINGS_PAGES = {
    "wifi": "ms-settings:network-wifi",
    "network": "ms-settings:network",
    "bluetooth": "ms-settings:bluetooth",
    "display": "ms-settings:display",
    "sound": "ms-settings:sound",
    "battery": "ms-settings:batterysaver",
    "power": "ms-settings:powersleep",
    "update": "ms-settings:windowsupdate",
    "apps": "ms-settings:appsfeatures",
    "privacy": "ms-settings:privacy",
    "personalization": "ms-settings:personalization",
    "storage": "ms-settings:storagesense",
    "notifications": "ms-settings:notifications",
    "default apps": "ms-settings:defaultapps",
    "settings": "ms-settings:",
}


def resolve_path(p: str) -> Path:
    p = (p or "").strip().strip('"')
    if p.lower() in FOLDER_ALIASES:
        return FOLDER_ALIASES[p.lower()]
    return Path(os.path.expandvars(os.path.expanduser(p)))


def _start_apps() -> List[dict]:
    """List Start-menu apps (name + AppID) via PowerShell."""
    cmd = ["powershell", "-NoProfile", "-Command",
           "Get-StartApps | ForEach-Object { $_.Name + '|' + $_.AppID }"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    apps = []
    for line in out.splitlines():
        if "|" in line:
            name, appid = line.split("|", 1)
            apps.append({"name": name.strip(), "id": appid.strip()})
    return apps


# ----------------------------------------------------------------------------
# Apps
# ----------------------------------------------------------------------------
@tool("open_app", "Open an application by name", {"name": "app name, e.g. notepad, chrome, spotify"}, CATEGORY)
def open_app(name: str) -> ToolResult:
    key = name.lower().strip()
    if key in KNOWN_APPS:
        try:
            subprocess.Popen(["cmd", "/c", "start", "", KNOWN_APPS[key][0]], shell=False)
            return ok(f"Opening {name}.")
        except Exception:
            pass  # fall through to Start-menu search

    apps = _start_apps()
    names = [a["name"].lower() for a in apps]
    match = difflib.get_close_matches(key, names, n=1, cutoff=0.5)
    if not match:
        sub = [a for a in apps if key in a["name"].lower()]
        if not sub:
            return fail(f"I couldn't find an app called {name}.")
        app = sub[0]
    else:
        app = apps[names.index(match[0])]
    subprocess.Popen(["explorer", f"shell:AppsFolder\\{app['id']}"])
    return ok(f"Opening {app['name']}.")


@tool("close_app", "Close a running application", {"name": "app or process name"}, CATEGORY)
def close_app(name: str) -> ToolResult:
    key = name.lower().strip()
    target = KNOWN_APPS.get(key, (None, key))[1].lower().replace(".exe", "")
    closed = 0
    for p in psutil.process_iter(["name"]):
        pname = (p.info["name"] or "").lower().replace(".exe", "")
        if target in pname:
            try:
                p.terminate()
                closed += 1
            except Exception:
                pass
    return ok(f"Closed {name}.") if closed else fail(f"{name} doesn't seem to be running.")


@tool("search_apps", "Search installed apps in the Start menu", {"query": "part of an app name"}, CATEGORY)
def search_apps(query: str) -> ToolResult:
    hits = [a["name"] for a in _start_apps() if query.lower() in a["name"].lower()][:8]
    if not hits:
        return fail(f"No apps matching {query}.")
    return ok("I found: " + ", ".join(hits), hits)


@tool("list_windows", "List currently open windows", {}, CATEGORY)
def list_windows() -> ToolResult:
    from pywinauto import Desktop
    titles = [w.window_text() for w in Desktop(backend="uia").windows() if w.window_text().strip()]
    return ok(f"You have {len(titles)} windows open.", titles)


@tool("focus_window", "Bring a window to the front", {"title": "part of the window title"}, CATEGORY)
def focus_window(title: str) -> ToolResult:
    from pywinauto import Desktop
    for w in Desktop(backend="uia").windows():
        if title.lower() in w.window_text().lower():
            try:
                w.restore()
            except Exception:
                pass
            w.set_focus()
            return ok(f"Switched to {w.window_text()}.")
    return fail(f"I couldn't find a window with {title}.")


# ----------------------------------------------------------------------------
# Files & folders
# ----------------------------------------------------------------------------
@tool("open_path", "Open a file or folder in its default app", {"path": "file/folder path or alias like downloads"}, CATEGORY)
def open_path(path: str) -> ToolResult:
    p = resolve_path(path)
    if not p.exists():
        return fail(f"{p} does not exist.")
    os.startfile(str(p))
    return ok(f"Opened {p.name or p}.")


@tool("list_folder", "List files in a folder", {"path": "folder path or alias", "limit": "max items (default 15)"}, CATEGORY)
def list_folder(path: str = "desktop", limit: int = 15) -> ToolResult:
    p = resolve_path(path)
    if not p.is_dir():
        return fail(f"{p} is not a folder.")
    items = sorted(x.name for x in p.iterdir())
    shown = items[: int(limit)]
    return ok(f"{p.name} has {len(items)} items. " + ", ".join(shown[:5]), shown)


@tool("find_files", "Find files by name", {"query": "part of file name", "root": "folder to search (default home)"}, CATEGORY)
def find_files(query: str, root: str = "home") -> ToolResult:
    base = resolve_path(root)
    hits = []
    for dirpath, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "AppData", "__pycache__")]
        for f in files:
            if query.lower() in f.lower():
                hits.append(str(Path(dirpath) / f))
                if len(hits) >= 10:
                    return ok(f"Found {len(hits)}+ files.", hits)
    return ok(f"Found {len(hits)} files.", hits) if hits else fail(f"No files matching {query}.")


@tool("create_folder", "Create a folder", {"path": "new folder path"}, CATEGORY)
def create_folder(path: str) -> ToolResult:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return ok(f"Created folder {p.name}.", str(p))


@tool("create_file", "Create a text file", {"path": "file path", "content": "text content"}, CATEGORY)
def create_file(path: str, content: str = "") -> ToolResult:
    p = resolve_path(path)
    if p.exists():
        return fail(f"{p.name} already exists.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return ok(f"Created {p.name}.", str(p))


@tool("copy_path", "Copy a file or folder", {"src": "source", "dst": "destination"}, CATEGORY)
def copy_path(src: str, dst: str) -> ToolResult:
    s, d = resolve_path(src), resolve_path(dst)
    if s.is_dir():
        shutil.copytree(s, d / s.name if d.is_dir() else d)
    else:
        shutil.copy2(s, d)
    return ok(f"Copied {s.name}.")


@tool("move_path", "Move or rename a file or folder", {"src": "source", "dst": "destination"}, CATEGORY)
def move_path(src: str, dst: str) -> ToolResult:
    s, d = resolve_path(src), resolve_path(dst)
    shutil.move(str(s), str(d))
    return ok(f"Moved {s.name}.")


@tool("delete_path", "Send a file or folder to the Recycle Bin", {"path": "path to delete"}, CATEGORY, dangerous=True)
def delete_path(path: str) -> ToolResult:
    p = resolve_path(path)
    if not p.exists():
        return fail(f"{p} does not exist.")
    try:
        from send2trash import send2trash
        send2trash(str(p))
        return ok(f"Moved {p.name} to the Recycle Bin.")
    except ImportError:
        return fail("Install send2trash (pip install send2trash) to enable safe deleting.")


# ----------------------------------------------------------------------------
# System controls
# ----------------------------------------------------------------------------
def _volume_iface():
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    dev = AudioUtilities.GetSpeakers()
    if hasattr(dev, "EndpointVolume"):          # newer pycaw
        return dev.EndpointVolume
    iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(iface, POINTER(IAudioEndpointVolume))


@tool("set_volume", "Set system volume", {"level": "0-100"}, CATEGORY)
def set_volume(level: int) -> ToolResult:
    level = max(0, min(100, int(level)))
    _volume_iface().SetMasterVolumeLevelScalar(level / 100, None)
    return ok(f"Volume set to {level} percent.")


@tool("change_volume", "Raise or lower volume", {"delta": "e.g. +10 or -10"}, CATEGORY)
def change_volume(delta: int) -> ToolResult:
    v = _volume_iface()
    cur = int(round(v.GetMasterVolumeLevelScalar() * 100))
    new = max(0, min(100, cur + int(delta)))
    v.SetMasterVolumeLevelScalar(new / 100, None)
    return ok(f"Volume is now {new} percent.")


@tool("mute", "Mute or unmute audio", {"state": "true to mute, false to unmute"}, CATEGORY)
def mute(state: bool = True) -> ToolResult:
    _volume_iface().SetMute(1 if state else 0, None)
    return ok("Muted." if state else "Unmuted.")


@tool("set_brightness", "Set screen brightness", {"level": "0-100"}, CATEGORY)
def set_brightness(level: int) -> ToolResult:
    import screen_brightness_control as sbc
    level = max(0, min(100, int(level)))
    sbc.set_brightness(level)
    return ok(f"Brightness set to {level} percent.")


@tool("take_screenshot", "Capture the screen and save it", {"name": "optional file name"}, CATEGORY)
def take_screenshot(name: str = "") -> ToolResult:
    folder = Path.home() / "Pictures" / "Nova Screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    fname = (name or time.strftime("screenshot_%Y%m%d_%H%M%S")).replace(".png", "") + ".png"
    path = folder / fname
    pyautogui.screenshot(str(path))
    return ok("Screenshot saved.", str(path))


@tool("lock_pc", "Lock the computer", {}, CATEGORY)
def lock_pc() -> ToolResult:
    ctypes.windll.user32.LockWorkStation()
    return ok("Locking your PC.")


@tool("shutdown_pc", "Shut down the computer", {"delay": "seconds (default 10)"}, CATEGORY, dangerous=True)
def shutdown_pc(delay: int = 10) -> ToolResult:
    subprocess.Popen(["shutdown", "/s", "/t", str(int(delay))])
    return ok(f"Shutting down in {int(delay)} seconds.")


@tool("restart_pc", "Restart the computer", {"delay": "seconds (default 10)"}, CATEGORY, dangerous=True)
def restart_pc(delay: int = 10) -> ToolResult:
    subprocess.Popen(["shutdown", "/r", "/t", str(int(delay))])
    return ok(f"Restarting in {int(delay)} seconds.")


@tool("cancel_shutdown", "Cancel a pending shutdown or restart", {}, CATEGORY)
def cancel_shutdown() -> ToolResult:
    subprocess.Popen(["shutdown", "/a"])
    return ok("Shutdown cancelled.")


@tool("sleep_pc", "Put the computer to sleep", {}, CATEGORY, dangerous=True)
def sleep_pc() -> ToolResult:
    subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
    return ok("Going to sleep.")


# ----------------------------------------------------------------------------
# Clipboard
# ----------------------------------------------------------------------------
@tool("get_clipboard", "Read the clipboard text", {}, CATEGORY)
def get_clipboard() -> ToolResult:
    text = pyperclip.paste()
    if not text:
        return fail("The clipboard is empty.")
    return ok(f"The clipboard says: {text[:200]}", text)


@tool("set_clipboard", "Copy text to the clipboard", {"text": "text to copy"}, CATEGORY)
def set_clipboard(text: str) -> ToolResult:
    pyperclip.copy(text)
    return ok("Copied to clipboard.")


# ----------------------------------------------------------------------------
# Processes
# ----------------------------------------------------------------------------
@tool("list_processes", "Show the top processes by memory", {"limit": "default 5"}, CATEGORY)
def list_processes(limit: int = 5) -> ToolResult:
    procs = []
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            procs.append((p.info["name"], p.info["memory_info"].rss / 1e6))
        except Exception:
            continue
    procs.sort(key=lambda x: x[1], reverse=True)
    top = procs[: int(limit)]
    msg = ", ".join(f"{n} using {m:.0f} megabytes" for n, m in top[:3])
    return ok("Top memory users: " + msg, top)


@tool("kill_process", "Force-kill a process by name", {"name": "process name"}, CATEGORY, dangerous=True)
def kill_process(name: str) -> ToolResult:
    n = 0
    for p in psutil.process_iter(["name"]):
        if name.lower() in (p.info["name"] or "").lower():
            try:
                p.kill()
                n += 1
            except Exception:
                pass
    return ok(f"Killed {n} process(es) named {name}.") if n else fail(f"No process named {name}.")


# ----------------------------------------------------------------------------
# Keyboard / mouse automation (pyautogui)
# ----------------------------------------------------------------------------
@tool("type_text", "Type text into the active window", {"text": "text to type"}, CATEGORY)
def type_text(text: str) -> ToolResult:
    old = pyperclip.paste()
    pyperclip.copy(text)                     # paste = reliable for unicode/long text
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.1)
    pyperclip.copy(old)
    return ok("Typed it.")


@tool("press_keys", "Press a key or shortcut", {"keys": "e.g. enter, ctrl+c, alt+tab, win+d"}, CATEGORY)
def press_keys(keys: str) -> ToolResult:
    parts = [k.strip().lower() for k in keys.replace(" ", "").split("+")]
    pyautogui.hotkey(*parts) if len(parts) > 1 else pyautogui.press(parts[0])
    return ok(f"Pressed {keys}.")


@tool("mouse_click", "Click at screen coordinates", {"x": "x", "y": "y", "button": "left/right", "double": "true/false"}, CATEGORY)
def mouse_click(x: int, y: int, button: str = "left", double: bool = False) -> ToolResult:
    pyautogui.click(int(x), int(y), button=button, clicks=2 if double else 1)
    return ok("Clicked.")


@tool("mouse_scroll", "Scroll the mouse wheel", {"amount": "positive=up, negative=down"}, CATEGORY)
def mouse_scroll(amount: int = -500) -> ToolResult:
    pyautogui.scroll(int(amount))
    return ok("Scrolled.")


# ----------------------------------------------------------------------------
# Windows settings
# ----------------------------------------------------------------------------
@tool("open_settings", "Open a Windows Settings page", {"page": "wifi, bluetooth, display, sound, battery, update, apps, privacy..."}, CATEGORY)
def open_settings(page: str = "settings") -> ToolResult:
    key = page.lower().strip()
    uri = SETTINGS_PAGES.get(key)
    if not uri:
        m = difflib.get_close_matches(key, SETTINGS_PAGES.keys(), n=1, cutoff=0.5)
        uri = SETTINGS_PAGES[m[0]] if m else "ms-settings:"
    os.startfile(uri)
    return ok(f"Opening {page} settings.")
