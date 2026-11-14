"""Auto-click the Copilot "Allow Once" button in VS Code (macOS).

Takes periodic screenshots, template-matches the button with OpenCV and, only
while an allowed editor is the frontmost app, reads the command shown above
the button with macOS Vision. It clicks only when the whole command matches a
line of the allow list, then re-checks the screen once more right before the
click. Anything unexpected means no click.

Press Ctrl+Option+Cmd+P to pause or resume. Stop with Ctrl+C or by flicking
the mouse into a corner of the main screen.

This module imports pyautogui and the macOS frameworks lazily, inside the
functions that need them, so the decision logic can be tested headless.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from allowlist import AllowList, AllowListWatcher
from command_region import CropSettings, Rect, TextLine, command_crop, extract_command, vision_box_to_rect
from state import Status

LOG = logging.getLogger("copilot-click")

APP_NAME = "copilot-click"
ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_TEMPLATE = ASSETS_DIR / "allow_once_ss.png"
CURSOR_BUNDLE_ID = "com.todesktop.230313mzl4w4u92"
EDITOR_FAMILIES = {
    "com.microsoft.VSCode": "VS Code",
    "com.microsoft.VSCodeInsiders": "VS Code Insiders",
    CURSOR_BUNDLE_ID: "Cursor (untested)",
}
ALLOWED_BUNDLE_IDS = frozenset(EDITOR_FAMILIES)
DEFAULT_THRESHOLD = 0.85
SCALES = (1.0, 0.95, 1.05, 0.9, 1.1)
CORNER_TOLERANCE = 2  # points from the screen edge on both axes
DISABLED_HOLD = 1.0  # seconds the red "disabled" frame stays up before exiting
DEFAULT_PAUSE_KEY = "p"
CONFIRM_TOLERANCE_PX = 3  # how far the button may move between the two looks
DEFAULT_CONFIRM_DELAY = 0.25  # seconds between deciding to click and the re-check

# NSEvent modifier flags (stable AppKit values), so the hotkey check needs no AppKit import.
MOD_SHIFT = 1 << 17
MOD_CONTROL = 1 << 18
MOD_OPTION = 1 << 19
MOD_COMMAND = 1 << 20
PAUSE_MODIFIERS = MOD_CONTROL | MOD_OPTION | MOD_COMMAND
_HOTKEY_CHECKED = MOD_SHIFT | MOD_CONTROL | MOD_OPTION | MOD_COMMAND  # caps lock and fn are ignored


# --- paths -------------------------------------------------------------------


def _xdg_dir(env: dict[str, str] | os._Environ[str], name: str, fallback: Path) -> Path:
    value = env.get(name, "")
    return Path(value) if value and Path(value).is_absolute() else fallback


def default_allow_path(env: dict[str, str] | None = None, home: Path | None = None) -> Path:
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    return _xdg_dir(env, "XDG_CONFIG_HOME", home / ".config") / APP_NAME / "allow.txt"


def default_log_path(platform: str | None = None, env: dict[str, str] | None = None, home: Path | None = None) -> Path:
    platform = sys.platform if platform is None else platform
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    if platform == "darwin":
        return home / "Library" / "Logs" / APP_NAME / f"{APP_NAME}.log"
    return _xdg_dir(env, "XDG_STATE_HOME", home / ".local" / "state") / APP_NAME / f"{APP_NAME}.log"


# --- template matching --------------------------------------------------------


@dataclass(frozen=True)
class Match:
    score: float
    x: int  # centre of the match, in screenshot pixels
    y: int
    template: str = ""
    width: int = 0  # matched template size, in screenshot pixels
    height: int = 0

    def rect(self) -> Rect:
        width, height = self.width or 80, self.height or 32
        return Rect(self.x - width // 2, self.y - height // 2, width, height)


def _named_templates(
    template_bgr: np.ndarray | Sequence[tuple[str, np.ndarray]],
) -> list[tuple[str, np.ndarray]]:
    if isinstance(template_bgr, np.ndarray):
        return [("", template_bgr)]
    return [(name, image) for name, image in template_bgr]


def find_button(
    screen_bgr: np.ndarray,
    template_bgr: np.ndarray | Sequence[tuple[str, np.ndarray]],
    threshold: float,
    margin: int = 0,
) -> Match | None:
    """Return the best template match at or above threshold, or None.

    template_bgr is one BGR image, or several (name, image) pairs. The winner
    is the pair with the highest score; ties keep the earlier pair.

    margin crops that many pixels off every screen edge before matching, so the
    overlay border can never be mistaken for (or hide) the button.
    """
    if margin:
        screen_bgr = screen_bgr[margin:-margin, margin:-margin]
    screen = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)

    best: Match | None = None
    for name, template_bgr_one in _named_templates(template_bgr):
        template = cv2.cvtColor(template_bgr_one, cv2.COLOR_BGR2GRAY)
        for scale in SCALES:
            scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            th, tw = scaled.shape
            if th > screen.shape[0] or tw > screen.shape[1] or th < 1 or tw < 1:
                continue
            result = cv2.matchTemplate(screen, scaled, cv2.TM_CCOEFF_NORMED)
            _, score, _, (left, top) = cv2.minMaxLoc(result)
            if score >= threshold and (best is None or score > best.score):
                best = Match(float(score), left + tw // 2 + margin, top + th // 2 + margin, name, tw, th)
    return best


def load_templates(path: Path) -> list[tuple[str, np.ndarray]]:
    """Load one PNG, or every PNG in a directory, as (filename, BGR image)."""
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".png")
    elif path.is_file():
        files = [path]
    else:
        raise SystemExit(f"Could not read template image: {path}")
    loaded: list[tuple[str, np.ndarray]] = []
    for file in files:
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read template image: {file}")
        loaded.append((file.name, image))
    if not loaded:
        raise SystemExit(f"No PNG templates found in {path}")
    return loaded


def pixels_to_points(match: Match, scale: float) -> tuple[float, float]:
    """Convert screenshot pixel coords to pyautogui point coords (scale is 2.0 on Retina)."""
    return match.x / scale, match.y / scale


def at_corner(x: float, y: float, width: int, height: int) -> bool:
    """True when the pointer is within tolerance of a corner of the main screen.

    Positions outside the main screen (other displays) never count.
    """
    t = CORNER_TOLERANCE
    in_x = 0 <= x <= t or width - 1 - t <= x <= width - 1
    in_y = 0 <= y <= t or height - 1 - t <= y <= height - 1
    return in_x and in_y


def pick_window(windows: Sequence[dict[str, Any]], pid: int | None, x: float, y: float) -> Rect | None:
    """Bounds (points) of the front-most normal window of pid that contains (x, y).

    windows is CGWindowListCopyWindowInfo output, ordered front to back.
    """
    if pid is None:
        return None
    for info in windows:
        if info.get("kCGWindowOwnerPID") != pid or info.get("kCGWindowLayer", 0) != 0:
            continue
        bounds = info.get("kCGWindowBounds") or {}
        try:
            left, top = float(bounds["X"]), float(bounds["Y"])
            width, height = float(bounds["Width"]), float(bounds["Height"])
        except (KeyError, TypeError, ValueError):
            continue
        if left <= x < left + width and top <= y < top + height:
            return Rect(math.floor(left), math.floor(top), math.ceil(width), math.ceil(height))
    return None


def scale_rect(rect: Rect, scale: float) -> Rect:
    return Rect(
        math.floor(rect.left * scale),
        math.floor(rect.top * scale),
        math.ceil(rect.width * scale),
        math.ceil(rect.height * scale),
    )


# --- decisions (pure) ------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One look at the screen: where the button is and what command is above it."""

    match: Match
    command: str | None  # None: no trustworthy command text (see reason)
    reason: str
    front_app: str | None
    scale: float = 1.0  # screenshot pixels per point (2.0 on Retina)
    crop: np.ndarray | None = None


@dataclass(frozen=True)
class Policy:
    dry_run: bool = False
    trust_all: bool = False
    any_app: bool = False
    confirm_delay: float = DEFAULT_CONFIRM_DELAY
    tolerance_px: int = CONFIRM_TOLERANCE_PX


def decide_click(*, paused: bool, policy: Policy, observation: Observation, allow: AllowList) -> tuple[bool, str]:
    """Whether a detection may be clicked (before the re-check), and the log outcome."""
    if paused:
        return False, "skipped: paused"
    if not policy.trust_all:
        if observation.command is None:
            return False, f"skipped: {observation.reason}"
        verdict = allow.evaluate(observation.command)
        if not verdict.allowed:
            return False, f"skipped: {verdict.reason}"
    if policy.dry_run:
        return False, "dry run: would click"
    return True, "clicked"


def front_app_ok(front: str | None, first_front: str | None, any_app: bool) -> bool:
    """The app in front at click time must be the one we saw, and (unless --any-app) an allowed editor."""
    if front is None or front != first_front:
        return False
    return any_app or front in ALLOWED_BUNDLE_IDS


def confirm_click(
    first: Observation, second: Observation | None, front_now: str | None, policy: Policy
) -> tuple[bool, str]:
    """Re-check right before clicking: same button, same place, same command, same app."""
    if second is None:
        return False, "button gone on re-check"
    dx, dy = abs(second.match.x - first.match.x), abs(second.match.y - first.match.y)
    if dx > policy.tolerance_px or dy > policy.tolerance_px:
        return False, f"button moved on re-check ({dx}px, {dy}px)"
    if not policy.trust_all and (second.command is None or second.command != first.command):
        return False, "command changed on re-check"
    if not front_app_ok(second.front_app, first.front_app, policy.any_app) or not front_app_ok(
        front_now, first.front_app, policy.any_app
    ):
        return False, "frontmost app changed on re-check"
    return True, "confirmed"


def handle_detection(
    first: Observation,
    *,
    status: Status,
    policy: Policy,
    allow: AllowList,
    look_again: Callable[[], Observation | None],
    click: Callable[[Observation], None],
) -> str:
    """Decide, re-check and maybe click one detection. Returns the log outcome."""
    ok, outcome = decide_click(paused=status.paused, policy=policy, observation=first, allow=allow)
    if not ok:
        return outcome
    if status.stop.wait(policy.confirm_delay):
        return "skipped: stopping"
    second = look_again()
    ok, why = confirm_click(first, second, status.front_app, policy)
    if not ok:
        return f"skipped: {why}"
    if status.paused:
        return "skipped: paused"
    if status.stop.is_set() or second is None:
        return "skipped: stopping"
    click(second)
    return "clicked"


# --- pause hotkey (pure) ------------------------------------------------------------


def is_pause_hotkey(flags: int, chars: str, key: str = DEFAULT_PAUSE_KEY) -> bool:
    """True only for exactly Ctrl+Option+Cmd+<key>: Shift or a missing modifier does not count."""
    return bool(key) and (flags & _HOTKEY_CHECKED) == PAUSE_MODIFIERS and chars.lower() == key.lower()


def hotkey_label(key: str = DEFAULT_PAUSE_KEY) -> str:
    return f"Ctrl+Option+Cmd+{key.upper()}"


# --- logging ------------------------------------------------------------------------

_SECRETS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{16,})"), "[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"), "[REDACTED]"),
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED]"),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{8,}"), "[REDACTED]"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{30,}"), "[REDACTED]"),
    (re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1[REDACTED]"),
    (
        re.compile(r"(?i)(--?(?:password|passwd|pass|token|secret|api[-_]?key|auth)(?:=|\s+))(\S+)"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(?i)\b((?:[a-z_]*_)?(?:password|passwd|token|secret|api[-_]?key)=)(\S+)"),
        r"\1[REDACTED]",
    ),
    (re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^/\s:@]+:)[^@\s/]+@"), r"\1[REDACTED]@"),
)


def redact(text: str) -> str:
    """Mask obvious secrets (tokens, keys, --password=...) before text is logged."""
    for pattern, replacement in _SECRETS:
        text = pattern.sub(replacement, text)
    return text


def describe_text(text: str | None) -> str:
    """One-line, redacted form of OCR text for logs; line breaks show as ⏎."""
    if text is None or not str(text).strip():
        return "(no text)"
    return redact(" \u23ce ".join(" ".join(line.split()) for line in str(text).splitlines() if line.strip()))


def format_detection(when: str, template: str, score: float, text: str | None, outcome: str) -> str:
    shown = describe_text(text)
    return f"{when}  template={template}  score={score:.3f}  text={shown}  {outcome}"


def append_private(path: Path, line: str) -> None:
    """Append a line to a file that only the current user can read (0600, directory 0700)."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        if os.fstat(fd).st_mode & 0o077:
            os.fchmod(fd, 0o600)
        os.write(fd, (line + "\n").encode("utf-8"))
    finally:
        os.close(fd)


def append_detection(path: Path, template: str, score: float, text: str | None, outcome: str) -> None:
    """Append one detection. Callers must not use this when nothing matched."""
    append_private(path, format_detection(time.strftime("%Y-%m-%d %H:%M:%S"), template, score, text, outcome))


def log_messages(messages: Sequence[str]) -> None:
    for message in messages:
        LOG.warning(message)


class EditorGate:
    """Tracks whether an allowed editor is frontmost, logging only on changes."""

    def __init__(self) -> None:
        self._paused_for: str | None = None
        self._family: str | None = None

    def is_open(self, front: str | None) -> bool:
        if front not in ALLOWED_BUNDLE_IDS:
            if front != self._paused_for:
                LOG.info("waiting: frontmost app is %s, not an allowed editor", front)
                self._paused_for = front
            self._family = None
            return False
        family = EDITOR_FAMILIES[front]
        if self._paused_for is not None or self._family != family:
            LOG.info("%s is frontmost, scanning", family)
            self._paused_for = None
            self._family = family
        return True


# --- looking at the screen ------------------------------------------------------------


@dataclass
class Looker:
    """Captures the screen, finds the button and reads the command. Collaborators are injected."""

    templates: Sequence[tuple[str, np.ndarray]]
    capture: Callable[[], np.ndarray]
    points_width: Callable[[], float]
    front_app: Callable[[], str | None]
    ocr: Callable[[np.ndarray], list[TextLine] | None]
    window_bounds: Callable[[float, float], Rect | None] = lambda x, y: None
    threshold: float = DEFAULT_THRESHOLD
    border: int = 6
    read_text: bool = True  # False only with --trust-all
    crop_settings: CropSettings | None = None
    min_confidence: float = 0.5

    def look(self) -> Observation | None:
        front = self.front_app()
        screen = self.capture()
        height, width = screen.shape[:2]
        scale = width / self.points_width()
        margin = math.ceil(self.border * scale) + 1  # keep the overlay border out of the match
        match = find_button(screen, self.templates, self.threshold, margin=margin)
        if match is None:
            return None
        if not self.read_text:
            return Observation(match, None, "not read (--trust-all)", front, scale)
        button = match.rect()
        bounds_pt = self.window_bounds(match.x / scale, match.y / scale)
        bounds = scale_rect(bounds_pt, scale) if bounds_pt is not None else None
        crop = command_crop(button, width, height, scale, margin=margin, bounds=bounds, settings=self.crop_settings)
        if crop is None:
            return Observation(match, None, "no room above the button", front, scale)
        image = screen[crop.top : crop.bottom, crop.left : crop.right]
        lines = self.ocr(image)
        if lines is None:
            return Observation(match, None, "text recognition unavailable", front, scale, image)
        extraction = extract_command(
            image, lines, button.offset(-crop.left, -crop.top), min_confidence=self.min_confidence
        )
        return Observation(match, extraction.command, extraction.reason, front, scale, image)


# --- macOS / pyautogui glue (not unit-tested; imported lazily) -------------------------

_vision_error_reported = False


def vision_available() -> bool:
    try:
        import Vision  # noqa: F401
        from Foundation import NSData  # noqa: F401
        from Quartz import CIImage  # noqa: F401
    except ImportError:
        return False
    return True


def recognize_lines(image_bgr: np.ndarray) -> list[TextLine] | None:
    """Read text with macOS Vision, with each line's box. None when Vision is unavailable or fails."""
    global _vision_error_reported
    if image_bgr.size == 0:
        return []
    try:
        import Vision
        from Foundation import NSData
        from Quartz import CIImage
    except ImportError:
        return None
    try:
        ok, encoded = cv2.imencode(".png", image_bgr)
        if not ok:
            return None
        payload = encoded.tobytes()
        data = NSData.dataWithBytes_length_(payload, len(payload))
        image = CIImage.imageWithData_(data)
        if image is None:
            return None
        request = Vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(False)
        handler = Vision.VNImageRequestHandler.alloc().initWithCIImage_options_(image, None)
        performed = handler.performRequests_error_([request], None)
        success = bool(performed[0]) if isinstance(performed, tuple) else bool(performed)
        if not success:
            return None
        height, width = image_bgr.shape[:2]
        lines: list[TextLine] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            box = observation.boundingBox()
            rect = vision_box_to_rect(box.origin.x, box.origin.y, box.size.width, box.size.height, width, height)
            lines.append(TextLine(str(candidates[0].string()), rect, float(candidates[0].confidence())))
        return lines
    except Exception:
        if not _vision_error_reported:
            LOG.exception("text recognition failed; not clicking")
            _vision_error_reported = True
        return None


def frontmost_app() -> tuple[str | None, int | None]:
    """(bundle id, pid) of the frontmost app. Pumps the run loop so NSWorkspace stays current."""
    try:
        from AppKit import NSWorkspace

        from overlay import pump_runloop
    except ImportError:
        return None, None
    pump_runloop()
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        return None, None
    return str(app.bundleIdentifier()), int(app.processIdentifier())


def window_bounds_for(pid: int | None, x: float, y: float) -> Rect | None:
    """Bounds (points) of pid's window under (x, y), from Quartz. None when unknown."""
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGNullWindowID,
            kCGWindowListExcludeDesktopElements,
            kCGWindowListOptionOnScreenOnly,
        )
    except ImportError:
        return None
    options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
    windows = CGWindowListCopyWindowInfo(options, kCGNullWindowID) or []
    return pick_window([dict(info) for info in windows], pid, x, y)


def capture_bgr() -> np.ndarray:
    import pyautogui

    return cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)


def screen_points_width() -> float:
    import pyautogui

    return float(pyautogui.size().width)


def click_at(x: float, y: float) -> None:
    """Click, then put the cursor back where the user had it."""
    import pyautogui

    origin = pyautogui.position()
    pyautogui.click(x, y)
    pyautogui.moveTo(origin.x, origin.y)


def install_pause_monitor(status: Status, key: str) -> Any:
    """Install a macOS global key monitor for Ctrl+Option+Cmd+<key>. None without AppKit."""
    label = hotkey_label(key)
    try:
        from AppKit import NSEvent

        try:
            from AppKit import NSEventMaskKeyDown
        except ImportError:
            from AppKit import NSKeyDownMask as NSEventMaskKeyDown
    except ImportError:
        LOG.warning("pause hotkey %s unavailable (AppKit is not installed)", label)
        return None

    def handler(event: Any) -> None:
        try:
            flags = int(event.modifierFlags())
            chars = str(event.charactersIgnoringModifiers() or "")
        except Exception:  # never let an exception escape into AppKit
            LOG.exception("could not read a key event")
            return
        if not is_pause_hotkey(flags, chars, key):
            return
        status.paused = not status.paused
        if status.paused:
            LOG.warning("PAUSED: no clicks until you press %s again", label)
        else:
            LOG.warning("RESUMED: scanning again (%s pauses)", label)

    monitor = NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, handler)
    if monitor is None:
        LOG.warning("pause hotkey %s unavailable (grant Accessibility permission)", label)
    return monitor


def remove_pause_monitor(monitor: Any) -> None:
    if monitor is None:
        return
    from AppKit import NSEvent

    NSEvent.removeMonitor_(monitor)


class NullOverlay:
    """Stand-in used with --no-overlay or without AppKit."""

    def show(self, state: Any) -> None:
        pass

    def hold(self, seconds: float) -> None:
        time.sleep(seconds)

    def close(self) -> None:
        pass


def save_crop(directory: Path, observation: Observation, outcome: str) -> None:
    if observation.crop is None or observation.crop.size == 0:
        return
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", outcome.lower()).strip("-")[:40]
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}-{slug}.png"
    cv2.imwrite(str(directory / name), observation.crop)


# --- the loop ---------------------------------------------------------------------------


def scan_loop(args: argparse.Namespace, looker: Looker, status: Status, watcher: AllowListWatcher) -> None:
    """Screenshot, match and click until status.stop is set. Runs on a worker thread."""
    gate = EditorGate()
    policy = Policy(
        dry_run=args.dry_run,
        trust_all=args.trust_all,
        any_app=args.any_app,
        confirm_delay=args.confirm_delay,
    )
    while not status.stop.wait(args.interval):
        if not args.trust_all and watcher.refresh():
            log_messages(watcher.messages)
            LOG.info("allow list: %d usable line(s) from %s", len(watcher.allow), watcher.path)
        if status.paused:
            status.scanning = False
            continue
        status.scanning = args.any_app or gate.is_open(status.front_app)
        if not status.scanning:
            continue

        first = looker.look()
        if first is None:
            continue
        status.detections += 1
        outcome = handle_detection(
            first,
            status=status,
            policy=policy,
            allow=watcher.allow if not args.trust_all else AllowList(),
            look_again=looker.look,
            click=lambda seen: click_at(*pixels_to_points(seen.match, seen.scale)),
        )
        if outcome == "clicked":
            status.clicks += 1
        template_name = first.match.template or "template"
        LOG.info("%s score=%.3f text=%s - %s", template_name, first.match.score, describe_text(first.command), outcome)
        append_detection(args.log_file, template_name, first.match.score, first.command, outcome)
        if args.save_crops is not None:
            save_crop(args.save_crops, first, outcome)
        if outcome == "clicked" and args.max_clicks and status.clicks >= args.max_clicks:
            LOG.info("reached --max-clicks (%d), exiting", args.max_clicks)
            status.stop.set()
            return
        status.stop.wait(args.cooldown)


def scan_thread(args: argparse.Namespace, looker: Looker, status: Status, watcher: AllowListWatcher) -> None:
    import pyautogui

    try:
        scan_loop(args, looker, status, watcher)
    except pyautogui.FailSafeException:
        status.disable()  # pyautogui saw the pointer in a corner mid-action
    except Exception as exc:  # noqa: BLE001 - re-raised on the main thread by run()
        status.error = exc
        status.stop.set()


def summary(status: Status, dry_run: bool) -> str:
    if dry_run:
        return f"Dry run: found the button {status.detections} time(s), clicked 0."
    noun = "prompt" if status.clicks == 1 else "prompts"
    return f"Auto-allowed {status.clicks} {noun} this session."


def make_overlay(args: argparse.Namespace) -> Any:
    if args.no_overlay:
        return NullOverlay()
    try:
        from overlay import Overlay
    except ImportError:
        LOG.warning("overlay unavailable (AppKit is not installed); continuing without it")
        return NullOverlay()
    return Overlay(args.border, args.dry_run)


def run(args: argparse.Namespace) -> None:
    import pyautogui

    templates = load_templates(args.template)
    watcher = AllowListWatcher(args.allow_file)
    if args.trust_all:
        LOG.warning("--trust-all: clicking EVERY match without reading the command. This is dangerous.")
    else:
        watcher.refresh()
        log_messages(watcher.messages)
        if watcher.allow:
            LOG.info("allow list: %d usable line(s) from %s", len(watcher.allow), args.allow_file)
        else:
            LOG.warning(
                "allow list %s has no usable lines; nothing will be clicked. See allow.txt.example.",
                args.allow_file,
            )
        if not vision_available():
            LOG.error("macOS Vision is not available, so commands cannot be read; nothing will be clicked")
    if sys.platform != "darwin":
        LOG.warning("copilot-click is built for macOS; on %s it will not click without --trust-all", sys.platform)
    LOG.info("detection log: %s", args.log_file)

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    width, height = pyautogui.size()

    # The overlay and the frontmost-app lookup both need the main thread's run loop.
    overlay = make_overlay(args)
    front, pid = frontmost_app()
    status = Status(front_app=front, front_pid=pid)
    looker = Looker(
        templates=templates,
        capture=capture_bgr,
        points_width=screen_points_width,
        front_app=lambda: status.front_app,
        ocr=recognize_lines,
        window_bounds=lambda x, y: window_bounds_for(status.front_pid, x, y),
        threshold=args.threshold,
        border=args.border,
        read_text=not args.trust_all,
        crop_settings=CropSettings(max_width=args.command_width, max_height=args.command_height),
        min_confidence=args.min_ocr_confidence,
    )
    # Hold the monitor for the life of run(); AppKit drops it if nothing references it.
    hotkey_monitor = install_pause_monitor(status, args.pause_key)
    worker = threading.Thread(target=scan_thread, args=(args, looker, status, watcher), daemon=True)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    names = ", ".join(name for name, _ in templates)
    policy = "TRUST-ALL" if args.trust_all else f"allow list {args.allow_file}"
    print(
        f"[{mode}] watching for {names} (threshold {args.threshold}, {policy}). "
        f"Pause/resume with {hotkey_label(args.pause_key)}. Ctrl+C or a corner flick to stop."
    )
    worker.start()
    try:
        while not status.stop.is_set():
            # Also pumps the run loop for the overlay and the hotkey monitor.
            status.front_app, status.front_pid = frontmost_app()
            pointer = pyautogui.position()
            if at_corner(pointer.x, pointer.y, width, height):
                status.disable()
                break
            overlay.show(status.state)
            if status.front_app is None:
                time.sleep(0.05)  # no run loop to pump (not macOS); do not spin

        if status.disabled:
            overlay.show(status.state)
            LOG.warning("disabled: mouse flicked into a screen corner")
            overlay.hold(DISABLED_HOLD)
    finally:
        status.stop.set()
        try:
            remove_pause_monitor(hotkey_monitor)
        except Exception:
            LOG.exception("could not remove the pause hotkey monitor")
        worker.join(timeout=5)
        overlay.close()
        print(summary(status, args.dry_run))

    if status.error is not None:
        raise status.error


def _pause_key(value: str) -> str:
    if len(value) != 1 or not value.isalnum():
        raise argparse.ArgumentTypeError("must be a single letter or digit")
    return value.lower()


def _fraction(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--template",
        type=Path,
        default=ASSETS_DIR,
        help="PNG file, or a directory of PNGs to try (default: assets/)",
    )
    p.add_argument("--threshold", type=_fraction, default=DEFAULT_THRESHOLD, help="min match score, 0-1")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between scans")
    p.add_argument("--cooldown", type=float, default=1.5, help="seconds to wait after a match")
    p.add_argument(
        "--confirm-delay",
        type=float,
        default=DEFAULT_CONFIRM_DELAY,
        help="seconds between deciding to click and the re-check of the screen",
    )
    p.add_argument("--max-clicks", type=int, default=0, help="exit after N clicks (0 = unlimited)")
    p.add_argument("--dry-run", action="store_true", help="log what would happen but never click")
    p.add_argument(
        "--trust-all",
        action="store_true",
        help="DANGEROUS: click every match without reading the command (off by default)",
    )
    p.add_argument(
        "--allow-file",
        "--allow",
        dest="allow_file",
        type=Path,
        default=default_allow_path(),
        help="allow list (default: %(default)s)",
    )
    p.add_argument(
        "--log-file",
        "--log",
        dest="log_file",
        type=Path,
        default=default_log_path(),
        help="detection log, created mode 0600 (default: %(default)s)",
    )
    p.add_argument(
        "--pause-key",
        type=_pause_key,
        default=DEFAULT_PAUSE_KEY,
        help="letter used with Ctrl+Option+Cmd to pause/resume (default: p)",
    )
    p.add_argument(
        "--any-app",
        action="store_true",
        help="do not require VS Code to be frontmost (the app must still not change before a click)",
    )
    p.add_argument(
        "--command-width",
        type=float,
        default=CropSettings.max_width,
        help="points to the right of the button to read (default: %(default)s)",
    )
    p.add_argument(
        "--command-height",
        type=float,
        default=CropSettings.max_height,
        help="points above the button to read (default: %(default)s)",
    )
    p.add_argument(
        "--min-ocr-confidence",
        type=_fraction,
        default=0.5,
        help="refuse when Vision is less sure than this about a command line (default: %(default)s)",
    )
    p.add_argument(
        "--save-crops",
        type=Path,
        default=None,
        metavar="DIR",
        help="save each command crop as a PNG for troubleshooting (may contain secrets)",
    )
    p.add_argument("--border", type=int, default=6, help="glow border width in points")
    p.add_argument("--no-overlay", action="store_true", help="do not draw the glowing border")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    try:
        run(build_parser().parse_args(argv))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
