"""Auto-click the Copilot "Allow Once" button in VS Code or Cursor (macOS).

Takes periodic screenshots, template-matches the button with OpenCV, and clicks
it with pyautogui. Only clicks while VS Code, VS Code Insiders, or Cursor is
the frontmost app.

A live click also requires the command text above the button to match a regex
in allow.txt. That check is off only with --trust-all. Press the pause key
(default: p) to stop clicking without exiting.

A glowing border around the main screen shows the state (see README). Stop for
good with Ctrl+C or by flicking the mouse into a corner of the main screen.
"""

import argparse
import math
import os
import re
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyautogui

# AppKit, Vision, and the overlay window exist only on macOS. Pure tests import
# this module on Linux, so a missing framework must not break collection.
try:
    from AppKit import NSWorkspace
except ImportError:
    NSWorkspace = None

try:
    from AppKit import NSEvent, NSEventMaskKeyDown
except ImportError:
    try:
        from AppKit import NSEvent, NSKeyDownMask as NSEventMaskKeyDown
    except ImportError:
        NSEvent = None
        NSEventMaskKeyDown = None

try:
    import Vision
    from Foundation import NSData
    from Quartz import CIImage
except ImportError:
    Vision = None
    NSData = None
    CIImage = None

try:
    from overlay import Overlay, pump_runloop
except ImportError:
    Overlay = None
    pump_runloop = None

from state import Status

ASSETS_DIR = Path(__file__).parent / "assets"
DEFAULT_TEMPLATE = ASSETS_DIR / "allow_once_ss.png"
DEFAULT_ALLOW = Path("allow.txt")
DEFAULT_LOG = Path("copilot-click.log")
CURSOR_BUNDLE_ID = "com.todesktop.230313mzl4w4u92"
EDITOR_FAMILIES = {
    "com.microsoft.VSCode": "VS Code",
    "com.microsoft.VSCodeInsiders": "VS Code Insiders",
    CURSOR_BUNDLE_ID: "Cursor",
}
ALLOWED_BUNDLE_IDS = frozenset(EDITOR_FAMILIES)
DEFAULT_THRESHOLD = 0.85
SCALES = (1.0, 0.95, 1.05, 0.9, 1.1)
CORNER_TOLERANCE = 2  # points from the screen edge on both axes
DISABLED_HOLD = 1.0  # seconds the red "disabled" frame stays up before exiting
DEFAULT_PAUSE_KEY = "p"
_vision_error_reported = False


@dataclass(frozen=True)
class Match:
    score: float
    x: int  # centre of the match, in screenshot pixels
    y: int
    template: str = ""
    width: int = 0  # matched template size, in screenshot pixels
    height: int = 0


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
                best = Match(
                    float(score),
                    left + tw // 2 + margin,
                    top + th // 2 + margin,
                    name,
                    tw,
                    th,
                )
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


def crop_command_region(screen_bgr: np.ndarray, match: Match) -> np.ndarray:
    """Crop the band above the matched button, where the command text sits.

    The button itself is excluded so its label is not treated as the command.
    """
    height, width = screen_bgr.shape[:2]
    btn_w = match.width or 80
    btn_h = match.height or 32
    button_top = match.y - btn_h // 2
    bottom = max(0, button_top - 2)
    region_h = max(btn_h * 4, 64)
    top = max(0, bottom - region_h)
    pad_x = max(btn_w, 40)
    left = max(0, match.x - btn_w // 2 - pad_x)
    right = min(width, match.x + btn_w // 2 + pad_x)
    if bottom <= top or right <= left:
        return screen_bgr[0:0, 0:0]
    return screen_bgr[top:bottom, left:right]


def recognize_text(image_bgr: np.ndarray) -> str | None:
    """Read text with macOS Vision. None when Vision is missing or finds nothing."""
    if Vision is None or NSData is None or CIImage is None:
        return None
    if image_bgr.size == 0:
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
        observations = request.results() or []
        parts: list[str] = []
        for observation in observations:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            parts.append(str(candidates[0].string()))
        text = " ".join(part.strip() for part in parts if part and part.strip())
        return text or None
    except Exception as exc:
        global _vision_error_reported
        if not _vision_error_reported:
            log(f"text recognition failed ({exc}); not clicking without --trust-all")
            _vision_error_reported = True
        return None


def _compile_patterns(patterns: Iterable[str | re.Pattern[str]]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        if isinstance(pattern, re.Pattern):
            compiled.append(pattern)
            continue
        line = str(pattern).strip()
        if not line or line.startswith("#"):
            continue
        try:
            compiled.append(re.compile(line))
        except re.error as exc:
            log(f"allow list skipped invalid regex: {line} ({exc})")
    return compiled


def command_allowed(text: str | None, patterns: Sequence[str | re.Pattern[str]]) -> bool:
    """True when text matches one regex.

    An empty pattern list denies. Blank lines and # comments are ignored.
    An invalid regex is skipped and reported, not raised.
    """
    if text is None or not str(text).strip():
        return False
    compiled = _compile_patterns(patterns)
    if not compiled:
        return False
    return any(pattern.search(str(text)) for pattern in compiled)


def load_allow_patterns(path: Path) -> list[re.Pattern[str]]:
    """Compile allow.txt. A missing file yields no patterns (clicks stay off)."""
    if not path.is_file():
        return []
    return _compile_patterns(path.read_text(encoding="utf-8-sig").splitlines())


def decide_click(
    *,
    paused: bool,
    dry_run: bool,
    trust_all: bool,
    ocr_text: str | None,
    patterns: Sequence[str | re.Pattern[str]],
) -> tuple[bool, str]:
    """Whether a detection may be clicked, and the session-log outcome."""
    if paused:
        return False, "skipped: paused"
    if dry_run:
        return False, "skipped: dry run"
    if trust_all:
        return True, "clicked"
    if not _compile_patterns(patterns):
        return False, "skipped: allow list empty"
    if ocr_text is None or not str(ocr_text).strip():
        return False, "skipped: no text"
    if not command_allowed(ocr_text, patterns):
        return False, "skipped: not allowed"
    return True, "clicked"


def next_paused(paused: bool, pressed: str, hotkey: str) -> bool:
    """Toggle pause when the pressed key is the hotkey. Any other key is ignored."""
    if pressed and hotkey and pressed.lower() == hotkey.lower():
        return not paused
    return paused


def format_detection(when: str, template: str, score: float, text: str | None, outcome: str) -> str:
    if text is None or not str(text).strip():
        shown = "(no text)"
    else:
        shown = " ".join(str(text).split())
    return f"{when}  template={template}  score={score:.3f}  text={shown}  {outcome}"


def append_detection(path: Path, template: str, score: float, text: str | None, outcome: str) -> None:
    """Append one detection. Callers must not use this when nothing matched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = format_detection(time.strftime("%Y-%m-%d %H:%M:%S"), template, score, text, outcome)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def frontmost_bundle_id() -> str | None:
    # NSWorkspace only refreshes frontmostApplication while the run loop runs,
    # which a plain Python polling loop never does, so pump it before reading.
    if NSWorkspace is None or pump_runloop is None:
        return None
    pump_runloop()
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    return str(app.bundleIdentifier()) if app is not None else None


def capture_bgr() -> np.ndarray:
    return cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)


def click_at(x: float, y: float) -> None:
    """Click, then put the cursor back where the user had it."""
    origin = pyautogui.position()
    pyautogui.click(x, y)
    pyautogui.moveTo(origin.x, origin.y)


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}")


def install_pause_monitor(status: Status, hotkey: str):
    """Install a macOS global key monitor. No-op under pytest and without AppKit."""
    if "pytest" in sys.modules or os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    if NSEvent is None or NSEventMaskKeyDown is None:
        log(f"pause hotkey '{hotkey}' unavailable (AppKit is not installed)")
        return None

    def handler(event) -> None:
        try:
            chars = event.charactersIgnoringModifiers() or ""
        except Exception:
            return
        updated = next_paused(status.paused, str(chars), hotkey)
        if updated == status.paused:
            return
        status.paused = updated
        log("paused" if status.paused else "resumed")

    try:
        return NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(NSEventMaskKeyDown, handler)
    except Exception as exc:
        log(f"pause hotkey '{hotkey}' unavailable ({exc})")
        return None


class VSCodeGate:
    """Tracks whether an allowed editor is frontmost, logging only on changes."""

    def __init__(self) -> None:
        self._paused_for: str | None = None
        self._family: str | None = None

    def is_open(self, front: str | None) -> bool:
        if front not in ALLOWED_BUNDLE_IDS:
            if front != self._paused_for:
                log(f"paused: frontmost app is {front}, not an allowed editor")
                self._paused_for = front
            self._family = None
            return False
        family = EDITOR_FAMILIES[front]
        if self._paused_for is not None or self._family != family:
            log(f"{family} is frontmost, scanning")
            self._paused_for = None
            self._family = family
        return True


class NullOverlay:
    """Stand-in used with --no-overlay."""

    def show(self, state) -> None:
        pass

    def hold(self, seconds: float) -> None:
        time.sleep(seconds)

    def close(self) -> None:
        pass


def scan_loop(
    args: argparse.Namespace,
    templates: Sequence[tuple[str, np.ndarray]],
    status: Status,
    patterns: Sequence[re.Pattern[str]],
) -> None:
    """Screenshot, match and click until status.stop is set. Runs on a worker thread."""
    gate = VSCodeGate()
    while not status.stop.wait(args.interval):
        if status.paused:
            status.scanning = False
            continue
        status.scanning = args.any_app or gate.is_open(status.front_app)
        if not status.scanning:
            continue

        screen = capture_bgr()
        scale = screen.shape[1] / pyautogui.size().width
        margin = math.ceil(args.border * scale) + 1  # keep the overlay border out of the match
        match = find_button(screen, templates, args.threshold, margin=margin)
        if match is None:
            continue

        x, y = pixels_to_points(match, scale)
        status.detections += 1
        ocr_text = recognize_text(crop_command_region(screen, match))
        click, outcome = decide_click(
            paused=status.paused,
            dry_run=args.dry_run,
            trust_all=args.trust_all,
            ocr_text=ocr_text,
            patterns=patterns,
        )
        if click and status.stop.is_set():
            return
        if click and status.paused:
            click, outcome = False, "skipped: paused"

        template_name = match.template or "template"
        found = f"match {template_name} score={match.score:.3f} at ({x:.0f}, {y:.0f})"
        if click:
            click_at(x, y)
            status.clicks += 1
            log(f"{found} - clicked ({status.clicks})")
            append_detection(args.log, template_name, match.score, ocr_text, "clicked")
            if args.max_clicks and status.clicks >= args.max_clicks:
                print("Reached --max-clicks, exiting.")
                status.stop.set()
                return
        else:
            log(f"{found} - {outcome}")
            append_detection(args.log, template_name, match.score, ocr_text, outcome)

        status.stop.wait(args.cooldown)


def scan_thread(
    args: argparse.Namespace,
    templates: Sequence[tuple[str, np.ndarray]],
    status: Status,
    patterns: Sequence[re.Pattern[str]],
) -> None:
    try:
        scan_loop(args, templates, status, patterns)
    except pyautogui.FailSafeException:
        status.disable()  # pyautogui saw the pointer in a corner mid-action
    except Exception as exc:  # surface to the main thread rather than dying silently
        status.error = exc
        status.stop.set()


def summary(status: Status, dry_run: bool) -> str:
    if dry_run:
        return f"Dry run: found the button {status.detections} time(s), clicked 0."
    noun = "prompt" if status.clicks == 1 else "prompts"
    return f"Auto-allowed {status.clicks} {noun} this session."


def run(args: argparse.Namespace) -> None:
    templates = load_templates(args.template)
    patterns = load_allow_patterns(args.allow)
    if args.trust_all:
        log("--trust-all: clicking every match without an allow-list check")
    elif not patterns:
        log("allow list is empty; not clicking")
    else:
        log(f"allow list: {len(patterns)} pattern(s) from {args.allow}")

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    width, height = pyautogui.size()

    # The overlay and the frontmost-app lookup both need the main thread's run loop.
    if args.no_overlay or Overlay is None:
        if Overlay is None and not args.no_overlay:
            log("overlay unavailable (AppKit is not installed); continuing without it")
        overlay = NullOverlay()
    else:
        overlay = Overlay(args.border, args.dry_run)

    status = Status(front_app=frontmost_bundle_id())
    # Hold the monitor for the life of run(); AppKit drops it if nothing references it.
    hotkey_monitor = install_pause_monitor(status, args.pause_key)
    worker = threading.Thread(target=scan_thread, args=(args, templates, status, patterns), daemon=True)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    names = ", ".join(name for name, _ in templates)
    policy = "trust-all" if args.trust_all else "allow list"
    print(
        f"[{mode}] watching for {names} (threshold {args.threshold}, {policy}). "
        f"Pause with '{args.pause_key}'. Ctrl+C or a corner flick to stop."
    )
    worker.start()
    try:
        while not status.stop.is_set():
            status.front_app = frontmost_bundle_id()  # also pumps the run loop for the overlay
            pointer = pyautogui.position()
            if at_corner(pointer.x, pointer.y, width, height):
                status.disable()
                break
            overlay.show(status.state)

        if status.disabled:
            overlay.show(status.state)
            log("disabled: mouse flicked into a screen corner")
            overlay.hold(DISABLED_HOLD)
    finally:
        status.stop.set()
        if hotkey_monitor is not None and NSEvent is not None:
            try:
                NSEvent.removeMonitor_(hotkey_monitor)
            except Exception:
                pass
        worker.join(timeout=5)
        overlay.close()
        print(summary(status, args.dry_run))

    if status.error is not None:
        raise status.error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--template",
        type=Path,
        default=ASSETS_DIR,
        help="PNG file, or a directory of PNGs to try (default: assets/)",
    )
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="min match score, 0-1")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between scans")
    p.add_argument("--cooldown", type=float, default=1.5, help="seconds to wait after a match")
    p.add_argument("--max-clicks", type=int, default=0, help="exit after N clicks (0 = unlimited)")
    p.add_argument("--dry-run", action="store_true", help="log matches but never click")
    p.add_argument(
        "--trust-all",
        action="store_true",
        help="click every match without an allow-list or OCR check (off by default)",
    )
    p.add_argument("--allow", type=Path, default=DEFAULT_ALLOW, help="regex allow list (default: allow.txt)")
    p.add_argument("--log", type=Path, default=DEFAULT_LOG, help="session log path (default: copilot-click.log)")
    p.add_argument("--pause-key", default=DEFAULT_PAUSE_KEY, help="global key that toggles pause (default: p)")
    p.add_argument(
        "--any-app",
        action="store_true",
        help="do not require VS Code, Insiders, or Cursor to be frontmost",
    )
    p.add_argument("--border", type=int, default=6, help="glow border width in points")
    p.add_argument("--no-overlay", action="store_true", help="do not draw the glowing border")
    return p.parse_args()


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
