"""Auto-click the Copilot "Allow Once" button in VS Code (macOS).

Takes periodic screenshots, template-matches the button with OpenCV, and clicks
it with pyautogui. Only clicks while VS Code is the frontmost app.

A glowing border around the main screen shows the state (see README). Stop for
good with Ctrl+C or by flicking the mouse into a corner of the main screen.
"""

import argparse
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyautogui
from AppKit import NSWorkspace

from overlay import Overlay, pump_runloop
from state import State, Status

DEFAULT_TEMPLATE = Path(__file__).parent / "assets" / "allow_once_ss.png"
VSCODE_BUNDLE_IDS = frozenset({"com.microsoft.VSCode", "com.microsoft.VSCodeInsiders"})
DEFAULT_THRESHOLD = 0.85
SCALES = (1.0, 0.95, 1.05, 0.9, 1.1)
CORNER_TOLERANCE = 2  # points from the screen edge on both axes
DISABLED_HOLD = 1.0  # seconds the red "disabled" frame stays up before exiting


@dataclass(frozen=True)
class Match:
    score: float
    x: int  # centre of the match, in screenshot pixels
    y: int


def find_button(
    screen_bgr: np.ndarray,
    template_bgr: np.ndarray,
    threshold: float,
    margin: int = 0,
) -> Match | None:
    """Return the best template match at or above threshold, or None.

    margin crops that many pixels off every screen edge before matching, so the
    overlay border can never be mistaken for (or hide) the button.
    """
    if margin:
        screen_bgr = screen_bgr[margin:-margin, margin:-margin]
    screen = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    best: Match | None = None
    for scale in SCALES:
        scaled = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        h, w = scaled.shape
        if h > screen.shape[0] or w > screen.shape[1]:
            continue
        result = cv2.matchTemplate(screen, scaled, cv2.TM_CCOEFF_NORMED)
        _, score, _, (left, top) = cv2.minMaxLoc(result)
        if score >= threshold and (best is None or score > best.score):
            best = Match(float(score), left + w // 2 + margin, top + h // 2 + margin)
    return best


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


def frontmost_bundle_id() -> str | None:
    # NSWorkspace only refreshes frontmostApplication while the run loop runs,
    # which a plain Python polling loop never does, so pump it before reading.
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


class VSCodeGate:
    """Tracks whether VS Code is frontmost, logging only when that changes."""

    def __init__(self) -> None:
        self._paused_for: str | None = None

    def is_open(self, front: str | None) -> bool:
        if front not in VSCODE_BUNDLE_IDS:
            if front != self._paused_for:
                log(f"paused: frontmost app is {front}, not VS Code")
                self._paused_for = front
            return False
        if self._paused_for is not None:
            log("VS Code is frontmost, scanning")
            self._paused_for = None
        return True


class NullOverlay:
    """Stand-in used with --no-overlay."""

    def show(self, state: State) -> None:
        pass

    def hold(self, seconds: float) -> None:
        time.sleep(seconds)

    def close(self) -> None:
        pass


def scan_loop(args: argparse.Namespace, template: np.ndarray, status: Status) -> None:
    """Screenshot, match and click until status.stop is set. Runs on a worker thread."""
    gate = VSCodeGate()
    while not status.stop.wait(args.interval):
        status.scanning = args.any_app or gate.is_open(status.front_app)
        if not status.scanning:
            continue

        screen = capture_bgr()
        scale = screen.shape[1] / pyautogui.size().width
        margin = math.ceil(args.border * scale) + 1  # keep the overlay border out of the match
        match = find_button(screen, template, args.threshold, margin=margin)
        if match is None:
            continue

        x, y = pixels_to_points(match, scale)
        found = f"match score={match.score:.3f} at ({x:.0f}, {y:.0f})"
        status.detections += 1

        if args.dry_run:
            log(f"{found} - dry run, not clicking")
        else:
            if status.stop.is_set():  # a corner flick landed while we were matching
                return
            click_at(x, y)
            status.clicks += 1
            log(f"{found} - clicked ({status.clicks})")
            if args.max_clicks and status.clicks >= args.max_clicks:
                print("Reached --max-clicks, exiting.")
                status.stop.set()
                return

        status.stop.wait(args.cooldown)


def scan_thread(args: argparse.Namespace, template: np.ndarray, status: Status) -> None:
    try:
        scan_loop(args, template, status)
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
    template = cv2.imread(str(args.template), cv2.IMREAD_COLOR)
    if template is None:
        raise SystemExit(f"Could not read template image: {args.template}")

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    width, height = pyautogui.size()

    # The overlay and the frontmost-app lookup both need the main thread's run loop.
    overlay = NullOverlay() if args.no_overlay else Overlay(args.border, args.dry_run)

    status = Status(front_app=frontmost_bundle_id())
    worker = threading.Thread(target=scan_thread, args=(args, template, status), daemon=True)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"[{mode}] watching for {args.template.name} (threshold {args.threshold}). Ctrl+C or a corner flick to stop.")
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
        worker.join(timeout=5)
        overlay.close()
        print(summary(status, args.dry_run))

    if status.error is not None:
        raise status.error


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="button image to look for")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="min match score, 0-1")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between scans")
    p.add_argument("--cooldown", type=float, default=1.5, help="seconds to wait after a match")
    p.add_argument("--max-clicks", type=int, default=0, help="exit after N clicks (0 = unlimited)")
    p.add_argument("--dry-run", action="store_true", help="log matches but never click")
    p.add_argument("--any-app", action="store_true", help="do not require VS Code to be frontmost")
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
