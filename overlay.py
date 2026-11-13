"""Click-through glowing border around the main screen showing the clicker's state (macOS, PyObjC)."""

import time

from AppKit import (
    NSApplication,
    NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered,
    NSBezierPath,
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSScreen,
    NSScreenSaverWindowLevel,
    NSView,
    NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces,
    NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowCollectionBehaviorStationary,
    NSWindowStyleMaskBorderless,
)
from Foundation import NSDate, NSInsetRect, NSMakeRect, NSRunLoop, NSString

from state import State

COLORS = {
    State.ACTIVE: (0.20, 0.85, 0.40),
    State.WAITING: (1.00, 0.72, 0.20),
    State.PAUSED: (0.62, 0.42, 0.95),
    State.DISABLED: (1.00, 0.25, 0.25),
}
DRY_RUN_ACTIVE = (0.25, 0.60, 1.00)  # blue, so a dry run is never mistaken for a live one
DISABLED_LABEL = "Auto-click disabled"
PAUSED_LABEL = "Paused"


def pump_runloop(seconds: float = 0.05) -> None:
    """Run the main run loop briefly. AppKit only draws and refreshes app state while it runs."""
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(seconds))


class GlowView(NSView):
    rgb = COLORS[State.WAITING]
    label = ""

    def drawRect_(self, rect):
        r, g, b = self.rgb
        bounds = self.bounds()
        # One-point rings that fade towards the middle of the screen read as a glow.
        for i in range(int(self.border)):
            alpha = 0.95 * (1 - i / self.border) ** 2
            NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, alpha).set()
            ring = NSBezierPath.bezierPathWithRect_(NSInsetRect(bounds, i + 0.5, i + 0.5))
            ring.setLineWidth_(1.0)
            ring.stroke()

        if self.label:
            self._draw_label(bounds)

    def _draw_label(self, bounds):
        r, g, b = self.rgb
        attrs = {
            NSFontAttributeName: NSFont.boldSystemFontOfSize_(18),
            NSForegroundColorAttributeName: NSColor.whiteColor(),
        }
        text = NSString.stringWithString_(self.label)
        size = text.sizeWithAttributes_(attrs)
        pad = 14.0
        pill = NSMakeRect(
            (bounds.size.width - size.width) / 2 - pad,
            80.0,  # above the bottom edge, clear of the menu bar and notch
            size.width + 2 * pad,
            size.height + pad,
        )
        NSColor.colorWithSRGBRed_green_blue_alpha_(r, g, b, 0.92).set()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(pill, 10, 10).fill()
        text.drawAtPoint_withAttributes_((pill.origin.x + pad, pill.origin.y + pad / 2), attrs)


class Overlay:
    def __init__(self, border_points: int, dry_run: bool) -> None:
        self._dry_run = dry_run
        self._shown: State | None = None

        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon
        app.finishLaunching()

        screens = NSScreen.screens()
        if not screens:
            raise RuntimeError("No display found to draw the overlay on (try --no-overlay)")
        frame = screens[0].frame()  # primary display, same space as pyautogui
        window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False
        )
        window.setOpaque_(False)
        window.setBackgroundColor_(NSColor.clearColor())
        window.setHasShadow_(False)
        window.setIgnoresMouseEvents_(True)  # clicks pass straight through
        window.setLevel_(NSScreenSaverWindowLevel)
        window.setCollectionBehavior_(
            NSWindowCollectionBehaviorCanJoinAllSpaces
            | NSWindowCollectionBehaviorStationary
            | NSWindowCollectionBehaviorFullScreenAuxiliary
        )

        view = GlowView.alloc().initWithFrame_(NSMakeRect(0, 0, frame.size.width, frame.size.height))
        view.border = float(border_points)
        window.setContentView_(view)
        window.orderFrontRegardless()
        self._window, self._view = window, view

    def show(self, state: State) -> None:
        if state is self._shown:
            return
        self._shown = state
        self._view.rgb = DRY_RUN_ACTIVE if (self._dry_run and state is State.ACTIVE) else COLORS[state]
        if state is State.DISABLED:
            self._view.label = DISABLED_LABEL
        elif state is State.PAUSED:
            self._view.label = PAUSED_LABEL
        else:
            self._view.label = ""
        self._view.setNeedsDisplay_(True)
        self._window.displayIfNeeded()

    def hold(self, seconds: float) -> None:
        """Keep the current frame on screen for a moment."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            pump_runloop()

    def close(self) -> None:
        self._window.orderOut_(None)
