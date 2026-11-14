"""State shared between the scanner thread and the main thread (corner watching and overlay)."""

import threading
from enum import Enum


class State(Enum):
    ACTIVE = "active"  # an allowed editor is frontmost and being scanned
    WAITING = "waiting"  # running, but another app is in front
    PAUSED = "paused"  # paused with the hotkey; the scanner does not click
    DISABLED = "disabled"  # stopped by a corner flick


class Status:
    def __init__(self, front_app: str | None = None, front_pid: int | None = None) -> None:
        self.stop = threading.Event()
        self.scanning = False
        self.paused = False  # written by the hotkey monitor, read by the scanner
        self.disabled = False
        self.clicks = 0  # live clicks made
        self.detections = 0  # buttons seen, including in --dry-run
        self.front_app = front_app  # written by the main thread, read by the scanner
        self.front_pid = front_pid  # process id of front_app, for window bounds
        self.error: BaseException | None = None

    @property
    def state(self) -> State:
        # Derived, so a late write from the scanner can never overwrite DISABLED.
        if self.disabled:
            return State.DISABLED
        if self.paused:
            return State.PAUSED
        return State.ACTIVE if self.scanning else State.WAITING

    def disable(self) -> None:
        self.disabled = True
        self.stop.set()
