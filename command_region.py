"""Where to read the command, and which recognized text is the command.

Pure functions (numpy/OpenCV only, no macOS APIs) so they can be tested with
synthetic images and synthetic OCR boxes.

The approval widget is assumed to look like this, with the command in a code
block whose background differs from the panel around it:

    ┌──────────────── chat panel ────────────────┐
    │ Run command in terminal                    │
    │ ┌─────────── code block ─────────────────┐ │
    │ │ git status                             │ │
    │ └────────────────────────────────────────┘ │
    │ [Allow Once]  [Skip]                       │
    └────────────────────────────────────────────┘

The crop is a generous region above the button. Inside it, the code block is
found by flood-filling its background colour from the text line closest to
the button. Every rule below fails closed: when the layout is not exactly
what is expected, no command is returned and nothing is clicked.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:  # exclusive
        return self.left + self.width

    @property
    def bottom(self) -> int:  # exclusive
        return self.top + self.height

    def contains(self, other: Rect, slack: int = 0) -> bool:
        return (
            other.left >= self.left - slack
            and other.top >= self.top - slack
            and other.right <= self.right + slack
            and other.bottom <= self.bottom + slack
        )

    def intersects(self, other: Rect) -> bool:
        return (
            other.left < self.right and self.left < other.right and other.top < self.bottom and self.top < other.bottom
        )

    def offset(self, dx: int, dy: int) -> Rect:
        return Rect(self.left + dx, self.top + dy, self.width, self.height)


@dataclass(frozen=True)
class TextLine:
    """One OCR observation, in crop pixel coordinates (origin top-left)."""

    text: str
    box: Rect
    confidence: float = 1.0


@dataclass(frozen=True)
class Extraction:
    command: str | None  # None means: do not click
    reason: str
    block: Rect | None = None


@dataclass(frozen=True)
class CropSettings:
    """Crop size in points; multiplied by the screen scale (2 on Retina)."""

    max_width: float = 900.0  # to the right of the button's left edge
    max_height: float = 600.0  # above the button
    left_pad: float = 40.0  # to the left of the button's left edge


def vision_box_to_rect(x: float, y: float, w: float, h: float, image_w: int, image_h: int) -> Rect:
    """Convert a Vision normalized bounding box (origin bottom-left) to pixel coords (origin top-left)."""
    left = int(np.floor(x * image_w))
    right = int(np.ceil((x + w) * image_w))
    top = int(np.floor((1.0 - (y + h)) * image_h))
    bottom = int(np.ceil((1.0 - y) * image_h))
    left, right = max(0, left), min(image_w, right)
    top, bottom = max(0, top), min(image_h, bottom)
    return Rect(left, top, max(0, right - left), max(0, bottom - top))


def command_crop(
    button: Rect,
    screen_w: int,
    screen_h: int,
    scale: float,
    *,
    margin: int = 0,
    bounds: Rect | None = None,
    settings: CropSettings | None = None,
) -> Rect | None:
    """The screen region (pixels) to read the command from, or None if there is no room.

    The region sits above the button, starts a little left of it and extends
    far to the right, clipped to the screen (minus the overlay margin) and,
    when known, to the editor window's bounds.
    """
    settings = settings or CropSettings()
    lo_x, lo_y, hi_x, hi_y = margin, margin, screen_w - margin, screen_h - margin
    if bounds is not None:
        lo_x, lo_y = max(lo_x, bounds.left), max(lo_y, bounds.top)
        hi_x, hi_y = min(hi_x, bounds.right), min(hi_y, bounds.bottom)
    left = max(lo_x, button.left - round(settings.left_pad * scale))
    right = min(hi_x, button.left + round(settings.max_width * scale))
    bottom = min(hi_y, button.top - 1)
    top = max(lo_y, button.top - round(settings.max_height * scale))
    if right - left < 1 or bottom - top < 1:
        return None
    return Rect(left, top, right - left, bottom - top)


def find_block(image: np.ndarray, anchor: Rect, tolerance: int = 6) -> Rect | None:
    """Bounding box of the uniform background region behind the anchor text line.

    The background colour is the per-channel median of the anchor box (glyphs
    cover a minority of it). The region is flood-filled from a pixel of that
    colour, so text inside the block leaves holes but does not split it.
    Returns None when no pixel of the anchor box has the background colour.
    """
    h, w = image.shape[:2]
    x0, y0 = max(0, anchor.left), max(0, anchor.top)
    x1, y1 = min(w, anchor.right), min(h, anchor.bottom)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = image[y0:y1, x0:x1].reshape(-1, image.shape[2]).astype(np.int16)
    background = np.median(patch, axis=0)
    distance = np.abs(patch - background).max(axis=1)
    index = int(np.argmin(distance))
    if distance[index] > tolerance:
        return None
    patch_w = x1 - x0
    seed = (x0 + index % patch_w, y0 + index // patch_w)
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    diff = (tolerance,) * 3
    cv2.floodFill(image.copy(), mask, seed, (0, 0, 0), diff, diff, flags)
    ys, xs = np.nonzero(mask[1:-1, 1:-1])
    if xs.size == 0:
        return None
    left, top = int(xs.min()), int(ys.min())
    return Rect(left, top, int(xs.max()) - left + 1, int(ys.max()) - top + 1)


def _rows(lines: Sequence[TextLine]) -> list[list[TextLine]]:
    """Group lines whose boxes overlap vertically by more than half a line."""
    rows: list[list[TextLine]] = []
    for line in sorted(lines, key=lambda item: (item.box.top, item.box.left)):
        if rows:
            last = rows[-1][-1].box
            overlap = min(last.bottom, line.box.bottom) - max(last.top, line.box.top)
            if overlap > 0.5 * min(last.height, line.box.height):
                rows[-1].append(line)
                continue
        rows.append([line])
    return rows


def extract_command(
    image: np.ndarray,
    lines: Sequence[TextLine],
    button: Rect,
    *,
    min_confidence: float = 0.5,
    edge: int = 2,
    tolerance: int = 6,
) -> Extraction:
    """Pick the command text out of the OCR lines of a crop, or refuse.

    image is the crop, lines are its OCR observations and button is the
    button's rectangle, all in crop coordinates (the button sits just below
    the crop). Refuses (command=None) when:

    * nothing was read, or the text nearest the button is not close to it;
    * that text does not sit on a distinct code-block background, or the
      block touches the top, left, right or bottom of the crop (it may be cut
      off, or the "block" is really the whole panel);
    * a recognized line straddles the block's edge, two observations share a
      row (layout is ambiguous), a line runs to the block's right edge or ends
      in an ellipsis (possibly clipped), or OCR confidence is low.
    """
    lines = [line for line in lines if line.text.strip() and line.box.width > 0 and line.box.height > 0]
    if not lines:
        return Extraction(None, "no text above the button")
    crop_h, crop_w = image.shape[:2]
    line_h = statistics.median(line.box.height for line in lines)
    max_gap = max(3 * button.height, 4 * line_h)

    anchor = max(lines, key=lambda line: line.box.bottom)
    if button.top - anchor.box.bottom > max_gap:
        return Extraction(None, "no text directly above the button")
    if anchor.box.right <= button.left or anchor.box.left >= button.right + button.width:
        return Extraction(None, "text above the button is not aligned with it")

    block = find_block(image, anchor.box, tolerance)
    if block is None:
        return Extraction(None, "no code block behind the command")
    if block.top <= edge or block.left <= edge or block.right >= crop_w - edge or block.bottom >= crop_h - edge:
        return Extraction(None, "code block touches the edge of the crop (possibly cut off)", block)
    if button.top - block.bottom > max_gap or block.right <= button.left or block.left >= button.right:
        return Extraction(None, "code block is not directly above the button", block)

    inside: list[TextLine] = []
    for line in lines:
        if block.contains(line.box, slack=1):
            inside.append(line)
        elif block.intersects(line.box):
            return Extraction(None, f"text straddles the code block edge: {line.text!r}", block)

    for line in inside:
        if line.confidence < min_confidence:
            return Extraction(None, f"low OCR confidence ({line.confidence:.2f}) for {line.text!r}", block)
        if line.box.right >= block.right - max(edge, round(line_h / 2)):
            return Extraction(None, "command runs to the code block's right edge (possibly clipped)", block)
        stripped = line.text.rstrip()
        if stripped.endswith(("\u2026", "...")):
            return Extraction(None, "command ends in an ellipsis (possibly truncated)", block)

    rows = _rows(inside)
    if any(len(row) > 1 for row in rows):
        return Extraction(None, "more than one text fragment on a line (ambiguous)", block)
    command = "\n".join(row[0].text.strip() for row in rows)
    if not command.strip():
        return Extraction(None, "no text in the code block", block)
    return Extraction(command, "ok", block)
