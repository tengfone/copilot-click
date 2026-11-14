"""Synthetic screens and a fake OCR engine for headless tests.

"Text" is drawn as thin vertical stripes in a colour unique to each line, so
the fake OCR can find each line's bounding box exactly while the background
colour still dominates the line's box (as it does for real glyphs).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from command_region import Rect, TextLine

PANEL = (30, 30, 30)
BLOCK = (58, 52, 48)


def make_button() -> np.ndarray:
    """Synthetic blue button with white text, similar to the Copilot one."""
    btn = np.full((32, 120, 3), (160, 110, 50), dtype=np.uint8)  # BGR steel blue
    cv2.putText(btn, "Allow Once", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return btn


def line_colour(index: int) -> tuple[int, int, int]:
    return (210, 200, 100 + index)


def draw_text(image: np.ndarray, rect: Rect, colour: tuple[int, int, int]) -> None:
    image[rect.top + 2 : rect.bottom - 2, rect.left + 2 : rect.right - 2 : 4] = colour


@dataclass
class FakeOcr:
    """Maps each line colour back to its text and reports the box Vision would."""

    texts: dict[tuple[int, int, int], str]
    confidence: float = 1.0
    calls: int = 0

    def __call__(self, image: np.ndarray) -> list[TextLine]:
        self.calls += 1
        lines: list[TextLine] = []
        h, w = image.shape[:2]
        for colour, text in self.texts.items():
            ys, xs = np.nonzero(np.all(image == np.array(colour, dtype=np.uint8), axis=2))
            if xs.size == 0:
                continue
            left, top = max(0, int(xs.min()) - 2), max(0, int(ys.min()) - 2)
            right, bottom = min(w, int(xs.max()) + 3), min(h, int(ys.max()) + 3)
            lines.append(TextLine(text, Rect(left, top, right - left, bottom - top), self.confidence))
        return lines


@dataclass
class Widget:
    """An approval widget: optional title, a code block with command lines, the button."""

    screen: np.ndarray
    button: Rect
    block: Rect
    ocr: FakeOcr


def make_widget(
    command_lines: list[str],
    *,
    size: tuple[int, int] = (900, 1400),
    block_left: int = 300,
    block_width: int = 700,
    button_top: int = 580,
    line_height: int = 20,
    line_gap: int = 8,
    line_width: int = 200,
    block_colour: tuple[int, int, int] = BLOCK,
    title: str | None = "Run command in terminal",
    line_indents: list[int] | None = None,
) -> Widget:
    h, w = size
    screen = np.full((h, w, 3), PANEL, dtype=np.uint8)
    pad = 12
    n = len(command_lines)
    block_height = pad * 2 + n * line_height + (n - 1) * line_gap
    block = Rect(block_left, button_top - 20 - block_height, block_width, block_height)
    screen[block.top : block.bottom, block.left : block.right] = block_colour
    texts: dict[tuple[int, int, int], str] = {}
    indents = line_indents or [0] * n
    for i, text in enumerate(command_lines):
        if not text:
            continue  # a blank line in the command
        top = block.top + pad + i * (line_height + line_gap)
        rect = Rect(block.left + 16 + indents[i], top, line_width, line_height)
        colour = line_colour(i)
        draw_text(screen, rect, colour)
        texts[colour] = text
    if title:
        colour = line_colour(50)
        draw_text(screen, Rect(block.left, block.top - 40, 260, line_height), colour)
        texts[colour] = title
    btn = make_button()
    button = Rect(block_left, button_top, btn.shape[1], btn.shape[0])
    screen[button.top : button.bottom, button.left : button.right] = btn
    return Widget(screen, button, block, FakeOcr(texts))


def crop_of(widget: Widget, crop: Rect) -> tuple[np.ndarray, Rect]:
    """The crop image and the button in crop coordinates."""
    image = widget.screen[crop.top : crop.bottom, crop.left : crop.right]
    return image, widget.button.offset(-crop.left, -crop.top)
