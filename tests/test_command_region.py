import numpy as np
import pytest
from synthetic import BLOCK, PANEL, FakeOcr, crop_of, draw_text, line_colour, make_widget

from command_region import (
    CropSettings,
    Rect,
    TextLine,
    command_crop,
    extract_command,
    find_block,
    vision_box_to_rect,
)

SCREEN_H, SCREEN_W = 900, 1400


def default_crop(widget, **kwargs):
    crop = command_crop(widget.button, SCREEN_W, SCREEN_H, 1.0, margin=7, **kwargs)
    assert crop is not None
    return crop


def extract(widget, crop=None, **kwargs):
    crop = crop or default_crop(widget)
    image, button = crop_of(widget, crop)
    return extract_command(image, widget.ocr(image), button, **kwargs)


# --- Vision boxes -------------------------------------------------------------------


def test_vision_box_flips_the_y_axis():
    # Vision: normalized, origin bottom-left. A box at the very top of a 200x100 image:
    rect = vision_box_to_rect(0.25, 0.8, 0.5, 0.2, 200, 100)
    assert rect == Rect(50, 0, 100, 20)


def test_vision_box_is_clamped_to_the_image():
    rect = vision_box_to_rect(-0.1, -0.1, 1.3, 0.5, 100, 100)
    assert rect.left == 0 and rect.right == 100
    assert rect.bottom == 100


# --- the crop -----------------------------------------------------------------------


def test_crop_sits_above_the_button_and_is_generous():
    button = Rect(300, 580, 120, 32)
    crop = command_crop(button, SCREEN_W, SCREEN_H, 1.0)
    assert crop.bottom <= button.top  # never includes the button
    assert crop.left < button.left
    assert crop.right >= button.left + 700  # covers a wide command block
    assert crop.height >= 500


def test_crop_scales_with_retina():
    button = Rect(600, 1160, 240, 64)
    crop = command_crop(button, 2 * SCREEN_W, 2 * SCREEN_H, 2.0, settings=CropSettings(900, 300, 40))
    assert crop.top == 1160 - 600
    assert crop.left == 600 - 80
    assert crop.right == 600 + 1800


def test_crop_is_clipped_to_screen_margin_and_window():
    button = Rect(1300, 100, 80, 30)
    crop = command_crop(button, SCREEN_W, SCREEN_H, 1.0, margin=7)
    assert crop.top == 7
    assert crop.right == SCREEN_W - 7
    window = Rect(1200, 50, 190, 400)
    crop = command_crop(button, SCREEN_W, SCREEN_H, 1.0, margin=7, bounds=window)
    assert crop.left >= 1200 and crop.top == 50 and crop.right == 1390


def test_crop_is_none_without_room_above():
    assert command_crop(Rect(300, 3, 80, 30), SCREEN_W, SCREEN_H, 1.0, margin=7) is None


# --- extraction: the happy path ---------------------------------------------------


def test_single_line_command_is_extracted():
    widget = make_widget(["git status"])
    result = extract(widget)
    assert result.command == "git status", result.reason
    assert result.block is not None
    # The block found is the drawn code block, in crop coordinates.
    crop = default_crop(widget)
    assert result.block.offset(crop.left, crop.top) == widget.block


def test_title_above_the_block_is_not_part_of_the_command():
    widget = make_widget(["npm test"], title="Run command in terminal")
    assert extract(widget).command == "npm test"


def test_multi_line_command_keeps_every_line():
    widget = make_widget(["rm -rf ~", "git status"])
    result = extract(widget)
    assert result.command == "rm -rf ~\ngit status"


def test_blank_line_inside_the_block_does_not_hide_the_top_line():
    widget = make_widget(["rm -rf ~", "", "", "", "git status"])
    result = extract(widget)
    assert result.command is not None
    assert result.command.startswith("rm -rf ~")
    assert result.command.endswith("git status")


def test_indented_last_line_does_not_hide_the_first():
    widget = make_widget(["rm -rf ~ \\", "git status"], line_indents=[0, 40])
    assert extract(widget).command == "rm -rf ~ \\\ngit status"


# --- extraction: refusals ---------------------------------------------------------


def test_no_text_is_refused():
    widget = make_widget(["git status"])
    crop = default_crop(widget)
    image, button = crop_of(widget, crop)
    result = extract_command(image, [], button)
    assert result.command is None


def test_block_cut_off_at_the_top_of_the_crop_is_refused():
    widget = make_widget([f"line {i}" for i in range(12)], line_height=20, line_gap=8, title=None)
    crop = default_crop(widget, settings=CropSettings(max_height=150))
    result = extract(widget, crop)
    assert result.command is None
    assert "edge of the crop" in result.reason


def test_block_wider_than_the_crop_is_refused():
    widget = make_widget(["git status"], block_width=1050)
    crop = default_crop(widget)
    assert widget.block.right > crop.right
    result = extract(widget, crop)
    assert result.command is None
    assert "edge of the crop" in result.reason


def test_block_cut_off_at_the_left_is_refused():
    widget = make_widget(["git status"])
    crop = default_crop(widget, settings=CropSettings(left_pad=0))
    result = extract(widget, crop)
    assert result.command is None


def test_no_distinct_code_block_background_is_refused():
    widget = make_widget(["git status"], block_colour=PANEL)
    result = extract(widget)
    assert result.command is None
    assert "edge of the crop" in result.reason or "code block" in result.reason


def test_text_straddling_the_block_edge_is_refused():
    widget = make_widget(["git status"])
    # Something drawn across the block's top border.
    colour = line_colour(70)
    draw_text(widget.screen, Rect(widget.block.left + 300, widget.block.top - 10, 100, 20), colour)
    widget.ocr.texts[colour] = "; rm -rf ~"
    result = extract(widget)
    assert result.command is None
    assert "straddles" in result.reason


def test_two_fragments_on_one_row_are_refused():
    widget = make_widget(["git status"])
    colour = line_colour(71)
    row = Rect(widget.block.left + 400, widget.block.top + 12, 120, 20)
    draw_text(widget.screen, row, colour)
    widget.ocr.texts[colour] = "; rm -rf ~"
    result = extract(widget)
    assert result.command is None
    assert "more than one" in result.reason


def test_text_running_to_the_block_right_edge_is_refused():
    widget = make_widget(["git status --porcelain --untracked-files=all"], line_width=680)
    result = extract(widget)
    assert result.command is None
    assert "right edge" in result.reason


def test_ellipsis_is_refused():
    for ending in ("...", chr(0x2026)):
        widget = make_widget([f"git status{ending}"])
        result = extract(widget)
        assert result.command is None, ending
        assert "ellipsis" in result.reason


def test_low_confidence_is_refused():
    widget = make_widget(["git status"])
    widget.ocr.confidence = 0.3
    result = extract(widget)
    assert result.command is None
    assert "confidence" in result.reason


def test_text_far_above_the_button_is_refused():
    widget = make_widget(["git status"], button_top=580)
    # Move the button far below the block.
    button = widget.button
    widget.screen[button.top : button.bottom, button.left : button.right] = PANEL
    widget.button = Rect(button.left, 840, button.width, button.height)
    crop = command_crop(widget.button, SCREEN_W, SCREEN_H, 1.0, margin=7)
    result = extract(widget, crop)
    assert result.command is None
    assert "directly above" in result.reason


def test_text_not_aligned_with_the_button_is_refused():
    widget = make_widget(["git status"], block_left=300)
    lines = [TextLine("git status", Rect(10, 400, 100, 20))]
    image = np.full((560, 1000, 3), PANEL, dtype=np.uint8)
    result = extract_command(image, lines, Rect(600, 571, 120, 32))
    assert result.command is None
    assert widget  # keep the fixture referenced


# --- find_block -----------------------------------------------------------------------


def test_find_block_ignores_glyph_holes():
    image = np.full((200, 400, 3), PANEL, dtype=np.uint8)
    image[50:150, 40:360] = BLOCK
    draw_text(image, Rect(60, 70, 200, 20), (255, 255, 255))
    draw_text(image, Rect(60, 110, 250, 20), (255, 255, 255))
    assert find_block(image, Rect(60, 70, 200, 20)) == Rect(40, 50, 320, 100)


def test_find_block_with_solid_box_returns_none_or_tiny():
    image = np.full((50, 50, 3), 0, dtype=np.uint8)
    image[10:20, 10:20] = 255
    image[12, 12] = 128  # no dominant colour inside the anchor box
    block = find_block(image, Rect(10, 10, 10, 10))
    assert block is None or block.width <= 10


@pytest.mark.parametrize("colour", [(0, 0, 0), (255, 255, 255), (40, 40, 40)])
def test_fake_ocr_finds_drawn_lines(colour):
    image = np.full((60, 200, 3), colour, dtype=np.uint8)
    draw_text(image, Rect(10, 10, 100, 20), line_colour(0))
    lines = FakeOcr({line_colour(0): "x"})(image)
    assert lines and lines[0].box.left <= 12
