import cv2
import numpy as np

from autoclick import (
    DEFAULT_TEMPLATE,
    DEFAULT_THRESHOLD,
    Match,
    VSCodeGate,
    at_corner,
    find_button,
    pixels_to_points,
    summary,
)
from state import State, Status

SCREEN_W, SCREEN_H = 1512, 982


def make_button() -> np.ndarray:
    """Synthetic blue button with white text, similar to the Copilot one."""
    btn = np.full((32, 120, 3), (160, 110, 50), dtype=np.uint8)  # BGR steel blue
    cv2.putText(btn, "Allow Once", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return btn


def make_screen(button: np.ndarray, top: int, left: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    screen = rng.integers(20, 60, size=(600, 900, 3), dtype=np.uint8)
    h, w = button.shape[:2]
    screen[top : top + h, left : left + w] = button
    return screen


def test_finds_button_at_expected_centre():
    button = make_button()
    screen = make_screen(button, top=300, left=500)
    match = find_button(screen, button, DEFAULT_THRESHOLD)
    assert match is not None
    assert (match.x, match.y) == (500 + 60, 300 + 16)
    assert match.score > 0.99


def test_no_match_on_screen_without_button():
    blank = np.full((600, 900, 3), 40, dtype=np.uint8)
    assert find_button(blank, make_button(), DEFAULT_THRESHOLD) is None


def test_tolerates_slight_zoom():
    button = make_button()
    zoomed = cv2.resize(button, None, fx=1.05, fy=1.05, interpolation=cv2.INTER_LINEAR)
    screen = make_screen(zoomed, top=200, left=300)
    assert find_button(screen, button, DEFAULT_THRESHOLD) is not None


def test_template_larger_than_screen_is_skipped():
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    assert find_button(tiny, make_button(), DEFAULT_THRESHOLD) is None


def test_pixels_to_points_retina_halves_coordinates():
    assert pixels_to_points(Match(1.0, 2000, 1000), scale=2.0) == (1000.0, 500.0)


def test_pixels_to_points_non_retina_is_identity():
    assert pixels_to_points(Match(1.0, 640, 360), scale=1.0) == (640.0, 360.0)


def test_shipped_template_loads_and_matches_itself():
    template = cv2.imread(str(DEFAULT_TEMPLATE), cv2.IMREAD_COLOR)
    assert template is not None
    screen = make_screen(template, top=100, left=200)
    match = find_button(screen, template, DEFAULT_THRESHOLD)
    assert match is not None
    assert match.score > 0.99


def test_vscode_gate_logs_only_on_transitions(capsys):
    apps = [
        "com.apple.Terminal", "com.apple.Terminal",
        "com.microsoft.VSCode", "com.microsoft.VSCode",
        "com.apple.Terminal",
    ]
    gate = VSCodeGate()
    assert [gate.is_open(a) for a in apps] == [False, False, True, True, False]

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert "paused: frontmost app is com.apple.Terminal" in lines[0]
    assert "VS Code is frontmost, scanning" in lines[1]
    assert "paused: frontmost app is com.apple.Terminal" in lines[2]


def test_at_corner_detects_all_four_corners():
    w, h = SCREEN_W, SCREEN_H
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (1, 1), (w - 2, h - 2)]:
        assert at_corner(x, y, w, h), (x, y)


def test_at_corner_ignores_edges_centre_and_other_displays():
    w, h = SCREEN_W, SCREEN_H
    assert not at_corner(w // 2, 0, w, h)  # top edge, not a corner
    assert not at_corner(0, h // 2, w, h)  # left edge, not a corner
    assert not at_corner(w // 2, h // 2, w, h)
    assert not at_corner(-1, -1, w, h)  # display to the left/above
    assert not at_corner(w + 5, 0, w, h)  # display to the right
    assert not at_corner(10, 10, w, h)  # near but outside tolerance


def test_margin_hides_a_match_inside_the_border_band():
    button = make_button()
    screen = make_screen(button, top=2, left=2)  # button sits inside a 10px band
    assert find_button(screen, button, DEFAULT_THRESHOLD, margin=0) is not None
    assert find_button(screen, button, DEFAULT_THRESHOLD, margin=10) is None


def test_margin_keeps_full_screen_coordinates():
    button = make_button()
    screen = make_screen(button, top=300, left=500)
    match = find_button(screen, button, DEFAULT_THRESHOLD, margin=10)
    assert match is not None
    assert (match.x, match.y) == (560, 316)  # same centre as without a margin


def test_status_state_follows_scanning_and_disabled_wins():
    status = Status()
    assert status.state is State.WAITING
    status.scanning = True
    assert status.state is State.ACTIVE
    status.disable()
    assert status.stop.is_set()
    status.scanning = True  # late write from the scanner thread
    assert status.state is State.DISABLED


def test_summary_counts_clicks_with_correct_plural():
    status = Status()
    assert summary(status, dry_run=False) == "Auto-allowed 0 prompts this session."
    status.clicks = 1
    assert summary(status, dry_run=False) == "Auto-allowed 1 prompt this session."
    status.clicks = 7
    assert summary(status, dry_run=False) == "Auto-allowed 7 prompts this session."


def test_summary_in_dry_run_reports_detections_not_clicks():
    status = Status()
    status.detections = 3
    assert summary(status, dry_run=True) == "Dry run: found the button 3 time(s), clicked 0."
