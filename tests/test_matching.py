import re

import cv2
import numpy as np

from autoclick import (
    ASSETS_DIR,
    DEFAULT_TEMPLATE,
    DEFAULT_THRESHOLD,
    CURSOR_BUNDLE_ID,
    Match,
    VSCodeGate,
    at_corner,
    command_allowed,
    crop_command_region,
    decide_click,
    find_button,
    format_detection,
    install_pause_monitor,
    load_allow_patterns,
    load_templates,
    next_paused,
    pixels_to_points,
    recognize_text,
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


def test_find_button_returns_the_best_named_template():
    button = make_button()
    faded = cv2.convertScaleAbs(button, alpha=0.55, beta=30)
    screen = make_screen(button, top=300, left=500)
    match = find_button(
        screen,
        [("faded.png", faded), ("allow.png", button)],
        DEFAULT_THRESHOLD,
    )
    assert match is not None
    assert match.template == "allow.png"
    assert (match.x, match.y) == (500 + 60, 300 + 16)
    assert match.score > 0.99


def test_find_button_rejects_every_template_below_threshold():
    screen = np.full((600, 900, 3), 40, dtype=np.uint8)
    other = make_button()
    cv2.rectangle(other, (4, 4), (116, 28), (0, 0, 180), -1)
    cv2.putText(other, "Deny", (24, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    assert find_button(screen, [("a.png", make_button()), ("b.png", other)], DEFAULT_THRESHOLD) is None


def test_shipped_assets_include_a_distinct_second_template():
    loaded = load_templates(ASSETS_DIR)
    names = [name for name, _ in loaded]
    assert "allow_once_ss.png" in names
    assert len(loaded) >= 2
    images = {name: image for name, image in loaded}
    original = images["allow_once_ss.png"]
    extras = [image for name, image in loaded if name != "allow_once_ss.png"]
    assert extras
    assert all(not np.array_equal(original, extra) for extra in extras)
    # The extra template must match a screen that contains itself.
    extra_name = next(name for name, _ in loaded if name != "allow_once_ss.png")
    extra = images[extra_name]
    screen = make_screen(extra, top=40, left=30)
    match = find_button(screen, loaded, DEFAULT_THRESHOLD)
    assert match is not None
    assert match.template == extra_name


def test_command_allowed_empty_list_denies():
    assert command_allowed("git status", []) is False


def test_command_allowed_matching_regex_allows():
    assert command_allowed("git status", [r"^git status$"]) is True


def test_command_allowed_non_matching_line_denies():
    assert command_allowed("rm -rf /tmp", [r"^git status$"]) is False


def test_command_allowed_ignores_comments_and_blank_lines():
    patterns = ["", "   ", "# git status", "git diff"]
    assert command_allowed("git diff --stat", patterns) is True
    assert command_allowed("git status", ["# git status", "", "   "]) is False


def test_command_allowed_skips_invalid_regex(capsys):
    assert command_allowed("git status", ["[unclosed", r"^git status$"]) is True
    assert "invalid regex" in capsys.readouterr().out
    assert command_allowed("git status", ["[unclosed"]) is False


def test_load_allow_patterns_missing_file_is_empty(tmp_path):
    assert load_allow_patterns(tmp_path / "allow.txt") == []


def test_load_allow_patterns_file_skips_comments_blanks_and_bad_regex(tmp_path, capsys):
    path = tmp_path / "allow.txt"
    path.write_text("# note\n\n^git status$\n[bad\n", encoding="utf-8")
    patterns = load_allow_patterns(path)
    assert command_allowed("git status", patterns) is True
    assert command_allowed("git diff", patterns) is False
    assert "invalid regex" in capsys.readouterr().out


def test_decide_click_refuses_without_a_matching_pattern():
    git = [re.compile(r"^git status$")]
    assert decide_click(paused=False, dry_run=False, trust_all=False, ocr_text=None, patterns=[]) == (
        False,
        "skipped: allow list empty",
    )
    assert decide_click(paused=False, dry_run=False, trust_all=False, ocr_text=None, patterns=git) == (
        False,
        "skipped: no text",
    )
    assert decide_click(paused=False, dry_run=False, trust_all=False, ocr_text="rm -rf /", patterns=git) == (
        False,
        "skipped: not allowed",
    )
    assert decide_click(paused=False, dry_run=False, trust_all=False, ocr_text="git status", patterns=git) == (
        True,
        "clicked",
    )
    assert decide_click(paused=False, dry_run=False, trust_all=True, ocr_text=None, patterns=[]) == (
        True,
        "clicked",
    )
    assert decide_click(paused=True, dry_run=False, trust_all=True, ocr_text="git status", patterns=git) == (
        False,
        "skipped: paused",
    )
    assert decide_click(paused=False, dry_run=True, trust_all=True, ocr_text="git status", patterns=git) == (
        False,
        "skipped: dry run",
    )


def test_shipped_templates_do_not_match_each_other():
    loaded = load_templates(ASSETS_DIR)
    for name, image in loaded:
        others = [(other_name, other) for other_name, other in loaded if other_name != name]
        screen = make_screen(image, top=40, left=30)
        assert find_button(screen, others, DEFAULT_THRESHOLD) is None, name


def test_pause_monitor_stays_off_under_pytest(capsys):
    status = Status()
    assert install_pause_monitor(status, "p") is None
    assert capsys.readouterr().out == ""


def test_pause_hotkey_toggles_only_its_key():
    assert next_paused(False, "p", "p") is True
    assert next_paused(True, "p", "p") is False
    assert next_paused(False, "P", "p") is True
    assert next_paused(False, "x", "p") is False
    assert next_paused(True, "x", "p") is True


def test_paused_status_shows_paused_and_disabled_still_wins():
    status = Status()
    status.scanning = True
    status.paused = True
    assert status.state is State.PAUSED
    status.paused = False
    assert status.state is State.ACTIVE
    status.paused = True
    status.disable()
    assert status.stop.is_set()
    assert status.state is State.DISABLED


def test_gate_allows_cursor_and_logs_its_family(capsys):
    gate = VSCodeGate()
    assert gate.is_open(CURSOR_BUNDLE_ID) is True
    assert gate.is_open(CURSOR_BUNDLE_ID) is True  # no second log while it stays frontmost
    assert capsys.readouterr().out.count("Cursor is frontmost, scanning") == 1


def test_gate_names_vscode_insiders(capsys):
    gate = VSCodeGate()
    assert gate.is_open("com.microsoft.VSCodeInsiders") is True
    assert "VS Code Insiders is frontmost, scanning" in capsys.readouterr().out


def test_recognize_text_without_vision_returns_none():
    blank = np.zeros((24, 48, 3), dtype=np.uint8)
    assert recognize_text(blank) is None


def test_command_crop_is_above_the_button():
    screen = np.zeros((200, 400, 3), dtype=np.uint8)
    match = Match(0.9, 200, 120, "btn.png", width=80, height=30)
    button_top = match.y - match.height // 2
    screen[button_top : button_top + match.height, match.x - 40 : match.x + 40] = 9
    screen[:button_top, :] = 3
    crop = crop_command_region(screen, match)
    assert crop.size > 0
    assert int(crop.max()) == 3


def test_format_detection_records_score_text_and_outcome():
    line = format_detection("2026-11-13 14:40:01", "allow_once_ss.png", 0.91234, None, "skipped: no text")
    assert line == "2026-11-13 14:40:01  template=allow_once_ss.png  score=0.912  text=(no text)  skipped: no text"
    clicked = format_detection("2026-11-13 14:40:02", "allow_once_light.png", 0.88, "git\nstatus", "clicked")
    assert "text=git status" in clicked
    assert clicked.endswith("clicked")
