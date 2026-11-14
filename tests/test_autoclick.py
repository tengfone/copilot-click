import logging
import os
import stat
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic import PANEL, FakeOcr, draw_text, line_colour, make_button, make_widget

from allowlist import AllowList, parse_allow_list
from autoclick import (
    ASSETS_DIR,
    CURSOR_BUNDLE_ID,
    DEFAULT_TEMPLATE,
    DEFAULT_THRESHOLD,
    MOD_COMMAND,
    MOD_CONTROL,
    MOD_OPTION,
    MOD_SHIFT,
    EditorGate,
    Looker,
    Match,
    Observation,
    Policy,
    append_detection,
    at_corner,
    build_parser,
    confirm_click,
    decide_click,
    default_allow_path,
    default_log_path,
    find_button,
    format_detection,
    handle_detection,
    hotkey_label,
    is_pause_hotkey,
    load_templates,
    pick_window,
    pixels_to_points,
    redact,
    summary,
)
from command_region import Rect
from state import State, Status

SCREEN_W, SCREEN_H = 1512, 982
VSCODE = "com.microsoft.VSCode"
REPO = Path(__file__).parent.parent


def make_screen(button: np.ndarray, top: int, left: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    screen = rng.integers(20, 60, size=(600, 900, 3), dtype=np.uint8)
    h, w = button.shape[:2]
    screen[top : top + h, left : left + w] = button
    return screen


# --- template matching -------------------------------------------------------------


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


def test_pixels_to_points():
    assert pixels_to_points(Match(1.0, 2000, 1000), scale=2.0) == (1000.0, 500.0)
    assert pixels_to_points(Match(1.0, 640, 360), scale=1.0) == (640.0, 360.0)


def test_shipped_template_loads_and_matches_itself():
    template = cv2.imread(str(DEFAULT_TEMPLATE), cv2.IMREAD_COLOR)
    assert template is not None
    screen = make_screen(template, top=100, left=200)
    match = find_button(screen, template, DEFAULT_THRESHOLD)
    assert match is not None
    assert match.score > 0.99


def test_only_real_templates_are_shipped():
    names = [name for name, _ in load_templates(ASSETS_DIR)]
    assert names == ["allow_once_ss.png"]


def test_margin_hides_a_match_inside_the_border_band():
    button = make_button()
    screen = make_screen(button, top=2, left=2)
    assert find_button(screen, button, DEFAULT_THRESHOLD, margin=0) is not None
    assert find_button(screen, button, DEFAULT_THRESHOLD, margin=10) is None


def test_margin_keeps_full_screen_coordinates():
    button = make_button()
    screen = make_screen(button, top=300, left=500)
    match = find_button(screen, button, DEFAULT_THRESHOLD, margin=10)
    assert match is not None
    assert (match.x, match.y) == (560, 316)


def test_find_button_returns_the_best_named_template():
    button = make_button()
    faded = cv2.convertScaleAbs(button, alpha=0.55, beta=30)
    screen = make_screen(button, top=300, left=500)
    match = find_button(screen, [("faded.png", faded), ("allow.png", button)], DEFAULT_THRESHOLD)
    assert match is not None
    assert match.template == "allow.png"
    assert match.rect() == Rect(500, 300, 120, 32)


def test_at_corner():
    w, h = SCREEN_W, SCREEN_H
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (1, 1), (w - 2, h - 2)]:
        assert at_corner(x, y, w, h), (x, y)
    for x, y in [(w // 2, 0), (0, h // 2), (w // 2, h // 2), (-1, -1), (w + 5, 0), (10, 10)]:
        assert not at_corner(x, y, w, h), (x, y)


# --- state and gate ------------------------------------------------------------------


def test_status_state_and_disabled_wins():
    status = Status()
    assert status.state is State.WAITING
    status.scanning = True
    assert status.state is State.ACTIVE
    status.paused = True
    assert status.state is State.PAUSED
    status.disable()
    assert status.stop.is_set()
    assert status.state is State.DISABLED


def test_summary():
    status = Status()
    assert summary(status, dry_run=False) == "Auto-allowed 0 prompts this session."
    status.clicks = 1
    assert summary(status, dry_run=False) == "Auto-allowed 1 prompt this session."
    status.detections = 3
    assert summary(status, dry_run=True) == "Dry run: found the button 3 time(s), clicked 0."


def test_gate_logs_only_on_transitions(caplog):
    caplog.set_level(logging.INFO, logger="copilot-click")
    apps = ["com.apple.Terminal", "com.apple.Terminal", VSCODE, VSCODE, "com.apple.Terminal"]
    gate = EditorGate()
    assert [gate.is_open(a) for a in apps] == [False, False, True, True, False]
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 3
    assert "VS Code is frontmost, scanning" in messages[1]


def test_gate_marks_cursor_untested(caplog):
    caplog.set_level(logging.INFO, logger="copilot-click")
    assert EditorGate().is_open(CURSOR_BUNDLE_ID) is True
    assert "untested" in caplog.text


# --- pause hotkey ------------------------------------------------------------------------

COMBO = MOD_CONTROL | MOD_OPTION | MOD_COMMAND
CAPS_LOCK = 1 << 16


def test_pause_hotkey_needs_exactly_ctrl_option_cmd():
    assert is_pause_hotkey(COMBO, "p") is True
    assert is_pause_hotkey(COMBO, "P") is True
    assert is_pause_hotkey(COMBO | CAPS_LOCK, "p") is True  # caps lock is ignored


@pytest.mark.parametrize(
    "flags",
    [0, MOD_COMMAND, MOD_CONTROL, MOD_CONTROL | MOD_COMMAND, MOD_OPTION | MOD_COMMAND, COMBO | MOD_SHIFT],
)
def test_pause_hotkey_ignores_other_modifier_sets(flags):
    assert is_pause_hotkey(flags, "p") is False


def test_pause_hotkey_ignores_other_keys_and_custom_key():
    assert is_pause_hotkey(COMBO, "x") is False
    assert is_pause_hotkey(COMBO, "k", "k") is True
    assert is_pause_hotkey(COMBO, "p", "k") is False
    assert hotkey_label("p") == "Ctrl+Option+Cmd+P"


# --- decisions -----------------------------------------------------------------------------

GIT = parse_allow_list("git status")


def obs(command="git status", x=500, y=300, front=VSCODE, reason="ok"):
    return Observation(Match(0.95, x, y, "allow_once_ss.png", 120, 32), command, reason, front)


def test_decide_click_outcomes():
    live = Policy()
    assert decide_click(paused=True, policy=live, observation=obs(), allow=GIT) == (False, "skipped: paused")
    assert decide_click(paused=False, policy=live, observation=obs(), allow=GIT) == (True, "clicked")
    assert decide_click(paused=False, policy=live, observation=obs(), allow=AllowList()) == (
        False,
        "skipped: allow list empty",
    )
    assert decide_click(paused=False, policy=live, observation=obs(None, reason="no code block"), allow=GIT) == (
        False,
        "skipped: no code block",
    )
    refused, outcome = decide_click(paused=False, policy=live, observation=obs("git status; rm -rf ~"), allow=GIT)
    assert refused is False and "';'" in outcome
    assert decide_click(paused=False, policy=Policy(dry_run=True), observation=obs(), allow=GIT) == (
        False,
        "dry run: would click",
    )
    dry_refused = decide_click(paused=False, policy=Policy(dry_run=True), observation=obs("rm -rf ~"), allow=GIT)
    assert dry_refused == (False, "skipped: not allowed")
    assert decide_click(paused=False, policy=Policy(trust_all=True), observation=obs(None), allow=AllowList()) == (
        True,
        "clicked",
    )


def test_confirm_click_requires_same_button_command_and_app():
    p = Policy()
    first = obs()
    assert confirm_click(first, obs(), VSCODE, p) == (True, "confirmed")
    assert confirm_click(first, obs(x=502, y=298), VSCODE, p)[0] is True  # within tolerance
    assert confirm_click(first, None, VSCODE, p)[1] == "button gone on re-check"
    assert "moved" in confirm_click(first, obs(x=520), VSCODE, p)[1]
    assert "command changed" in confirm_click(first, obs("git status; rm -rf ~"), VSCODE, p)[1]
    assert "command changed" in confirm_click(first, obs(None), VSCODE, p)[1]
    assert "frontmost app" in confirm_click(first, obs(front="com.apple.Terminal"), VSCODE, p)[1]
    assert "frontmost app" in confirm_click(first, obs(), "com.apple.Terminal", p)[1]
    assert "frontmost app" in confirm_click(first, obs(), None, p)[1]


def test_confirm_click_with_any_app_still_requires_the_same_app():
    p = Policy(any_app=True)
    first = obs(front="com.apple.Terminal")
    assert confirm_click(first, obs(front="com.apple.Terminal"), "com.apple.Terminal", p)[0] is True
    assert confirm_click(first, obs(front="com.apple.Terminal"), "com.apple.Safari", p)[0] is False


class Clicks:
    def __init__(self):
        self.seen = []

    def __call__(self, observation):
        self.seen.append(observation)


def run_flow(first, second, *, status=None, policy=None, allow=GIT, on_look=None):
    status = status or Status(front_app=VSCODE)
    clicks = Clicks()

    def look_again():
        if on_look:
            on_look(status)
        return second

    outcome = handle_detection(
        first,
        status=status,
        policy=policy or Policy(confirm_delay=0),
        allow=allow,
        look_again=look_again,
        click=clicks,
    )
    return outcome, clicks.seen


def test_flow_clicks_once_when_the_recheck_agrees():
    outcome, clicks = run_flow(obs(), obs(x=501))
    assert outcome == "clicked"
    assert len(clicks) == 1
    assert clicks[0].match.x == 501  # clicks where the button is now


def test_flow_does_not_click_when_the_command_changes_before_the_click():
    outcome, clicks = run_flow(obs("git status"), obs("git status && rm -rf ~"))
    assert outcome == "skipped: command changed on re-check"
    assert clicks == []


def test_flow_does_not_click_when_the_button_moves_or_vanishes():
    assert run_flow(obs(), obs(y=360))[1] == []
    assert run_flow(obs(), None)[1] == []


def test_flow_does_not_click_when_the_app_changes():
    def switch(status):
        status.front_app = "com.apple.Terminal"

    outcome, clicks = run_flow(obs(), obs(), on_look=switch)
    assert "frontmost app" in outcome
    assert clicks == []


def test_flow_does_not_click_when_paused_or_stopped_during_the_recheck():
    def pause(status):
        status.paused = True

    def stop(status):
        status.stop.set()

    assert run_flow(obs(), obs(), on_look=pause) == ("skipped: paused", [])
    assert run_flow(obs(), obs(), on_look=stop) == ("skipped: stopping", [])


def test_flow_never_rechecks_or_clicks_a_refused_command():
    looked = []
    outcome, clicks = run_flow(obs("rm -rf ~"), obs("git status"), on_look=looked.append)
    assert outcome == "skipped: not allowed"
    assert clicks == [] and looked == []


def test_flow_dry_run_never_clicks():
    outcome, clicks = run_flow(obs(), obs(), policy=Policy(dry_run=True, confirm_delay=0))
    assert outcome == "dry run: would click"
    assert clicks == []


def test_flow_trust_all_skips_text_but_still_rechecks_position():
    policy = Policy(trust_all=True, confirm_delay=0)
    assert run_flow(obs(None), obs(None), policy=policy, allow=AllowList())[0] == "clicked"
    assert run_flow(obs(None), obs(None, x=600), policy=policy, allow=AllowList())[1] == []


# --- Looker end to end with a synthetic screen and fake OCR --------------------------------


def make_looker(screens, ocr, front=VSCODE, **kwargs):
    frames = iter(screens)
    last = {}

    def capture():
        last["screen"] = next(frames, last.get("screen"))
        return last["screen"]

    return Looker(
        templates=[("btn.png", make_button())],
        capture=capture,
        points_width=lambda: screens[0].shape[1],
        front_app=lambda: front,
        ocr=ocr,
        **kwargs,
    )


def test_looker_reads_the_command_above_the_button():
    widget = make_widget(["git status"])
    looker = make_looker([widget.screen], widget.ocr)
    seen = looker.look()
    assert seen is not None
    assert seen.command == "git status", seen.reason
    assert (seen.match.x, seen.match.y) == (widget.button.left + 60, widget.button.top + 16)
    assert seen.front_app == VSCODE


def test_looker_refuses_a_multi_line_command_that_hides_rm():
    widget = make_widget(["rm -rf ~", "", "git status"])
    seen = make_looker([widget.screen], widget.ocr).look()
    assert seen is not None
    assert GIT.evaluate(seen.command).allowed is False


def test_looker_without_vision_never_has_a_command():
    widget = make_widget(["git status"])
    seen = make_looker([widget.screen], lambda image: None).look()
    assert seen.command is None
    assert seen.reason == "text recognition unavailable"


def test_looker_trust_all_does_not_run_ocr():
    widget = make_widget(["git status"])
    seen = make_looker([widget.screen], widget.ocr, read_text=False).look()
    assert seen.command is None
    assert widget.ocr.calls == 0


def test_looker_returns_none_without_a_button():
    blank = np.full((900, 1400, 3), PANEL, dtype=np.uint8)
    assert make_looker([blank], FakeOcr({})).look() is None


def test_end_to_end_swap_between_look_and_click_is_not_clicked():
    before = make_widget(["git status"])
    after = before.screen.copy()
    # The command in the block changes between the decision and the click.
    colour = line_colour(9)
    draw_text(after, Rect(before.block.left + 240, before.block.top + 12, 150, 20), colour)
    ocr = FakeOcr({**before.ocr.texts, colour: "&& rm -rf ~"})
    looker = make_looker([before.screen, after], ocr)
    first = looker.look()
    assert first.command == "git status"
    status = Status(front_app=VSCODE)
    clicked = []
    outcome = handle_detection(
        first,
        status=status,
        policy=Policy(confirm_delay=0),
        allow=GIT,
        look_again=looker.look,
        click=clicked.append,
    )
    assert outcome.startswith("skipped")
    assert clicked == []


def test_end_to_end_unchanged_screen_is_clicked_once():
    widget = make_widget(["git status"])
    looker = make_looker([widget.screen, widget.screen], widget.ocr)
    clicked = []
    outcome = handle_detection(
        looker.look(),
        status=Status(front_app=VSCODE),
        policy=Policy(confirm_delay=0),
        allow=GIT,
        look_again=looker.look,
        click=clicked.append,
    )
    assert outcome == "clicked"
    assert len(clicked) == 1


def test_window_bounds_limit_the_crop():
    widget = make_widget(["git status"], block_width=700)
    # A window narrower than the code block: the block now touches the crop's edge.
    looker = make_looker(
        [widget.screen], widget.ocr, window_bounds=lambda x, y: Rect(0, 0, widget.block.left + 400, 900)
    )
    seen = looker.look()
    assert seen.command is None
    assert "edge" in seen.reason


def test_pick_window_chooses_the_front_window_of_the_app_under_the_point():
    windows = [
        {"kCGWindowOwnerPID": 9, "kCGWindowLayer": 0, "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 50, "Height": 50}},
        {
            "kCGWindowOwnerPID": 7,
            "kCGWindowLayer": 25,
            "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 999, "Height": 999},
        },
        {
            "kCGWindowOwnerPID": 7,
            "kCGWindowLayer": 0,
            "kCGWindowBounds": {"X": 10, "Y": 20, "Width": 300, "Height": 200},
        },
        {"kCGWindowOwnerPID": 7, "kCGWindowLayer": 0, "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 999, "Height": 999}},
    ]
    assert pick_window(windows, 7, 100, 100) == Rect(10, 20, 300, 200)
    assert pick_window(windows, 7, 500, 500) == Rect(0, 0, 999, 999)
    assert pick_window(windows, 8, 100, 100) is None
    assert pick_window(windows, None, 100, 100) is None
    assert pick_window([{"kCGWindowOwnerPID": 7, "kCGWindowBounds": {}}], 7, 1, 1) is None


# --- paths, logging and redaction ---------------------------------------------------------


def test_default_paths(tmp_path):
    assert default_allow_path({}, tmp_path) == tmp_path / ".config" / "copilot-click" / "allow.txt"
    assert default_allow_path({"XDG_CONFIG_HOME": "/x"}, tmp_path) == Path("/x/copilot-click/allow.txt")
    assert default_allow_path({"XDG_CONFIG_HOME": "rel"}, tmp_path) == tmp_path / ".config/copilot-click/allow.txt"
    assert default_log_path("darwin", {}, tmp_path) == tmp_path / "Library/Logs/copilot-click/copilot-click.log"
    assert default_log_path("linux", {}, tmp_path) == tmp_path / ".local/state/copilot-click/copilot-click.log"
    assert default_log_path("linux", {"XDG_STATE_HOME": "/s"}, tmp_path) == Path("/s/copilot-click/copilot-click.log")


def test_defaults_are_not_relative_to_the_working_directory():
    args = build_parser().parse_args([])
    assert args.allow_file.is_absolute()
    assert args.log_file.is_absolute()
    assert args.pause_key == "p"


def test_cli_flags_and_aliases(tmp_path):
    args = build_parser().parse_args(["--allow-file", str(tmp_path / "a"), "--log", str(tmp_path / "l")])
    assert args.allow_file == tmp_path / "a"
    assert args.log_file == tmp_path / "l"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--pause-key", "pp"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--threshold", "2"])


def test_detection_log_is_private(tmp_path):
    path = tmp_path / "logs" / "copilot-click.log"
    append_detection(path, "t.png", 0.9, "git status", "clicked")
    append_detection(path, "t.png", 0.9, None, "skipped: no text")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[0].endswith("clicked")


def test_existing_open_log_is_tightened(tmp_path):
    path = tmp_path / "copilot-click.log"
    path.write_text("old\n", encoding="utf-8")
    os.chmod(path, 0o644)
    append_detection(path, "t.png", 0.9, "x", "clicked")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    ("secret", "kept"),
    [
        ("ghp_" + "a" * 36, ""),
        ("github_pat_" + "B" * 30, ""),
        ("sk-" + "c" * 32, ""),
        ("AKIA" + "D" * 16, ""),
        ("xoxb-" + "1" * 20, ""),
        ("--password=hunter2", "--password="),
        ("--token hunter2", "--token "),
        ("API_KEY=hunter2", "API_KEY="),
        ("Authorization: Bearer abcdefghijkl", "Bearer "),
        ("https://user:hunter2@example.com/repo.git", "https://user:"),
    ],
)
def test_redact_masks_secrets(secret, kept):
    out = redact(f"cmd {secret} end")
    assert "[REDACTED]" in out
    assert "hunter2" not in out
    assert kept in out
    assert out.startswith("cmd ") and out.endswith(" end")


def test_redact_leaves_ordinary_commands_alone():
    for text in ["git status", "npm run test", "pytest -q tests/test_x.py", "git diff --stat"]:
        assert redact(text) == text


def test_format_detection_redacts_and_shows_line_breaks():
    line = format_detection("2026-11-13 14:40:01", "allow_once_ss.png", 0.91234, None, "skipped: no text")
    assert line == "2026-11-13 14:40:01  template=allow_once_ss.png  score=0.912  text=(no text)  skipped: no text"
    multi = format_detection("t", "a.png", 0.88, "git   status\nrm -rf ~", "skipped")
    assert "text=git status " + chr(0x23CE) + " rm -rf ~" in multi
    secret = format_detection("t", "a.png", 0.9, "curl --password=hunter2 x", "skipped")
    assert "hunter2" not in secret


# --- headless import and packaging hygiene ---------------------------------------------------


def test_import_does_not_load_gui_or_macos_modules():
    code = (
        "import sys, autoclick; "
        "bad = [m for m in ('pyautogui', 'AppKit', 'Vision', 'Quartz', 'overlay') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_production_code_has_no_test_runner_special_cases():
    for name in ("autoclick.py", "allowlist.py", "command_region.py", "overlay.py", "state.py"):
        assert "pytest" not in (REPO / name).read_text(encoding="utf-8"), name
