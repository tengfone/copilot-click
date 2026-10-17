# copilot-click

Watches the screen for the Copilot "Allow Once" button in VS Code and clicks it (macOS, OpenCV + pyautogui).

## Setup

1. Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). Run `uv sync` in this folder.
2. The button image lives at `assets/allow_once_ss.png`. If the match score is low on your display
   (different theme or zoom), replace it with a tight crop of the button taken on that display.
3. System Settings > Privacy & Security: grant your terminal app both
   **Screen Recording** (otherwise screenshots are blank) and **Accessibility** (otherwise clicks are ignored).
   Restart the terminal afterwards.

## Usage

Run from this folder, in a terminal app that has the permissions above:

```
uv run autoclick.py --dry-run     # detect and log only; use this first to check the score
uv run autoclick.py               # live
```

Then switch to VS Code. The script only scans while VS Code (or Insiders) is the frontmost app,
unless `--any-app` is set, and it logs whenever it pauses or resumes.

Options: `--threshold` (0.85), `--interval` (0.5s), `--cooldown` (1.5s), `--max-clicks`, `--template`, `--any-app`,
`--border` (glow width in points, 6), `--no-overlay`.

### Stopping

Press Ctrl+C, or flick the mouse into any corner of the main screen. A corner flick stops it for good:
the border turns red for a moment and the script exits. Other displays' corners do not count.

On exit it prints how many prompts it auto-allowed this session, for example `Auto-allowed 12 prompts this session.`
(a `--dry-run` reports how many times it found the button instead). The count is not saved between runs.

### Status glow

A thin click-through border around the main screen shows what the clicker is doing:

| Colour | Meaning |
| --- | --- |
| Green | live, VS Code is in front, scanning |
| Blue | same, but `--dry-run` (never clicks) |
| Amber | running, waiting because another app is in front |
| Red + "Auto-click disabled" | stopped by a corner flick; the script exits a moment later |

The border band is cropped out of every screenshot before matching, so it can never be mistaken for the button.
Use `--no-overlay` to turn the border off.

## How it behaves

- After a click the cursor is returned to where it was, and a cooldown prevents double-clicking the same prompt.
- It approves every matching prompt without reading the command. Stop it when working with untrusted content.
- The button must be visible: not covered by another window and on the current Space.

## Project layout

| File | Purpose |
| --- | --- |
| `autoclick.py` | CLI, template matching, scan loop, corner watcher |
| `overlay.py` | the glowing border window (PyObjC) |
| `state.py` | state shared between the scanner thread and the main thread |
| `assets/allow_once_ss.png` | the button image to look for |
| `tests/test_matching.py` | `uv run pytest` |
