# copilot-click

Watches the screen for a Copilot approval button in VS Code or Cursor and clicks it (macOS, OpenCV + pyautogui).

By default it does **not** click a match. A live click happens only when text recognized just above the button matches a regex in `allow.txt`. `--trust-all` restores the old click-every-match behaviour and is off unless you pass it.

## Setup

1. Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). Run `uv sync` in this folder.
2. Button images live in `assets/`. Every PNG in that folder is tried; the best score at or above the threshold wins. `allow_once_ss.png` is the original dark "Allow Once" crop. `allow_once_light.png` is a light-theme variant generated from that crop so a brightness shift still has a second template. Drop additional PNGs into `assets/` for Cursor, or for light/dark buttons this repo cannot photograph.
3. Copy `allow.txt.example` to `allow.txt` and list the commands you are willing to auto-allow. `allow.txt` is gitignored.
4. System Settings > Privacy & Security: grant your terminal app both
   **Screen Recording** (otherwise screenshots are blank) and **Accessibility** (otherwise clicks are ignored, and the pause key cannot be read).
   Restart the terminal afterwards.

## Usage

Run from this folder, in a terminal app that has the permissions above:

```
uv run autoclick.py --dry-run     # detect and log only; use this first to check the score
uv run autoclick.py               # live, but clicks only when allow.txt matches
uv run autoclick.py --trust-all   # live, click every match (the old behaviour)
```

Then switch to VS Code, VS Code Insiders, or Cursor. The script only scans while one of those is the frontmost app, unless `--any-app` is set, and it logs whenever it pauses or resumes, naming which family is in front (`VS Code`, `VS Code Insiders`, or `Cursor`). Cursor's bundle id is `com.todesktop.230313mzl4w4u92`.

Options:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--threshold` | 0.85 | minimum match score, 0–1 |
| `--interval` | 0.5 | seconds between scans |
| `--cooldown` | 1.5 | seconds to wait after a detection |
| `--max-clicks` | 0 | exit after N live clicks (0 = unlimited) |
| `--template` | `assets/` | one PNG, or a directory of PNGs |
| `--allow` | `allow.txt` | regex allow list |
| `--log` | `copilot-click.log` | session log path |
| `--pause-key` | `p` | global key that toggles pause |
| `--trust-all` | off | click every match; skip the allow list and OCR |
| `--dry-run` | off | log matches but never click |
| `--any-app` | off | do not require an allowed editor to be frontmost |
| `--border` | 6 | glow border width in points |
| `--no-overlay` | off | do not draw the glowing border |

### Pause

Press `p` (or `--pause-key`) to toggle pause. The key is a macOS global monitor, so it works while VS Code or Cursor is in front. It is not installed when the program is running under pytest.

While paused the scanner does not click. The border turns purple and reads "Paused". Press the key again to resume. Pause wins over `--trust-all`.

### Allow list

`allow.txt` is one Python regular expression per line. Blank lines and lines starting with `#` are ignored. A bad regex is skipped and reported; it does not crash the run.

Before any live click, a region above the match is read with macOS Vision (`pyobjc-framework-Vision`). The recognized text must match one pattern. If `allow.txt` is missing or has no usable patterns, the script logs `allow list is empty; not clicking` and does not click. If Vision cannot be imported (Linux, or a machine without the framework), recognition returns no text and live mode refuses to click unless `--trust-all` is set.

Patterns are searched, not anchored, unless you write `^` and `$`. Prefer specific patterns (`^git status$`) over broad ones.

### Session log

Each detection is appended to `copilot-click.log` in the working directory (override with `--log`): time, template name, score, OCR text or `(no text)`, and whether it clicked or skipped and why. Nothing is written on the scan timer when no template matched.

### Stopping

Press Ctrl+C, or flick the mouse into any corner of the main screen. A corner flick stops it for good:
the border turns red for a moment and the script exits. Other displays' corners do not count.

On exit it prints how many prompts it auto-allowed this session, for example `Auto-allowed 12 prompts this session.`
(a `--dry-run` reports how many times it found the button instead). The count is not saved between runs.

### Status glow

A thin click-through border around the main screen shows what the clicker is doing:

| Colour | Meaning |
| --- | --- |
| Green | live, an allowed editor is in front, scanning |
| Blue | same, but `--dry-run` (never clicks) |
| Amber | running, waiting because another app is in front |
| Purple + "Paused" | pause hotkey is on; no clicks until you press it again |
| Red + "Auto-click disabled" | stopped by a corner flick; the script exits a moment later |

The border band is cropped out of every screenshot before matching, so it can never be mistaken for the button.
Use `--no-overlay` to turn the border off.

## How it behaves

- After a click the cursor is returned to where it was, and a cooldown prevents double-clicking the same prompt.
- Only one click is considered per scan: the highest-scoring template at or above the threshold. Extra templates do not click extra times.
- Without `--trust-all`, a match is skipped when the allow list is empty, when no command text is recognized, or when the text matches no pattern.
- The button must be visible: not covered by another window and on the current Space.

## Adding a template

Save a tight PNG crop of the button into `assets/` and run again. Filename order is only a tie-break; the highest score wins. A photo of the button on your own theme is better than the generated `allow_once_light.png`.

## Suggested GitHub topics

`macos`, `vscode`, `cursor`, `opencv`, `autoclicker`

## Project layout

| File | Purpose |
| --- | --- |
| `autoclick.py` | CLI, template matching, allow list, scan loop, corner watcher |
| `overlay.py` | the glowing border window (PyObjC) |
| `state.py` | state shared between the scanner thread and the main thread |
| `assets/*.png` | button images to look for |
| `allow.txt.example` | sample allow list; copy to `allow.txt` |
| `LICENSE` | MIT |
| `tests/test_matching.py` | `uv run pytest` |
