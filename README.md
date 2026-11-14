# copilot-click

Watches the screen for the Copilot "Allow Once" approval button in VS Code and clicks it, but only for commands you have allow-listed (macOS; OpenCV, pyautogui and the macOS Vision framework).

By default nothing is clicked. A click happens only when **all** of these hold:

1. VS Code or VS Code Insiders is the frontmost app (unless `--any-app`).
2. The button template matches on screen.
3. macOS Vision reads the command from the code block directly above the button, and the layout is unambiguous (see [How the command is read](#how-the-command-is-read)).
4. The whole command matches a line of your allow list and contains no shell operators (unless that line opts in with `unsafe:`).
5. A second look a moment later finds the same button in the same place, the same command text, and the same frontmost app.

## Security model and its limits

This tool approves commands that an AI agent wants to run on your machine. Treat it as a convenience for a short list of harmless, read-only commands, not as a sandbox.

- **It is OCR, and it is best effort.** It reads pixels. OCR can misread characters (`|` as `l`, `;` as `:`), and text that is not visible on screen cannot be checked. Notably, if VS Code clips or horizontally scrolls a long command, text past the visible edge (for example after a long run of spaces) is invisible to OCR. Long or multi-line commands are refused for this reason, but no screenshot check can be complete.
- **Keep the allow list narrow.** Patterns must match the entire command. Prefer exact commands (`git status`) over wildcards (`git .*`). A pattern that allows a program which itself runs arbitrary code (`python .*`, `npm run .*`, `make .*`, `bash .*`) allows anything the agent can write into a file first.
- **Fail closed.** Missing or unreadable allow list, Vision unavailable, low OCR confidence, text touching the edge of the region being read, two text fragments on one line, an ellipsis, a change between the two looks: each of these means no click. The reason is written to the log.
- **Protect the allow list.** It lives outside your projects by default (`~/.config/copilot-click/allow.txt`). The tool refuses a world-writable allow list and warns when it is group-writable or inside a git work tree, where an agent could edit it.
- **Residual race.** The screen is checked twice, about 0.25 s apart, and the frontmost app is checked again right before the click. A change in the few milliseconds between the second check and the click cannot be detected.
- `--trust-all` turns all of this off and clicks every match. Do not use it unless you would approve every command unseen.

## Setup

1. Requires macOS, Python 3.13+ and [uv](https://docs.astral.sh/uv/). Run `uv sync` in this folder.
2. Create your allow list:

   ```
   mkdir -p ~/.config/copilot-click
   cp allow.txt.example ~/.config/copilot-click/allow.txt
   chmod 600 ~/.config/copilot-click/allow.txt
   ```

3. System Settings > Privacy & Security: grant your terminal app **Screen Recording** (otherwise screenshots are blank) and **Accessibility** (otherwise clicks and the pause hotkey do not work). Restart the terminal afterwards.
4. Check detection and reading without clicking:

   ```
   uv run autoclick.py --dry-run
   ```

   Trigger a Copilot terminal command approval in VS Code and watch the output. Each detection logs the command it read and the outcome, e.g. `dry run: would click` or `skipped: contains ';'`. If it says `no code block behind the command` or `... touches the edge of the crop`, see [Troubleshooting](#troubleshooting).

## Usage

```
uv run autoclick.py --dry-run     # detect, read and log; never click
uv run autoclick.py               # live: clicks only allow-listed commands
```

The script only scans while VS Code or VS Code Insiders is the frontmost app (unless `--any-app`), and logs when that changes.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--allow-file` | `~/.config/copilot-click/allow.txt` | allow list (`$XDG_CONFIG_HOME` is honoured) |
| `--log-file` | `~/Library/Logs/copilot-click/copilot-click.log` | detection log, created with mode 0600 (`~/.local/state/...` on other systems) |
| `--dry-run` | off | log what would happen; never click |
| `--threshold` | 0.85 | minimum template match score, 0–1 |
| `--interval` | 0.5 | seconds between scans |
| `--cooldown` | 1.5 | seconds to wait after a detection |
| `--confirm-delay` | 0.25 | seconds between deciding to click and the second look |
| `--max-clicks` | 0 | exit after N clicks (0 = unlimited) |
| `--template` | `assets/` | one PNG, or a directory of PNGs |
| `--pause-key` | `p` | letter used with Ctrl+Option+Cmd to pause/resume |
| `--command-width` | 900 | points to the right of the button's left edge to read |
| `--command-height` | 600 | points above the button to read |
| `--min-ocr-confidence` | 0.5 | refuse when Vision is less sure about a command line |
| `--save-crops DIR` | off | save each region that was read as a PNG, for troubleshooting (may contain secrets) |
| `--any-app` | off | do not require VS Code to be frontmost (the app still must not change before the click) |
| `--trust-all` | off | **dangerous**: click every match without reading the command |
| `--border` | 6 | glow border width in points |
| `--no-overlay` | off | do not draw the glowing border |

`--allow` and `--log` are accepted as aliases of `--allow-file` and `--log-file`.

### Pause

Press **Ctrl+Option+Cmd+P** to pause; press it again to resume. Exactly those modifiers are required: plain `p`, Cmd+P, or Ctrl+Option+Cmd+Shift+P do nothing. `--pause-key` changes the letter, not the modifiers. The terminal prints `PAUSED` / `RESUMED`, and the border turns purple with a "Paused" label. The hotkey needs the Accessibility permission. Pause wins over everything, including `--trust-all`.

### Allow list

One Python regular expression per line; blank lines and `#` comments are ignored. The file is re-read whenever it changes, and the patterns are compiled once per load.

- **Whole-command match.** Each pattern is matched with `re.fullmatch` against the command after Unicode normalization and after collapsing runs of whitespace to one space. `git status` allows exactly `git status`, not `git status -s`, `sudo git status` or `git status && curl … | sh`. `^` and `$` are unnecessary.
- **Shell operators are refused.** A command containing `;`, `&&`, `||`, `|`, `&`, a backtick, `$(`, `${`, `>`, `<`, a backslash, or more than one line is refused even if a pattern matches it. Look-alike characters OCR may produce (full-width `；`, `¦`, …) count too.
- **`unsafe:` opts in.** Prefix a line with `unsafe:` to allow operators for commands that line matches. It still has to match the whole command:

  ```
  unsafe:git log --oneline \| head
  ```

- **Broad patterns are rejected.** A line that would match an empty command or something like `rm -rf /` or `curl … | sh` is skipped with an error when the file loads (`.*`, `(npm test)?`, `sudo .*`). Lines with unbounded wildcards such as `.*`, `.+` or `\S+` load, with a warning.
- Invalid regexes are skipped and reported; they do not stop the program.

Examples (see `allow.txt.example`):

```
git status
git diff( --stat)?
git log --oneline( -n \d{1,3})?
npm (run )?test
pytest( -q)?
```

### How the command is read

The approval widget is expected to show the command in a code block, with the button below it. On each detection:

1. A generous region above the button is cropped: from a little left of the button to `--command-width` points right of it and `--command-height` points up, clipped to the screen and, when it can be determined, to the editor window.
2. Vision returns each text line with its bounding box.
3. The line closest above the button is the anchor. The code block is found by flood-filling the anchor's background colour; it must have a colour distinct from the panel around it.
4. Refusals: the block touches any edge of the crop (it may be cut off, or it is not a block at all); text straddles the block's edge; two fragments share a line; a line reaches the block's right edge or ends in `…`; OCR confidence is below `--min-ocr-confidence`.
5. All lines inside the block are the command, joined with line breaks, so a multi-line command is refused unless an `unsafe:` line matches it.

### Session log

Every detection is appended to the log file (default `~/Library/Logs/copilot-click/copilot-click.log`, mode 0600, directory 0700): time, template, score, the command read (or `(no text)`), and the outcome with its reason. Obvious secrets in the command (`ghp_…`, `sk-…`, `AKIA…`, `--password=…`, `TOKEN=…`, bearer tokens, credentials in URLs) are replaced with `[REDACTED]`. Nothing is written when no button was found.

### Stopping

Press Ctrl+C, or flick the mouse into any corner of the main screen. A corner flick stops it for good: the border turns red for a moment and the script exits. Other displays' corners do not count. On exit it prints how many prompts it auto-allowed this session.

### Status glow

A thin click-through border around the main screen shows the state:

| Colour | Meaning |
| --- | --- |
| Green | live, VS Code is in front, scanning |
| Blue | same, but `--dry-run` (never clicks) |
| Amber | running, waiting because another app is in front |
| Purple + "Paused" | paused with the hotkey |
| Red + "Auto-click disabled" | stopped by a corner flick; exits a moment later |

The border band is cropped out of every screenshot before matching. Use `--no-overlay` to turn it off.

## Troubleshooting

- **`no code block behind the command` / `code block touches the edge of the crop`**: run with `--dry-run --save-crops /tmp/copilot-crops` and look at the saved PNGs. The command's code block must be fully inside the crop and have a background different from the panel. If your chat panel is wider than 900 points, raise `--command-width`; if your theme gives code blocks the same background as the panel, choose a theme where they differ. Delete the saved crops afterwards.
- **`command runs to the code block's right edge`**: the command is long enough to wrap or be clipped. That is refused on purpose.
- **No matches at all**: check the score with `--dry-run`, and add a PNG crop of the button as it looks with your theme and display scale to `assets/` (see below).

## Editors

VS Code (`com.microsoft.VSCode`) and VS Code Insiders (`com.microsoft.VSCodeInsiders`) are supported. Cursor's bundle id (`com.todesktop.230313mzl4w4u92`) is also accepted, but Cursor is **untested** and no Cursor button template ships with this repo: add your own template to `assets/` and verify with `--dry-run` before relying on it.

## Adding a template

Save a tight PNG crop of the button, as it looks on your screen, into `assets/` and run again. Every PNG in the folder is tried and the highest score at or above the threshold wins; filename order is only a tie-break. Only one click is considered per scan.

## Development

```
uv run pytest
uvx ruff check
```

The decision logic is pure and runs headless (on Linux too): `allowlist.py` (parsing and matching), `command_region.py` (crop geometry and command extraction), and the decide/re-check flow in `autoclick.py`. pyautogui and the macOS frameworks are imported only inside the functions that use them. Template matching, the crop and extraction, and the two-look click flow are tested with synthetic screens and a fake OCR engine; Vision itself, the hotkey monitor, the overlay and real clicks can only be checked on a Mac.

## Project layout

| File | Purpose |
| --- | --- |
| `autoclick.py` | CLI, template matching, decide/re-check/click flow, macOS glue, scan loop |
| `allowlist.py` | allow-list parsing, whole-command matching, shell-operator and breadth checks, file reloading |
| `command_region.py` | where to read the command and which OCR text is the command |
| `overlay.py` | the glowing border window (PyObjC) |
| `state.py` | state shared between the scanner thread and the main thread |
| `assets/allow_once_ss.png` | the "Allow Once" button image |
| `allow.txt.example` | sample allow list |
| `tests/` | `uv run pytest` |
| `LICENSE` | MIT |
