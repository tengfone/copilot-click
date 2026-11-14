"""Allow-list parsing and command matching for copilot-click.

Everything here is pure Python (no screen, OCR or macOS APIs) so it can be
unit-tested anywhere.

Rules, in short (see README "Allow list"):

* One regular expression per line. It must match the *whole* command
  (``re.fullmatch``) after whitespace is normalized; there is no substring
  matching, so ``^`` and ``$`` are unnecessary.
* A command that contains a shell control or meta operator (``;``, ``&&``,
  ``||``, ``|``, ``&``, backticks, ``$(``, ``${``, ``>``, ``<``, a backslash,
  or more than one line) is refused, even when a pattern matches it, unless
  the matching line starts with ``unsafe:``.
* Lines that would allow an empty command or an obviously destructive one
  (``rm -rf /``, ``curl ... | sh`` and similar) are rejected when the file is
  loaded. Lines that use unbounded wildcards such as ``.*`` load with a
  warning.
"""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

UNSAFE_PREFIX = "unsafe:"

# Checked in order; longer operators first so the reason names the real one.
SHELL_OPERATORS: tuple[tuple[str, str], ...] = (
    ("\n", "a line break (multi-line command)"),
    ("\r", "a carriage return"),
    ("&&", "'&&'"),
    ("||", "'||'"),
    ("$(", "'$('"),
    ("${", "'${'"),
    (";", "';'"),
    ("|", "'|'"),
    ("&", "'&'"),
    ("`", "a backtick"),
    (">", "'>'"),
    ("<", "'<'"),
    ("\\", "a backslash"),
)

# OCR (and copy/paste) can produce look-alike characters. Map them to the
# ASCII character the shell would see, so they cannot slip past the operator
# check and so patterns written in ASCII still match.
_LOOKALIKES = str.maketrans(
    {
        "\u00a6": "|",  # broken bar
        "\u2223": "|",  # divides
        "\u2502": "|",  # box drawings light vertical
        "\u2758": "|",  # light vertical bar
        "\u037e": ";",  # Greek question mark
        "\u2018": "'",  # curly quotes
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",  # hyphen, non-breaking hyphen, figure dash, en dash
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "--",  # em dash: OCR often reads "--" as one
        "\u2212": "-",  # minus sign
        "\u200b": None,  # zero-width space, non-joiner, joiner, word joiner, BOM
        "\u200c": None,
        "\u200d": None,
        "\u2060": None,
        "\ufeff": None,
    }
)

# A pattern that fully matches any of these is rejected at load time.
BROAD_PROBES: tuple[str, ...] = (
    "",
    "rm -rf /",
    "rm -rf ~",
    "sudo rm -rf /",
    "curl -fsSL https://example.com/install.sh | sh",
    "chmod -R 777 /",
    "dd if=/dev/zero of=/dev/disk0",
)

# Unbounded wildcards: allowed, but the loader warns about them.
_WILDCARD_RE = re.compile(r"(?<!\\)\.[*+]|\\S[*+]|\[\^[^\]]*\][*+]")


def prepare_command(text: str) -> str:
    """Unicode-normalize OCR text and trim the ends. Inner line breaks are kept."""
    text = unicodedata.normalize("NFKC", text).translate(_LOOKALIKES)
    lines = [line.strip() for line in text.splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def normalize_command(text: str) -> str:
    """The form patterns are matched against: NFKC, look-alikes mapped, one space between words."""
    return collapse_whitespace(prepare_command(text))


def shell_operator(text: str) -> str | None:
    """Describe the first shell control/meta operator in prepared text, or None."""
    for token, name in SHELL_OPERATORS:
        if token in text:
            return name
    return None


@dataclass(frozen=True)
class AllowRule:
    pattern: re.Pattern[str]
    unsafe: bool = False
    line_no: int = 0
    source: str = ""

    def matches(self, normalized: str) -> bool:
        return self.pattern.fullmatch(normalized) is not None


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str
    command: str = ""  # normalized command that was checked
    rule: AllowRule | None = None


@dataclass
class AllowList:
    rules: list[AllowRule] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # lines that were rejected
    warnings: list[str] = field(default_factory=list)  # lines that loaded but look risky

    def __bool__(self) -> bool:
        return bool(self.rules)

    def __len__(self) -> int:
        return len(self.rules)

    def evaluate(self, text: str | None) -> Verdict:
        """Decide whether OCR'd command text may be auto-approved. Fails closed."""
        if text is None or not text.strip():
            return Verdict(False, "no command text")
        if not self.rules:
            return Verdict(False, "allow list empty")
        prepared = prepare_command(text)
        command = collapse_whitespace(prepared)
        if not command:
            return Verdict(False, "no command text")
        operator = shell_operator(prepared)
        blocked_by: AllowRule | None = None
        for rule in self.rules:
            if not rule.matches(command):
                continue
            if operator and not rule.unsafe:
                blocked_by = blocked_by or rule
                continue
            return Verdict(True, "allowed", command, rule)
        if operator and blocked_by is not None:
            return Verdict(
                False,
                f"contains {operator}; line {blocked_by.line_no} matches but is not marked {UNSAFE_PREFIX}",
                command,
            )
        if operator:
            return Verdict(False, f"contains {operator}", command)
        return Verdict(False, "not allowed", command)


def check_pattern_breadth(pattern: re.Pattern[str]) -> str | None:
    """Return why a pattern is too broad to load, or None."""
    for probe in BROAD_PROBES:
        if pattern.fullmatch(collapse_whitespace(probe)) is not None:
            shown = probe or "an empty command"
            return f"it would allow {shown!r}"
    return None


def parse_allow_list(text: str, source: str = "allow list") -> AllowList:
    """Parse allow-list text. Bad or overly broad lines are reported, never raised."""
    result = AllowList()
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        unsafe = line.startswith(UNSAFE_PREFIX)
        body = line[len(UNSAFE_PREFIX) :].strip() if unsafe else line
        where = f"{source}:{line_no}"
        if not body:
            result.problems.append(f"{where}: empty pattern after {UNSAFE_PREFIX!r}; line skipped")
            continue
        try:
            pattern = re.compile(body)
        except re.error as exc:
            result.problems.append(f"{where}: invalid regex {body!r} ({exc}); line skipped")
            continue
        too_broad = check_pattern_breadth(pattern)
        if too_broad:
            result.problems.append(f"{where}: pattern {body!r} is too broad ({too_broad}); line skipped")
            continue
        if _WILDCARD_RE.search(body):
            result.warnings.append(
                f"{where}: pattern {body!r} uses an unbounded wildcard; it may allow more than you expect"
            )
        if unsafe:
            result.warnings.append(f"{where}: {UNSAFE_PREFIX} line allows shell operators: {body!r}")
        result.rules.append(AllowRule(pattern, unsafe, line_no, body))
    return result


def as_allow_list(patterns: AllowList | Iterable[str]) -> AllowList:
    if isinstance(patterns, AllowList):
        return patterns
    return parse_allow_list("\n".join(str(item) for item in patterns))


def command_allowed(text: str | None, patterns: AllowList | Iterable[str]) -> bool:
    """True only when the whole command matches a usable allow-list line."""
    return as_allow_list(patterns).evaluate(text).allowed


# --- the allow-list file -------------------------------------------------------


def find_git_worktree(path: Path) -> Path | None:
    """The nearest ancestor directory of path that contains .git, or None."""
    for parent in path.resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def file_safety(path: Path) -> tuple[list[str], list[str]]:
    """(fatal problems, warnings) about where the allow-list file lives and who can write it.

    A world-writable file is fatal: any local process could add rules. A
    group-writable file, one owned by someone else, or one inside a git
    work tree (which a coding agent may edit) only warns.
    """
    fatal: list[str] = []
    warnings: list[str] = []
    try:
        info = path.stat()
    except OSError:
        return fatal, warnings
    mode = stat.S_IMODE(info.st_mode)
    if mode & stat.S_IWOTH:
        fatal.append(f"{path} is world-writable (mode {mode:o}); refusing to use it. Run: chmod 600 {path}")
    elif mode & stat.S_IWGRP:
        warnings.append(f"{path} is group-writable (mode {mode:o}); consider: chmod 600 {path}")
    getuid = getattr(os, "getuid", None)
    if getuid is not None and info.st_uid != getuid():
        warnings.append(f"{path} is owned by uid {info.st_uid}, not you")
    repo = find_git_worktree(path)
    if repo is not None:
        warnings.append(
            f"{path} is inside the git work tree {repo}; an agent working there could edit your allow list. "
            "Prefer ~/.config/copilot-click/allow.txt"
        )
    return fatal, warnings


def load_allow_file(path: Path) -> tuple[AllowList, list[str]]:
    """Load and compile an allow-list file. Returns (allow list, messages to log).

    A missing, unreadable or world-writable file yields an empty list, so
    nothing is clicked.
    """
    if not path.is_file():
        return AllowList(), [f"allow list {path} not found; nothing will be clicked"]
    fatal, messages = file_safety(path)
    if fatal:
        return AllowList(), fatal
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return AllowList(), [f"could not read allow list {path} ({exc}); nothing will be clicked"]
    allow = parse_allow_list(text, source=str(path))
    return allow, messages + allow.problems + allow.warnings


class AllowListWatcher:
    """Holds the compiled allow list and reloads it when the file changes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stamp: tuple[int, int, int] | None = None
        self.allow = AllowList()
        self.messages: list[str] = []
        self.loaded = False

    def _current_stamp(self) -> tuple[int, int, int] | None:
        try:
            info = self.path.stat()
        except OSError:
            return None
        return (info.st_mtime_ns, info.st_size, info.st_mode)

    def refresh(self) -> bool:
        """Reload if the file's mtime, size or mode changed. True when it reloaded."""
        stamp = self._current_stamp()
        if self.loaded and stamp == self._stamp:
            return False
        self._stamp = stamp
        self.allow, self.messages = load_allow_file(self.path)
        self.loaded = True
        return True

    def current(self) -> AllowList:
        self.refresh()
        return self.allow
