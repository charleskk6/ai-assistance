"""Exercise the shell entry points for real.

These exist because .env token resolution has broken three separate times, and
each break was invisible to the Python test suite: a GNU-only sed construct that
BSD sed on macOS silently ignores, a blank placeholder shadowing a real token,
and a path resolved against the wrong directory.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from app.config import PROJECT_ROOT

SMOKE = PROJECT_ROOT / "scripts" / "smoke.sh"
SET_TOKEN = PROJECT_ROOT / "scripts" / "set-token.sh"
UNREACHABLE = "http://127.0.0.1:9"  # nothing listens here; we only test startup


def run_smoke(env_file, *args):
    clean = {k: v for k, v in os.environ.items() if k != "LOCAL_ASSISTANT_TOKEN"}
    return subprocess.run(
        [str(SMOKE), UNREACHABLE, *args],
        capture_output=True, text=True, env={**clean, "ENV_FILE": str(env_file)},
    )


def write_env(tmp_path, text: str):
    path = tmp_path / "custom.env"
    path.write_text(text)
    return path


# --- token resolution ------------------------------------------------------
@pytest.mark.parametrize(
    "content,label",
    [
        ("LOCAL_ASSISTANT_TOKEN=tok123\n", "plain"),
        ("HOST=0.0.0.0\nLOCAL_ASSISTANT_TOKEN=tok123\n", "after other settings"),
        ("export LOCAL_ASSISTANT_TOKEN=tok123\n", "export prefix"),
        ('LOCAL_ASSISTANT_TOKEN="tok123"\n', "double quoted"),
        ("LOCAL_ASSISTANT_TOKEN='tok123'\n", "single quoted"),
        ("LOCAL_ASSISTANT_TOKEN = tok123\n", "spaces around equals"),
        ("LOCAL_ASSISTANT_TOKEN=tok123\r\n", "CRLF line endings"),
        ("# LOCAL_ASSISTANT_TOKEN=\nLOCAL_ASSISTANT_TOKEN=tok123\n", "after a comment"),
        ("LOCAL_ASSISTANT_TOKEN=\nLOCAL_ASSISTANT_TOKEN=tok123\n", "blank then real"),
    ],
)
def test_smoke_finds_the_token(tmp_path, content, label):
    result = run_smoke(write_env(tmp_path, content))
    assert "token from" in result.stdout, f"{label}: {result.stdout}{result.stderr}"
    assert "No LOCAL_ASSISTANT_TOKEN" not in result.stdout, label


def test_a_single_real_assignment_is_never_reported_as_blank(tmp_path):
    """The regression that cost several rounds: a hand-rolled parser found
    nothing and blamed the file, which plainly held a valid token."""
    env = write_env(tmp_path, "# comment\nHOST=0.0.0.0\nLOCAL_ASSISTANT_TOKEN=IYMn7fIg\n")
    assert "token from" in run_smoke(env).stdout


def test_a_trailing_blank_assignment_empties_the_token(tmp_path):
    """The script must agree with the server, not be cleverer than it.

    pydantic-settings takes the last assignment, so a trailing blank one empties
    a real token above it and the backend refuses to start. Reporting a usable
    token here would send the script off testing against a server that is not
    running.
    """
    env = write_env(tmp_path, "LOCAL_ASSISTANT_TOKEN=tok123\nLOCAL_ASSISTANT_TOKEN=\n")
    out = run_smoke(env).stdout
    assert "No LOCAL_ASSISTANT_TOKEN" in out
    assert "line 1:" in out and "line 2:" in out  # both shown, so it is fixable


def test_script_and_app_resolve_the_same_token(tmp_path):
    """Guaranteed by construction: the script asks the app's own loader."""
    from app.config import Settings

    env = write_env(tmp_path, "HOST=0.0.0.0\nLOCAL_ASSISTANT_TOKEN=agreed-token\n")
    assert Settings(_env_file=env).local_assistant_token == "agreed-token"
    assert "token from" in run_smoke(env).stdout


def test_missing_file_and_blank_token_are_distinguished(tmp_path):
    missing = run_smoke(tmp_path / "nope.env").stdout
    assert "does not exist" in missing

    blank = run_smoke(write_env(tmp_path, "LOCAL_ASSISTANT_TOKEN=\n")).stdout
    assert "every assignment is blank" in blank
    assert "line 1:" in blank  # shows the offending line

    absent = run_smoke(write_env(tmp_path, "HOST=0.0.0.0\n")).stdout
    assert "no LOCAL_ASSISTANT_TOKEN line at all" in absent


def test_argument_beats_the_env_file(tmp_path):
    result = run_smoke(write_env(tmp_path, "LOCAL_ASSISTANT_TOKEN=from-file\n"), "from-arg")
    assert "token from argument" in result.stdout


# --- set-token.sh ----------------------------------------------------------
def test_set_token_collapses_duplicates_and_keeps_other_settings():
    env_path = PROJECT_ROOT / ".env"
    backup = PROJECT_ROOT / ".env.bak"
    if env_path.exists():
        pytest.skip("refusing to touch a real local .env")
    try:
        env_path.write_text(
            "LOCAL_ASSISTANT_TOKEN=old\nHOST=0.0.0.0\nLOCAL_ASSISTANT_TOKEN=\n"
        )
        subprocess.run([str(SET_TOKEN), "brand-new"], capture_output=True, check=True)
        assignments = [
            ln for ln in env_path.read_text().splitlines()
            if ln.strip().startswith("LOCAL_ASSISTANT_TOKEN")
        ]
        assert assignments == ["LOCAL_ASSISTANT_TOKEN=brand-new"]
        assert "HOST=0.0.0.0" in env_path.read_text()
    finally:
        env_path.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)


def test_set_token_generates_one_when_not_given():
    env_path = PROJECT_ROOT / ".env"
    backup = PROJECT_ROOT / ".env.bak"
    if env_path.exists():
        pytest.skip("refusing to touch a real local .env")
    try:
        subprocess.run([str(SET_TOKEN)], capture_output=True, check=True)
        line = next(
            ln for ln in env_path.read_text().splitlines()
            if ln.startswith("LOCAL_ASSISTANT_TOKEN=")
        )
        assert len(line.split("=", 1)[1]) >= 32
    finally:
        env_path.unlink(missing_ok=True)
        backup.unlink(missing_ok=True)
