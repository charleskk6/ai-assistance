"""Configuration loading. These cover two failures found on real hardware:
a .env read relative to the wrong directory, and a blank placeholder assignment
shadowing a real token."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.config import ENV_FILE, PROJECT_ROOT, Settings


def test_env_file_is_anchored_to_the_project_not_the_cwd():
    """A relative ".env" is silently ignored when the server is started from
    another directory."""
    assert ENV_FILE.is_absolute()
    assert ENV_FILE == PROJECT_ROOT / ".env"
    assert (PROJECT_ROOT / "app" / "config.py").exists()


def test_settings_find_the_env_file_from_any_working_directory(tmp_path):
    """Import and read config with the process CWD somewhere unrelated."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():  # never clobber a real local .env
        return
    env_path.write_text("LOCAL_ASSISTANT_TOKEN=anchored-token\n")
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "from app.config import Settings; print(Settings().local_assistant_token)"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            # The variable must be ABSENT, not empty: an exported empty string
            # still takes precedence over the .env file.
            env={k: v for k, v in os.environ.items() if k != "LOCAL_ASSISTANT_TOKEN"}
            | {"PYTHONPATH": str(PROJECT_ROOT)},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "anchored-token"
    finally:
        env_path.unlink()


def test_a_blank_assignment_after_a_real_one_wins_which_is_why_it_is_a_trap(tmp_path):
    """Documents the behaviour that caused the bug: the LAST assignment wins, so
    a leftover blank placeholder silently empties a real token. set-token.sh
    exists to guarantee there is only ever one assignment."""
    env = tmp_path / ".env"
    env.write_text("LOCAL_ASSISTANT_TOKEN=real\nLOCAL_ASSISTANT_TOKEN=\n")
    assert Settings(_env_file=env).local_assistant_token == ""

    env.write_text("LOCAL_ASSISTANT_TOKEN=real\n")
    assert Settings(_env_file=env).local_assistant_token == "real"


def test_env_example_does_not_ship_a_shadowing_blank_assignment():
    text = (PROJECT_ROOT / ".env.example").read_text()
    active = [
        ln for ln in text.splitlines()
        if ln.strip().startswith("LOCAL_ASSISTANT_TOKEN")
    ]
    assert active == [], f"uncommented placeholder would shadow a real token: {active}"
    assert "# LOCAL_ASSISTANT_TOKEN=" in text  # still documented


def test_set_token_script_leaves_exactly_one_assignment(tmp_path):
    repo_env = PROJECT_ROOT / ".env"
    if repo_env.exists():
        return
    try:
        repo_env.write_text(
            "LOCAL_ASSISTANT_TOKEN=old\nHOST=0.0.0.0\nLOCAL_ASSISTANT_TOKEN=\n"
        )
        subprocess.run(
            [str(PROJECT_ROOT / "scripts" / "set-token.sh"), "fresh-token"],
            capture_output=True, text=True, check=True,
        )
        lines = [
            ln for ln in repo_env.read_text().splitlines()
            if ln.strip().startswith("LOCAL_ASSISTANT_TOKEN")
        ]
        assert lines == ["LOCAL_ASSISTANT_TOKEN=fresh-token"]
        assert Settings(_env_file=repo_env).local_assistant_token == "fresh-token"
        assert "HOST=0.0.0.0" in repo_env.read_text()  # other settings preserved
    finally:
        repo_env.unlink(missing_ok=True)
        Path(PROJECT_ROOT / ".env.bak").unlink(missing_ok=True)
