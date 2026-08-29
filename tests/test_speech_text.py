from __future__ import annotations

from app.utils.text import to_speech_text


def test_strips_markdown_structure():
    md = (
        "## Latest release\n\n"
        "- **Python 3.13.1** is current\n"
        "- See [the docs](https://docs.python.org/3/) for details\n\n"
        "| Version | Date |\n|---|---|\n| 3.13.1 | Dec |\n\n"
        "```python\nprint('hi')\n```\n"
    )
    out = to_speech_text(md)
    for bad in ("#", "**", "|", "```", "http", "[", "]", "(https"):
        assert bad not in out
    assert "Python 3.13.1 is current" in out
    assert "the docs" in out  # link text survives, the URL does not


def test_removes_bare_urls_and_citations():
    out = to_speech_text("Python 3.13 is out [1]. Source: https://python.org/downloads")
    assert "http" not in out and "[1]" not in out
    assert "Python 3.13 is out" in out


def test_joins_bullets_into_speech_with_cjk_punctuation():
    out = to_speech_text("- 第一點\n- 第二點\n- 第三點")
    assert "-" not in out
    assert "第一點。" in out and "第三點" in out


def test_keeps_english_technical_terms():
    src = "呢個 request 會先經 Router 判斷需唔需要 Web Search。"
    assert to_speech_text(src) == src


def test_truncates_on_sentence_boundary():
    text = "This is a sentence. " * 200
    out = to_speech_text(text, max_chars=100)
    assert len(out) <= 100
    assert out.endswith(".")


def test_empty_input():
    assert to_speech_text("") == ""
