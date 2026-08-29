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


def test_drops_a_trailing_half_sentence():
    """The model hits its token ceiling mid-thought; a voice must not read a
    half-word."""
    cut_off = "Dependency injection 係一種 design pattern。例如你有個 server object，佢需要一個 databa"
    assert (
        to_speech_text(cut_off, drop_trailing_fragment=True)
        == "Dependency injection 係一種 design pattern。"
    )
    # Without the runtime's truncation signal, the text is left alone - plenty of
    # complete answers simply end without punctuation.
    assert to_speech_text(cut_off) == cut_off


def test_keeps_a_complete_answer_untouched():
    assert to_speech_text("全部完整。第二句都完整。") == "全部完整。第二句都完整。"
    assert to_speech_text("This is complete.") == "This is complete."


def test_does_not_trim_a_short_unpunctuated_reply():
    assert to_speech_text("yes", drop_trailing_fragment=True) == "yes"
    assert to_speech_text("3.14.7", drop_trailing_fragment=True) == "3.14.7"


def test_does_not_gut_an_answer_whose_fragment_is_the_bulk():
    """One early full stop then a long run-on: trimming would throw the answer
    away, so leave it."""
    text = "好。" + "x" * 200
    assert to_speech_text(text, drop_trailing_fragment=True) == text


def test_trailing_bracket_counts_as_a_finished_sentence():
    text = "佢係一個 object（唔係 class）"
    assert to_speech_text(text, drop_trailing_fragment=True) == text
