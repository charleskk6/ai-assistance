import pytest

from britcoach.coaching.detector import apply_corrections, detect
from britcoach.coaching.policy import CorrectionPolicy, Strategy


@pytest.mark.parametrize("text,original,suggestion", [
    ("Yesterday I bring my son to school.", "I bring", "I took"),
    ("Last week we visit my parents", "we visit", "we visited"),
    ("I didn't went there", "didn't went", "didn't go"),
    ("I am agree with you", "I am agree", "I agree"),
    ("We discussed about the plan", "discussed about", "discussed"),
    ("I live here since 3 years", "since 3 years", "for 3 years"),
    ("She like football", "She like", "She likes"),
    ("he don't care", "he don't", "he doesn't"),
    ("Can you explain me the rules?", "explain me", "explain to me"),
    ("I need some advices", "advices", "advice"),
    ("It's more easier now", "more easier", "easier"),
    ("My son has 7 years", "My son has 7 years", "My son is 7"),
    ("I made a photo of it", "made a photo", "took a photo"),
    ("I'm looking forward to see you", "looking forward to see", "looking forward to seeing"),
    ("People is very friendly", "People is", "People are"),
])
def test_detects_common_errors(text, original, suggestion):
    issues = detect(text)
    assert any(i.original == original and i.suggestion == suggestion for i in issues), issues
    for i in issues:
        assert i.original in text  # grounded: always an exact span of what was said


@pytest.mark.parametrize("text", [
    "I usually go to the park every weekend.",
    "Yesterday I took my son to school.",
    "Does she like tea? He doesn't like it.",
    "I really like it here.",
    "Did he go home yesterday?",
    "We're making the most of the sunshine.",
    "I've lived here since 2019.",
    "I'll bring my son to school tomorrow.",
])
def test_no_false_positives_on_correct_english(text):
    assert detect(text) == []


def test_apply_corrections():
    text = "Yesterday I bring my son to school"
    assert apply_corrections(text, detect(text)) == "Yesterday I took my son to school"


def test_policy_recasts_minor_then_briefs_on_repeat():
    p = CorrectionPolicy(sensitivity="medium", brief_cooldown_turns=3)
    d1 = p.decide(detect("Yesterday I bring my son to school"))
    assert d1.strategy is Strategy.RECAST
    d2 = p.decide(detect("Last night we go to the cinema"))
    assert d2.strategy is Strategy.RECAST
    d3 = p.decide(detect("Last week I buy a bike"))
    assert d3.strategy is Strategy.BRIEF and d3.issue.category == "past tense"
    # cooldown: the very next repeat is a recast, not another tip
    d4 = p.decide(detect("Yesterday I see her"))
    assert d4.strategy is Strategy.RECAST


def test_policy_no_issues_and_off():
    p = CorrectionPolicy(sensitivity="off")
    assert p.decide([]).strategy is Strategy.NONE
    d = p.decide(detect("I am agree"))
    assert d.strategy is Strategy.NONE and p.counts["verb pattern"] == 1  # still logged for review


def test_policy_low_ignores_minor():
    p = CorrectionPolicy(sensitivity="low")
    assert p.decide(detect("We discussed about it")).strategy is Strategy.NONE
    assert p.decide(detect("I am agree")).strategy is Strategy.RECAST


def test_policy_tutor_and_high():
    t = CorrectionPolicy(sensitivity="high", tutor=True)
    assert t.decide(detect("We discussed about it")).strategy is Strategy.BRIEF
    assert t.decide(detect("I need some advices")).strategy is Strategy.BRIEF  # tutor cooldown is 1 turn
    h = CorrectionPolicy(sensitivity="high")
    assert h.decide(detect("I am agree")).strategy is Strategy.BRIEF  # important
    assert h.decide(detect("It's more easier")).strategy is Strategy.RECAST  # minor, first time


def test_policy_picks_one_issue_most_important():
    p = CorrectionPolicy()
    d = p.decide(detect("We discussed about it and I am agree"))
    assert d.issue.original == "I am agree"
