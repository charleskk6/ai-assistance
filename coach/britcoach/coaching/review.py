"""End-of-session review (Strategy C).

Built in two layers:

1. A deterministic, always-grounded review from the rule detector, the phrasebank
   and transcript statistics. It needs no model and cannot invent mistakes.
2. Optional enrichment from an LLM (local by default, cloud only with explicit
   consent). Every item the LLM returns is *validated*: its quoted "original"
   must actually appear in something the learner said, otherwise it is dropped.

Pronunciation is never assessed: the transcript is text from speech recognition.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from ..conversation.state import ConversationState
from .detector import Issue, apply_corrections
from .phrasebank import PHRASES, by_topic, find_in

PRACTICE_BY_CATEGORY = {
    "past tense": "Tell a two-minute story about last weekend out loud, focusing on past forms (went, took, bought, saw). Record it and listen back.",
    "agreement": "Practise 'he/she' sentences about a friend's routine: 'She gets up at seven, she takes the bus…'.",
    "preposition": "Make five sentences with 'for' + a period ('for three years') and five with 'since' + a start point ('since 2019').",
    "verb pattern": "Write down the verb patterns you used today (agree, discuss, explain to) and say each in two new sentences.",
    "uncountable noun": "Practise 'some advice / a piece of information / a lot of homework' in short sentences.",
    "collocation": "Learn verb + noun pairs as chunks: take a photo, make a mistake, do homework, have a go.",
    "comparative": "Compare two places you know using -er/more without doubling them: 'easier', 'more expensive'.",
    "word order": "Use 'really' before verbs: 'I really like…', 'I really enjoyed…'.",
    "question form": "Ask five 'How do you say…?' and 'What do you call…?' questions in your next session.",
}
GENERAL_PRACTICE = [
    "Try shadowing: replay one of the coach's replies and say it at the same time, copying the rhythm.",
    "Next session, aim for longer answers: add a reason or a small story ('…because…', 'the funny thing was…').",
    "Pick two expressions from today's list and use each one in real life this week.",
]
PRONUNCIATION_NOTE = ("Pronunciation isn't assessed: this review is based only on the speech-recognition transcript, "
                      "which can't tell how something sounded.")


def _contains(haystack: str, needle: str) -> bool:
    h = " ".join(re.findall(r"[a-z0-9']+", haystack.lower()))
    n = " ".join(re.findall(r"[a-z0-9']+", needle.lower()))
    return bool(n) and n in h


@dataclass
class Review:
    mistakes: list[dict] = field(default_factory=list)
    expressions: list[dict] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)
    practice: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    enriched_by: str | None = None
    rejected_llm_items: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def build_review(state: ConversationState) -> Review:
    users = state.user_turns()
    assistant_text = " ".join(t.text for t in state.turns if t.role == "assistant")
    user_text = " ".join(t.text for t in users)
    r = Review()

    # --- mistakes: rank by severity, then how often the category occurred ---------------------
    all_issues: list[tuple[Issue, str]] = [(i, t.text) for t in users for i in t.issues]
    cat_counts = Counter(i.category for i, _ in all_issues)
    seen: set[str] = set()
    ranked = sorted(all_issues, key=lambda p: (p[0].severity != "important", -cat_counts[p[0].category]))
    for issue, sentence in ranked:
        if issue.category in seen:
            continue
        seen.add(issue.category)
        r.mistakes.append({"you_said": issue.original, "better": issue.suggestion, "explanation": issue.explanation,
                           "category": issue.category, "times": cat_counts[issue.category], "in_sentence": sentence,
                           "source": issue.source})
        if len(r.mistakes) == 3:
            break

    # --- expressions: ones the coach actually used, then topical ones; skip ones the user already uses
    used_by_user = {p.phrase for p in find_in(user_text)}
    picked: list = []
    for p in find_in(assistant_text) + by_topic(state.topics) + PHRASES:
        if p.phrase not in used_by_user and p not in picked:
            picked.append(p)
        if len(picked) == 5:
            break
    heard = {p.phrase for p in find_in(assistant_text)}
    r.expressions = [{**p.to_dict(), "heard_today": p.phrase in heard} for p in picked]

    # --- natural alternatives: corrected versions of sentences that had issues -----------------
    for t in users:
        if t.issues and len(r.alternatives) < 4:
            better = apply_corrections(t.text, t.issues)
            if better != t.text:
                r.alternatives.append({"original": t.text, "natural": better, "source": "rules"})

    # --- practice suggestions ------------------------------------------------------------------
    for cat, _ in cat_counts.most_common(3):
        if cat in PRACTICE_BY_CATEGORY:
            r.practice.append(PRACTICE_BY_CATEGORY[cat])
    avg_words = (sum(len(t.text.split()) for t in users) / len(users)) if users else 0
    for tip in GENERAL_PRACTICE:
        if len(r.practice) >= 4:
            break
        if "longer answers" in tip and avg_words >= 12:
            continue
        r.practice.append(tip)

    # --- stats ------------------------------------------------------------------------------------
    words = re.findall(r"[a-zA-Z']+", user_text.lower())
    r.stats = {
        "user_turns": len(users),
        "user_words": len(words),
        "unique_words": len(set(words)),
        "avg_words_per_turn": round(avg_words, 1),
        "issues_found": len(all_issues),
        "issues_per_100_words": round(100 * len(all_issues) / len(words), 1) if words else 0.0,
        "topics": state.topics,
        "estimated_level": state.level,
        "interruptions": sum(1 for t in state.turns if t.role == "assistant" and t.interrupted),
    }
    r.notes.append(PRONUNCIATION_NOTE)
    if not users:
        r.notes.append("No learner speech was recorded in this session.")
    return r


def review_messages(state: ConversationState) -> list[dict]:
    from ..conversation.prompts import REVIEW_SYSTEM

    lines = "\n".join(f"- {t.text}" for t in state.user_turns()[-30:])
    return [{"role": "system", "content": REVIEW_SYSTEM},
            {"role": "user", "content": f"Learner's lines:\n{lines}\n\nReturn the JSON now."}]


def parse_llm_json(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def enrich_review(review: Review, state: ConversationState, data: dict | None, source: str) -> Review:
    """Merge validated LLM findings into ``review``. Unsupported items are dropped and counted."""
    if not data:
        return review
    utterances = [t.text for t in state.user_turns()]

    def grounded(original: str) -> bool:
        return isinstance(original, str) and len(original.strip()) >= 2 and any(_contains(u, original) for u in utterances)

    existing = {m["you_said"].lower() for m in review.mistakes}
    for item in data.get("mistakes") or []:
        if not isinstance(item, dict):
            continue
        orig, corr = item.get("original", ""), item.get("correction", "")
        if not grounded(orig) or not corr or corr.strip().lower() == orig.strip().lower():
            review.rejected_llm_items += 1
            continue
        if orig.lower() in existing:
            continue
        if len(review.mistakes) < 3:
            review.mistakes.append({"you_said": orig, "better": corr, "explanation": item.get("explanation", ""),
                                    "category": "other", "times": 1, "in_sentence": next(u for u in utterances if _contains(u, orig)),
                                    "source": source})
            existing.add(orig.lower())

    have = {a["original"].lower() for a in review.alternatives}
    for item in data.get("alternatives") or []:
        if not isinstance(item, dict):
            continue
        orig, nat = item.get("original", ""), item.get("natural", "")
        if not grounded(orig) or not nat or nat.strip().lower() == orig.strip().lower():
            review.rejected_llm_items += 1
            continue
        if orig.lower() not in have and len(review.alternatives) < 6:
            review.alternatives.append({"original": orig, "natural": nat, "source": source})
            have.add(orig.lower())
    review.enriched_by = source
    return review
