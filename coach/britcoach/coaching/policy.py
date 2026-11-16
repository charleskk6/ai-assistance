"""Correction policy: decide per turn whether to recast, give a brief tip, or stay quiet.

Strategy A (RECAST)  – minor mistakes: the coach reuses the correct form naturally.
Strategy B (BRIEF)   – important or repeated mistakes: one short tip, then carry on.
Strategy C (REVIEW)  – everything is logged for the end-of-session review; never
                       interrupts the conversation with detailed explanations.

At most one issue is acted on per turn, and brief tips have a cooldown so the
coach never nags.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from .detector import Issue

SEVERITY_RANK = {"important": 2, "minor": 1}


class Strategy(str, Enum):
    NONE = "none"
    RECAST = "recast"
    BRIEF = "brief"


@dataclass
class Decision:
    strategy: Strategy
    issue: Issue | None = None
    reason: str = ""


@dataclass
class CorrectionPolicy:
    sensitivity: str = "medium"  # off | low | medium | high
    tutor: bool = False
    brief_cooldown_turns: int = 3
    counts: Counter = field(default_factory=Counter)
    _turns_since_brief: int = 99

    def configure(self, sensitivity: str | None = None, tutor: bool | None = None) -> None:
        if sensitivity in ("off", "low", "medium", "high"):
            self.sensitivity = sensitivity
        if tutor is not None:
            self.tutor = tutor

    def decide(self, issues: list[Issue]) -> Decision:
        """Record ``issues`` (all of them, for the review) and pick a strategy for this turn."""
        self._turns_since_brief += 1
        for i in issues:
            self.counts[i.category] += 1
        if not issues:
            return Decision(Strategy.NONE, reason="no issues")
        if self.sensitivity == "off":
            return Decision(Strategy.NONE, reason="corrections off (logged for review)")

        issue = max(issues, key=lambda i: (SEVERITY_RANK.get(i.severity, 0), self.counts[i.category]))
        repeats = self.counts[issue.category]
        important = issue.severity == "important"

        if self.sensitivity == "low":
            if important:
                return Decision(Strategy.RECAST, issue, "low sensitivity: recast important issues only")
            return Decision(Strategy.NONE, reason="low sensitivity: minor issue logged")

        cooldown = 1 if self.tutor else self.brief_cooldown_turns
        can_brief = self._turns_since_brief >= cooldown
        if self.tutor:
            wants_brief = True
        elif self.sensitivity == "high":
            wants_brief = important or repeats >= 2
        else:  # medium
            wants_brief = repeats >= 3 and important

        if wants_brief and can_brief:
            self._turns_since_brief = 0
            return Decision(Strategy.BRIEF, issue, f"repeated {repeats}x" if repeats > 1 else "tutor/important")
        return Decision(Strategy.RECAST, issue, "natural recast")
