"""Conversation state: history, topic memory, level estimation and context building."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from . import prompts
from .modes import MODES, SCENARIOS

TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "school": ("school", "teacher", "class", "pickup", "pick up", "homework", "nursery", "reception", "year two"),
    "children's activities": ("swimming", "football", "club", "party", "parties", "half term", "playdate", "ballet"),
    "work": ("work", "office", "job", "boss", "colleague", "meeting", "commute", "manager"),
    "software": ("code", "coding", "bug", "deploy", "python", "java", "sprint", "standup", "pull request", "kubernetes", "api"),
    "weekend": ("weekend", "saturday", "sunday", "day out", "trip"),
    "family": ("son", "daughter", "kids", "children", "wife", "husband", "parents", "family"),
    "food": ("cook", "dinner", "lunch", "restaurant", "takeaway", "roast", "curry", "tea"),
    "weather": ("rain", "raining", "sunny", "cold", "weather", "warm", "snow"),
    "neighbours": ("neighbour", "street", "bins", "parcel", "garden", "council"),
    "british culture": ("pub", "queue", "telly", "bbc", "football match", "british", "england", "london"),
    "travel": ("holiday", "travel", "flight", "train", "abroad", "visit"),
    "hobbies": ("hobby", "read", "running", "gym", "music", "film", "series", "game"),
    "health": ("doctor", "gp", "ill", "sick", "appointment", "dentist"),
}


def detect_topic(text: str) -> str | None:
    t = f" {text.lower()} "
    best, score = None, 0
    for topic, kws in TOPIC_KEYWORDS.items():
        s = sum(1 for k in kws if re.search(rf"\b{re.escape(k)}\b", t))
        if s > score:
            best, score = topic, s
    return best


def extract_questions(text: str) -> list[str]:
    return [q.strip() for q in re.findall(r"[^.!?]*\?", text) if len(q.strip()) > 3]


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str
    ts: float = field(default_factory=time.time)
    interrupted: bool = False
    synthetic: bool = False  # app-generated user turns (opening, idle nudge)
    issues: list = field(default_factory=list)
    note: str = ""  # coach note attached to this user turn (sent to the LLM only)


@dataclass
class ConversationState:
    mode: str = "free"
    scenario: str = "open"
    difficulty: str = "normal"
    level: str = "B2"
    max_messages: int = 14
    char_budget: int = 6000
    turns: list[Turn] = field(default_factory=list)
    topic: str | None = None
    topics: list[str] = field(default_factory=list)
    recent_questions: list[str] = field(default_factory=list)
    _auto_level: bool = True

    # ----------------------------------------------------------------- settings
    def configure(self, mode: str | None = None, scenario: str | None = None,
                  difficulty: str | None = None, level: str | None = None) -> None:
        if mode and mode in MODES:
            self.mode = mode
        if scenario and scenario in SCENARIOS:
            self.scenario = scenario
        if difficulty:
            self.difficulty = difficulty
        if level in ("B1", "B2", "C1"):
            self.level = level
            self._auto_level = False
        elif level == "auto":
            self._auto_level = True

    def system_prompt(self) -> str:
        return prompts.system_prompt(self.mode, self.scenario, self.level, self.difficulty)

    # ----------------------------------------------------------------- history
    def add_user(self, text: str, issues: list | None = None, note: str = "", synthetic: bool = False) -> Turn:
        turn = Turn("user", text, issues=list(issues or []), note=note, synthetic=synthetic)
        self.turns.append(turn)
        if not synthetic:
            topic = detect_topic(text)
            if topic and topic != self.topic:
                self.topic = topic
                if topic not in self.topics:
                    self.topics.append(topic)
            self._maybe_update_level()
        return turn

    def add_assistant(self, text: str, interrupted: bool = False) -> Turn:
        if interrupted and text:
            text = text.rstrip() + " —"
        turn = Turn("assistant", text, interrupted=interrupted)
        if text:
            self.turns.append(turn)
            self.recent_questions = (self.recent_questions + extract_questions(text))[-4:]
        return turn

    def pop_unanswered_user(self) -> Turn | None:
        """If the last turn is a real user turn with no reply (the reply was cancelled before it
        was heard), remove and return it so it can be merged with what the user says next."""
        if self.turns and self.turns[-1].role == "user" and not self.turns[-1].synthetic:
            return self.turns.pop()
        return None

    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user" and not t.synthetic]

    def short_answer_streak(self) -> int:
        n = 0
        for t in reversed(self.user_turns()):
            if len(t.text.split()) <= 3:
                n += 1
            else:
                break
        return n

    # ----------------------------------------------------------------- level
    def _maybe_update_level(self) -> None:
        users = self.user_turns()
        if not self._auto_level or len(users) < 4 or len(users) % 4:
            return
        recent = users[-4:]  # the block since the last re-estimate
        words = [w for t in recent for w in re.findall(r"[a-zA-Z']+", t.text.lower())]
        if not words:
            return
        avg_len = len(words) / len(recent)
        variety = len(set(words)) / len(words)
        errors_per_100 = 100 * sum(len(t.issues) for t in recent) / len(words)
        if avg_len >= 12 and errors_per_100 < 2.5 and variety > 0.55:
            self.level = "C1"
        elif avg_len < 5 or errors_per_100 > 8:
            self.level = "B1"
        else:
            self.level = "B2"

    # ----------------------------------------------------------------- context
    def build_messages(self, extra_note: str = "") -> list[dict]:
        """System prompt + the most recent turns that fit the budget. The newest user turn
        carries its coach note (plus ``extra_note``) so the cached prefix stays stable."""
        msgs: list[dict] = []
        used = 0
        for i, t in enumerate(reversed(self.turns)):
            content = t.text
            if i == 0 and t.role == "user":
                note = " ".join(n for n in (t.note, extra_note) if n)
                content = f"{t.text}\n\n{note}" if note else t.text
            if msgs and (len(msgs) >= self.max_messages or used + len(content) > self.char_budget):
                break
            msgs.append({"role": t.role, "content": content})
            used += len(content)
        msgs.reverse()
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        system = self.system_prompt()
        dropped = len(self.turns) - len(msgs)
        if dropped > 0 and self.topics:
            system += f"\n\nEarlier in this conversation you talked about: {', '.join(self.topics[-5:])}."
        return [{"role": "system", "content": system}, *msgs]

    def transcript(self) -> list[dict]:
        return [{"role": t.role, "text": t.text, "ts": t.ts, "interrupted": t.interrupted,
                 "issues": [i.to_dict() for i in t.issues]}
                for t in self.turns if not t.synthetic]
