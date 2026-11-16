"""Fast, conservative rule-based detector for common learner errors.

It runs in well under a millisecond, so it can steer the *current* reply (via a
coach note) without adding latency. Every issue carries the exact span of the
user's words it is based on, so feedback is always grounded in the transcript.

Precision matters more than recall here: a false correction is worse than a
missed one. The LLM also notices errors on its own when it recasts.

Caveat: Whisper's language model sometimes "fixes" grammar while transcribing,
so some real mistakes never reach the transcript.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

IRREGULAR_PAST = {
    "be": "was", "begin": "began", "bring": "brought", "buy": "bought", "catch": "caught", "choose": "chose",
    "come": "came", "do": "did", "drink": "drank", "drive": "drove", "eat": "ate", "fall": "fell", "feel": "felt",
    "find": "found", "fly": "flew", "forget": "forgot", "get": "got", "give": "gave", "go": "went", "have": "had",
    "hear": "heard", "keep": "kept", "know": "knew", "leave": "left", "lose": "lost", "make": "made", "meet": "met",
    "pay": "paid", "put": "put", "read": "read", "ride": "rode", "run": "ran", "say": "said", "see": "saw",
    "sell": "sold", "send": "sent", "sing": "sang", "sit": "sat", "sleep": "slept", "speak": "spoke",
    "spend": "spent", "stand": "stood", "swim": "swam", "take": "took", "teach": "taught", "tell": "told",
    "think": "thought", "understand": "understood", "wake": "woke", "wear": "wore", "win": "won", "write": "wrote",
}
REGULAR = {
    "arrive", "ask", "call", "change", "clean", "cook", "dance", "decide", "enjoy", "finish", "help", "like",
    "listen", "live", "look", "love", "miss", "move", "need", "open", "pick", "plan", "play", "rain", "start",
    "stay", "stop", "talk", "travel", "try", "visit", "walk", "want", "watch", "work",
}
PAST_TO_BASE = {v: k for k, v in IRREGULAR_PAST.items() if v not in ("put", "read")}


def regular_past(verb: str) -> str:
    if verb.endswith("e"):
        return verb + "d"
    if verb.endswith("y") and verb[-2] not in "aeiou":
        return verb[:-1] + "ied"
    if verb in ("stop", "plan"):
        return verb + verb[-1] + "ed"
    if verb == "travel":
        return "travelled"  # British spelling
    return verb + "ed"


def past_of(verb: str) -> str:
    return IRREGULAR_PAST.get(verb) or regular_past(verb)


PAST_MARKERS = re.compile(
    r"\b(yesterday|last (?:night|week|weekend|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday|time)"
    r"|\d+\s+\w+\s+ago|a (?:week|month|year|while) ago|ago)\b", re.I)
HABITUAL = re.compile(r"\b(every|usually|always|normally|often|sometimes|each)\b", re.I)
SUBJECT = r"(?:I|we|they|you|he|she|my (?:son|daughter|wife|husband|kids|children|friend|boss|mum|dad))"
VERBS = sorted(set(IRREGULAR_PAST) | REGULAR, key=len, reverse=True)
TENSE_RE = re.compile(rf"\b({SUBJECT})\s+({'|'.join(VERBS)})\b", re.I)


@dataclass
class Issue:
    category: str
    original: str  # exact span from the user's words
    suggestion: str  # corrected span
    explanation: str
    severity: str = "minor"  # minor | important
    source: str = "rules"

    def to_dict(self) -> dict:
        return asdict(self)


def _match_case(src: str, repl: str) -> str:
    return repl[0].upper() + repl[1:] if src[:1].isupper() and repl else repl


_AUX_BEFORE = {"did", "does", "will", "can", "could", "would", "should", "might", "must", "let", "to", "help",
               "make", "made", "see", "saw", "hear", "heard", "if", "that", "didn't", "doesn't", "won't", "would'nt", "do"}


def _past_tense(text: str) -> list[Issue]:
    issues: list[Issue] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if not PAST_MARKERS.search(sentence) or HABITUAL.search(sentence):
            continue
        for m in TENSE_RE.finditer(sentence):
            subj, verb = m.group(1), m.group(2).lower()
            before = re.findall(r"[a-z']+", sentence[: m.start()].lower())
            if before and before[-1] in _AUX_BEFORE:  # "Did he go…", "Will you take…"
                continue
            # e.g. "Yesterday I want to tell you…" is often fine in speech for some verbs; skip states.
            if verb in ("like", "love", "want", "need", "know", "understand", "think", "have", "be"):
                continue
            after = sentence[m.end():].lower()
            past = past_of(verb)
            if verb == "bring" and re.match(r"\s+\w+(?:\s+\w+)?\s+to\s+(school|work|nursery|the)", after):
                past = "took"  # "bring my son to school" from home -> "took"
            issues.append(Issue(
                "past tense", m.group(0), f"{subj} {past}",
                f"Use the past simple for finished past time: '{subj} {past}'.",
                "important" if verb in IRREGULAR_PAST else "minor"))
    return issues


# (pattern, category, replacement (str or callable), explanation, severity)
_RULES: list[tuple[str, str, object, str, str]] = [
    (r"\b(did ?n[o']?t|did not)\s+(" + "|".join(PAST_TO_BASE) + r")\b", "past tense",
     lambda m: f"{m.group(1)} {PAST_TO_BASE[m.group(2).lower()]}",
     "After 'didn't', use the base verb.", "important"),
    (r"\b(I|we|they|you)\s+(?:am|are|'m|'re)\s+agree\b", "verb pattern",
     lambda m: f"{m.group(1)} agree", "'Agree' is a verb: 'I agree', not 'I am agree'.", "important"),
    (r"\bdiscuss(?:ed|ing)?\s+about\b", "verb pattern",
     lambda m: m.group(0).rsplit(" ", 1)[0], "'Discuss' takes a direct object: 'discuss the plan'.", "minor"),
    (r"\bexplain(?:ed|s)?\s+(me|him|her|us|them)\b", "verb pattern",
     lambda m: f"{m.group(0).split()[0]} to {m.group(1)}", "Use 'explain to me', not 'explain me'.", "minor"),
    (r"\bsay(?:s)?\s+(me|him|her|us|them)\b", "word choice",
     lambda m: f"tell {m.group(1)}", "Use 'tell me' (or 'say to me'), not 'say me'.", "minor"),
    (r"\bsince\s+(\d+|a|one|two|three|four|five|six|seven|eight|nine|ten)\s+(years?|months?|weeks?|days?|hours?)\b",
     "preposition", lambda m: f"for {m.group(1)} {m.group(2)}",
     "Use 'for' with a length of time, 'since' with a starting point.", "important"),
    (r"\b(informations|advices|furnitures|equipments|homeworks|feedbacks|luggages|knowledges)\b", "uncountable noun",
     lambda m: m.group(1)[:-1], "This noun is uncountable in English, so it has no plural.", "minor"),
    (r"\bmore\s+(better|worse|easier|harder|bigger|smaller|faster|cheaper|nicer)\b", "comparative",
     lambda m: m.group(1), "Don't add 'more' to a comparative that already ends in -er.", "minor"),
    (r"\b(he|she|it)\s+don'?t\b", "agreement",
     lambda m: f"{m.group(1)} doesn't", "With he/she/it, use 'doesn't'.", "important"),
    (r"\bcan\s+able\s+to\b", "verb pattern", "can", "Use 'can' or 'be able to', not both.", "minor"),
    (r"\b(make|making|made)\s+(a|some)\s+(photos?|pictures?)\b", "collocation",
     lambda m: {"make": "take", "making": "taking", "made": "took"}[m.group(1).lower()] + f" {m.group(2)} {m.group(3)}",
     "In English you 'take' a photo.", "minor"),
    (r"\b(do|did|doing)\s+(a|the)\s+mistakes?\b", "collocation",
     lambda m: {"do": "make", "did": "made", "doing": "making"}[m.group(1).lower()] + m.group(0)[len(m.group(1)):],
     "You 'make' a mistake.", "minor"),
    (r"^how to say\b", "question form", "how do you say", "Use 'How do you say…?' for questions.", "minor"),
    (r"\bI\s+very\s+(like|love|enjoy|want)\b", "word order",
     lambda m: f"I really {m.group(1)}", "'Very' can't go before a verb; try 'I really like…'.", "minor"),
    (r"(?<!make\s)(?<!making\s)\bthe most of\b", "determiner", "most of",
     "Use 'most of the…' without 'the' before 'most'.", "minor"),
    (r"\bpeople\s+(is|was|has)\b", "agreement",
     lambda m: "people " + {"is": "are", "was": "were", "has": "have"}[m.group(1).lower()],
     "'People' is plural.", "important"),
    (r"\b(I|he|she|my son|my daughter|it)\s+(?:have|has)\s+(\d+)\s+years(?:\s+old)?\b", "verb pattern",
     lambda m: f"{m.group(1)} {'am' if m.group(1) == 'I' else 'is'} {m.group(2)}",
     "For age, use 'be': 'I'm 35', 'she's 7'.", "important"),
    (r"\b(go|went|going|come|came|coming)\s+to\s+home\b", "preposition",
     lambda m: f"{m.group(1)} home", "No 'to' before 'home': 'go home'.", "minor"),
    (r"\b(looking|look|looks)\s+forward\s+to\s+(see|meet|hear|go|have|work)\b", "verb pattern",
     lambda m: f"{m.group(1)} forward to {_ing(m.group(2))}", "'Look forward to' is followed by -ing.", "minor"),
    (r"\baccording to me\b", "phrase", "in my opinion", "Say 'in my opinion' or 'I think'.", "minor"),
]


def _third_person(v: str) -> str:
    v = v.lower()
    return {"have": "has", "go": "goes", "do": "does"}.get(v, v + "s")


def _ing(v: str) -> str:
    v = v.lower()
    if v.endswith("e") and v not in ("see",):
        return v[:-1] + "ing"
    return v + "ing"


_COMPILED = [(re.compile(p, re.I), c, r, e, s) for p, c, r, e, s in _RULES]


_THIRD_RE = re.compile(r"\b(he|she)\s+(have|go|want|like|need|work|live|know|do)\b(?!\s+n[o']?t\b)", re.I)


def _third_person_rule(text: str) -> list[Issue]:
    out = []
    for m in _THIRD_RE.finditer(text):
        before = re.findall(r"[a-z']+", text[: m.start()].lower())
        if before and before[-1] in _AUX_BEFORE:
            continue
        out.append(Issue("agreement", m.group(0), _match_case(m.group(0), f"{m.group(1)} {_third_person(m.group(2))}"),
                         "With he/she, add -s: 'she likes', 'he has'.", "important"))
    return out


def detect(text: str) -> list[Issue]:
    issues = _past_tense(text) + _third_person_rule(text)
    for rx, cat, repl, expl, sev in _COMPILED:
        for m in rx.finditer(text.strip()):
            fix = repl(m) if callable(repl) else repl
            issues.append(Issue(cat, m.group(0), _match_case(m.group(0), fix), expl, sev))
    # de-duplicate overlapping spans (keep the first)
    seen: set[str] = set()
    out = []
    for i in issues:
        key = i.original.lower()
        if key not in seen:
            seen.add(key)
            out.append(i)
    return out


def apply_corrections(text: str, issues: list[Issue]) -> str:
    out = text
    for i in issues:
        out = out.replace(i.original, i.suggestion, 1)
    return out
