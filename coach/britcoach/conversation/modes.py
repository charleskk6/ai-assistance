"""Learning modes and scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mode:
    id: str
    title: str
    description: str
    default_sensitivity: str  # correction sensitivity default for the mode


MODES: dict[str, Mode] = {m.id: m for m in [
    Mode("free", "Free Conversation", "Natural, spontaneous chat with gentle corrections.", "medium"),
    Mode("social", "British Social Practice", "School parents, neighbours, friends and colleagues.", "medium"),
    Mode("roleplay", "Role-play", "Act out a real-world situation at an adjustable difficulty.", "low"),
    Mode("tutor", "English Tutor", "More explicit corrections, vocabulary and alternatives.", "high"),
]}


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    modes: tuple[str, ...]
    setting: str  # used in the system prompt
    opener: str  # what the first line should be about
    role: str = ""  # role-play: who the coach plays
    keywords: tuple[str, ...] = field(default=())


SCENARIOS: dict[str, Scenario] = {s.id: s for s in [
    Scenario("open", "Anything goes", ("free", "tutor"),
             "a relaxed chat between friends about whatever comes up",
             "how the learner's day or week is going"),
    Scenario("school_gate", "School pickup", ("social", "roleplay", "free"),
             "chatting at the school gate at pickup time",
             "the school day, the children or the weather at pickup",
             role="another parent waiting at the school gate, a bit chatty",
             keywords=("school", "pickup", "teacher", "class", "homework", "nativity")),
    Scenario("kids_activities", "Children's activities", ("social", "free"),
             "chatting with another parent about clubs, swimming, birthday parties and half term",
             "after-school clubs or plans for half term",
             keywords=("club", "swimming", "football", "party", "half term")),
    Scenario("neighbours", "Neighbours & community", ("social", "roleplay"),
             "a friendly chat with a neighbour over the garden fence",
             "the street, the bins, a parcel, or something local",
             role="a friendly next-door neighbour who has just taken in a parcel for the learner",
             keywords=("neighbour", "street", "bins", "parcel", "council")),
    Scenario("workplace", "Workplace small talk", ("social", "free", "roleplay"),
             "small talk with a colleague in the office kitchen",
             "the weekend, the commute, or how busy work is",
             role="a friendly colleague making a cup of tea in the office kitchen",
             keywords=("office", "meeting", "commute", "colleague")),
    Scenario("software", "Software engineering", ("free", "tutor", "roleplay"),
             "a conversation between software engineers at a British tech company",
             "what the learner is working on at the moment",
             role="a senior engineer running a casual one-to-one or a sprint retro",
             keywords=("code", "deploy", "bug", "sprint", "standup", "review")),
    Scenario("culture", "British culture & daily life", ("free", "social"),
             "chatting about everyday life in Britain: food, telly, pubs, queues, the weather",
             "something British the learner has noticed or found funny",
             keywords=("pub", "tea", "telly", "queue", "weather")),
    Scenario("weekend", "Weekend & family plans", ("free", "social"),
             "a chat about weekend plans and family days out",
             "plans for the weekend",
             keywords=("weekend", "park", "trip", "family", "day out")),
    Scenario("gp", "Booking a GP appointment", ("roleplay",),
             "a phone call to a GP surgery to book an appointment",
             "the receptionist answering the phone",
             role="a busy but kind GP surgery receptionist"),
    Scenario("pub", "At the pub", ("roleplay",),
             "ordering drinks and food at a busy pub",
             "the bar staff greeting the learner",
             role="a cheerful member of bar staff at a busy local pub"),
    Scenario("returns", "Returning something in a shop", ("roleplay",),
             "returning a faulty item at a high-street shop",
             "the shop assistant asking how they can help",
             role="a shop assistant at the returns desk"),
]}

DIFFICULTIES = {
    "easy": "Speak slowly and simply, be very patient and helpful.",
    "normal": "Speak naturally, as a real person in this situation would.",
    "challenging": "Speak fast and idiomatically, add a small complication to the situation, as real life would.",
}


def scenarios_for(mode: str) -> list[Scenario]:
    return [s for s in SCENARIOS.values() if mode in s.modes]
