"""Curated everyday British English expressions, tagged by topic and level."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Expression:
    phrase: str
    meaning: str
    example: str
    topics: tuple[str, ...]
    level: str = "B2"
    pattern: str = ""  # regex to spot it in speech; defaults to the phrase itself

    def regex(self) -> re.Pattern:
        return re.compile(self.pattern or rf"\b{re.escape(self.phrase.lower())}\b", re.I)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("pattern")
        return d


PHRASES: list[Expression] = [
    Expression("How did he get on?", "How did it go for him?", "It was his first swimming lesson — how did he get on?",
               ("school", "children's activities", "family"), "B2", r"\bhow did (?:he|she|you|they|it) get on\b"),
    Expression("to drop someone off", "to take someone somewhere by car and leave them there",
               "I'll drop the kids off at school on my way to work.", ("school", "family"), "B2", r"\bdrop(?:ped|ping)?\b.*\boff\b"),
    Expression("to pick someone up", "to collect someone", "Who's picking them up today?", ("school", "family"), "B1",
               r"\bpick(?:ed|ing)?\b.*\bup\b"),
    Expression("half term", "the one-week school holiday in the middle of a term", "Are you going away at half term?",
               ("school", "children's activities")),
    Expression("a bit knackered", "quite tired (informal)", "I'm a bit knackered after the school run.",
               ("work", "family", "weekend"), "B2", r"\bknackered\b"),
    Expression("the school run", "taking children to and from school", "The school run was chaos this morning.",
               ("school",)),
    Expression("fancy (doing something)", "would like to", "Do you fancy a coffee after drop-off?",
               ("weekend", "food", "british culture"), "B2", r"\bfancy\b"),
    Expression("I'm not bothered", "I don't mind / it doesn't matter to me", "Pizza or curry? I'm not bothered.",
               ("food", "weekend"), "B2", r"\bnot bothered\b"),
    Expression("to have a lie-in", "to stay in bed later than usual", "I'm hoping to have a lie-in on Saturday.",
               ("weekend",), "B2", r"\blie-?in\b"),
    Expression("a day out", "a trip somewhere for the day", "We had a lovely day out at the seaside.", ("weekend", "family")),
    Expression("to pop round", "to visit someone's home briefly", "Pop round for a cuppa whenever you like.",
               ("neighbours", "weekend"), "B2", r"\bpop(?:ped|ping)? (?:round|over|in)\b"),
    Expression("a cuppa", "a cup of tea", "Fancy a cuppa?", ("british culture", "food", "neighbours")),
    Expression("to keep an eye on", "to watch or look after", "Could you keep an eye on the house while we're away?",
               ("neighbours", "family"), "B2", r"\bkeep(?:ing)? an eye on\b"),
    Expression("Cheers", "thanks (informal)", "Cheers for taking the parcel in!", ("neighbours", "british culture"), "B1",
               r"\bcheers\b"),
    Expression("to put the bins out", "to put rubbish bins on the street for collection", "Is it our turn to put the bins out?",
               ("neighbours",), "B2", r"\bbins? out\b"),
    Expression("it's chucking it down", "it's raining heavily", "Take a brolly — it's chucking it down.",
               ("weather",), "C1", r"\bchucking it down\b"),
    Expression("a bit nippy", "quite cold", "It's a bit nippy this morning, isn't it?", ("weather",), "B2", r"\bnippy\b"),
    Expression("mustn't grumble", "things are fine (modest reply to 'how are you')", "How's things? Oh, mustn't grumble.",
               ("british culture",), "C1"),
    Expression("to be snowed under", "to have too much work", "I'm snowed under this week with the release.",
               ("work", "software"), "C1", r"\bsnowed under\b"),
    Expression("to touch base", "to briefly talk to someone to check in", "Let's touch base after stand-up.",
               ("work", "software"), "B2", r"\btouch(?:ed|ing)? base\b"),
    Expression("to crack on", "to continue or start working on something", "Right, I'd better crack on with that ticket.",
               ("work", "software"), "C1", r"\bcrack(?:ed|ing)? on\b"),
    Expression("to get round to (something)", "to finally do something you intended to do",
               "I haven't got round to reviewing your PR yet.", ("work", "software"), "C1", r"\bg[eo]t(?:ten)? round to\b"),
    Expression("to be on the same page", "to agree / understand the same thing", "Let's make sure we're on the same page about scope.",
               ("work", "software"), "B2", r"\bon the same page\b"),
    Expression("a quick win", "something easy that brings a clear benefit", "Caching that endpoint would be a quick win.",
               ("software", "work"), "B2", r"\bquick wins?\b"),
    Expression("to flag something up", "to draw attention to a problem", "I'll flag it up in the retro.", ("work", "software"),
               "C1", r"\bflag(?:ged|ging)? (?:it |this |that )?up\b"),
    Expression("to sort something out", "to fix or organise something", "Don't worry, I'll sort it out.",
               ("work", "neighbours", "family"), "B2", r"\bsort(?:ed|ing)? (?:it |this |that |\w+ )?out\b"),
    Expression("to be up for (something)", "to be willing / keen to do something", "Are you up for a walk later?",
               ("weekend", "hobbies"), "B2", r"\bup for\b"),
    Expression("to have a go", "to try", "Go on, have a go!", ("hobbies", "children's activities"), "B2", r"\bha(?:ve|d) a go\b"),
    Expression("I reckon", "I think (informal)", "I reckon it'll be busy at the park.", ("weekend", "british culture"), "B2",
               r"\bi reckon\b"),
    Expression("to catch up", "to talk to someone you haven't seen for a while", "We must catch up properly soon.",
               ("weekend", "work"), "B1", r"\bcatch(?:ing)? up\b"),
    Expression("brilliant / lovely", "warm, very common British approval words", "Oh, lovely — that sounds brilliant.",
               ("british culture",), "B1", r"\b(?:brilliant|lovely)\b"),
    Expression("the telly", "the television (informal)", "Anything good on the telly tonight?", ("british culture", "hobbies"),
               "B1", r"\btelly\b"),
    Expression("to queue", "to wait in a line", "We queued for ages at the post office.", ("british culture",), "B1",
               r"\bqueu(?:e|ed|ing)\b"),
    Expression("a takeaway", "food bought from a restaurant to eat at home", "Shall we just get a takeaway tonight?",
               ("food",), "B1", r"\btakeaway\b"),
    Expression("to be under the weather", "to feel a bit ill", "She's a bit under the weather, so she's off school.",
               ("health", "family"), "C1", r"\bunder the weather\b"),
    Expression("to book an appointment", "to arrange a time to see someone", "I need to book an appointment with the GP.",
               ("health",), "B1", r"\bbook(?:ed|ing)? an appointment\b"),
    Expression("to get on with (someone)", "to have a good relationship", "Does he get on with the other kids in his class?",
               ("school", "work", "neighbours"), "B2", r"\bg[eo]t(?:s|ting)? on (?:well )?with\b"),
    Expression("to tag along", "to go somewhere with someone, informally", "Mind if I tag along to the park?",
               ("weekend", "children's activities"), "C1", r"\btag(?:ged|ging)? along\b"),
    Expression("to be a doddle", "to be very easy", "Don't worry, the form's a doddle.", ("work", "school"), "C1", r"\bdoddle\b"),
    Expression("fair enough", "that's reasonable / I accept that", "You'd rather stay in? Fair enough.",
               ("british culture", "work"), "B2", r"\bfair enough\b"),
    Expression("to be gutted", "to be very disappointed", "He was gutted when they lost the match.",
               ("children's activities", "hobbies"), "B2", r"\bgutted\b"),
    Expression("to take it easy", "to relax", "We're just taking it easy this weekend.", ("weekend",), "B1",
               r"\btak(?:e|ing) it easy\b"),
    Expression("a rough idea", "an approximate idea", "Can you give me a rough idea of the timeline?",
               ("work", "software"), "B2", r"\brough idea\b"),
]


def by_topic(topics: list[str]) -> list[Expression]:
    tset = set(topics)
    return [p for p in PHRASES if tset.intersection(p.topics)]


def find_in(text: str) -> list[Expression]:
    return [p for p in PHRASES if p.regex().search(text)]
