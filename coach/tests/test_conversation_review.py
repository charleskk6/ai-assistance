from britcoach.coaching.detector import detect
from britcoach.coaching.review import build_review, enrich_review, parse_llm_json
from britcoach.conversation.prompts import opening_note, system_prompt
from britcoach.conversation.state import ConversationState, detect_topic


def test_system_prompt_per_mode():
    sp = system_prompt("roleplay", "gp", "B2", "challenging")
    assert "receptionist" in sp and "Stay in character" in sp and "complication" in sp
    assert "quick tip" in system_prompt("tutor", "open", "B2")
    assert "don't point it out" in system_prompt("free", "open", "B2")
    assert "C1" in system_prompt("free", "open", "B2")
    assert "begin the role-play" in opening_note("roleplay", "pub")


def test_topic_tracking_and_change():
    c = ConversationState()
    c.add_user("My son started a new school and his teacher is lovely")
    assert c.topic == "school"
    c.add_user("At work we have a big sprint and a nasty bug to deploy")
    assert c.topic == "software" and c.topics == ["school", "software"]
    assert detect_topic("nothing in particular") is None


def test_notes_only_on_latest_user_message_and_system_stable():
    c = ConversationState()
    c.add_user("Hello there", note="[Coach note: a]")
    c.add_assistant("Hi! How are you?")
    c.add_user("Fine thanks", note="[Coach note: b]")
    msgs = c.build_messages()
    assert msgs[0]["role"] == "system"
    assert "[Coach note: b]" in msgs[-1]["content"]
    assert all("[Coach note: a]" not in m["content"] for m in msgs)
    assert c.build_messages()[0] == msgs[0]  # identical system prompt -> KV cache reuse


def test_context_is_trimmed_and_starts_with_user():
    c = ConversationState(max_messages=4)
    for i in range(10):
        c.add_user(f"user says thing {i} about the weekend")
        c.add_assistant(f"reply {i}?")
    c.add_user("last one")
    msgs = c.build_messages()
    assert len(msgs) <= 5 and msgs[1]["role"] == "user" and msgs[-1]["content"] == "last one"
    assert "Earlier in this conversation you talked about: weekend" in msgs[0]["content"]


def test_interrupted_reply_and_unanswered_merge():
    c = ConversationState()
    c.add_user("I think that")
    assert c.pop_unanswered_user().text == "I think that"
    c.add_user("Hello")
    c.add_assistant("Lovely to meet you", interrupted=True)
    assert c.turns[-1].text.endswith("—") and c.turns[-1].interrupted
    assert c.pop_unanswered_user() is None


def test_recent_questions_tracked():
    c = ConversationState()
    c.add_assistant("Nice. How was your weekend? Did you go anywhere?")
    assert c.recent_questions == ["How was your weekend?", "Did you go anywhere?"]


def test_level_adapts():
    c = ConversationState()
    for _ in range(4):
        c.add_user("Yes.")
    assert c.level == "B1"
    for text in ["Honestly, the commute has been dreadful lately because they're resurfacing the main road",
                 "We ended up taking the kids to the coast at the weekend, which was surprisingly lovely",
                 "My manager reckons the release will slip again, so I'm bracing myself for a hectic fortnight",
                 "I've been trying to get into cooking proper curries from scratch, with mixed results so far"]:
        c.add_user(text)
    assert c.level == "C1"


def _session_with_errors() -> ConversationState:
    c = ConversationState()
    for text in ["Yesterday I bring my son to school", "I am agree with the teacher",
                 "Last week we go to the park with the kids", "It was nice"]:
        c.add_user(text, detect(text))
        c.add_assistant("Oh lovely, how did he get on? Fancy a cuppa after drop-off?")
    return c


def test_review_is_grounded_and_complete():
    c = _session_with_errors()
    r = build_review(c)
    assert 1 <= len(r.mistakes) <= 3
    users = [t.text for t in c.user_turns()]
    for m in r.mistakes:
        assert any(m["you_said"] in u for u in users)
    assert r.mistakes[0]["category"] == "past tense" and r.mistakes[0]["times"] == 2
    assert len(r.expressions) == 5
    assert any(e["heard_today"] for e in r.expressions)  # "How did he get on?" was used by the coach
    assert r.alternatives[0]["natural"] == "Yesterday I took my son to school"
    assert r.practice and any("Pronunciation isn't assessed" in n for n in r.notes)
    assert r.stats["user_turns"] == 4


def test_review_rejects_invented_llm_items():
    c = _session_with_errors()
    r = build_review(c)
    data = parse_llm_json('noise {"mistakes": [{"original": "I has a dog", "correction": "I have a dog"},'
                          ' {"original": "It was nice", "correction": "It was lovely", "explanation": "more natural"}],'
                          ' "alternatives": [{"original": "I never said this", "natural": "x"},'
                          ' {"original": "It was nice", "natural": "It was really lovely, actually"}]} trailing')
    before = len(r.mistakes)
    enrich_review(r, c, data, "local-llm")
    assert r.rejected_llm_items == 2
    assert all(m["you_said"] != "I has a dog" for m in r.mistakes)
    assert len(r.mistakes) <= 3 and len(r.mistakes) >= before
    assert any(a["source"] == "local-llm" for a in r.alternatives)


def test_parse_llm_json_failure():
    assert parse_llm_json("not json at all") is None
    assert parse_llm_json("<think>{bad}</think> {\"mistakes\": []}") == {"mistakes": []}
