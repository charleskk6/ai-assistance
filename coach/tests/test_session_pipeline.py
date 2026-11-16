"""End-to-end pipeline tests with mock engines and a synthetic microphone.

Speech frames are loud noise and silence is zeros; the energy VAD maps those to
p=1 / p=0, so turn-taking timing is deterministic.
"""

import asyncio

import pytest

from britcoach.audio.pcm import unpack_audio
from britcoach.llm.mock import MockLLM
from britcoach.pipeline.session import Session, word_overlap
from britcoach.stt.engines import MockSTT
from britcoach.tts.engines import MockTTS

from .conftest import Recorder, make_engines, make_settings, silence_frame, speech_frame, to_pcm, wait_for


async def speak(sess: Session, n_speech: int = 12, n_silence: int = 14, realtime: bool = False) -> None:
    for f in [speech_frame()] * n_speech + [silence_frame()] * n_silence:
        await sess.handle_audio(to_pcm([f]))
        await asyncio.sleep(0.032 if realtime else 0)


async def new_session(settings=None, llm=None, stt=None, tts=None, opening=False):
    settings = settings or make_settings()
    eng = await make_engines(settings, llm=llm, stt=stt, tts=tts)
    rec = Recorder()
    sess = Session(eng, settings, rec.send_json, rec.send_bytes)
    if not opening:
        sess.conv.add_assistant("Hello! How are you?")  # skip the generated opening line
    await sess.handle_json({"type": "start"})
    return sess, rec, eng


async def test_voice_turn_end_to_end():
    llm = MockLLM(replies=["Oh, you took your son to school yesterday. How did he get on?"])
    stt = MockSTT(["Yesterday I bring my son to school"])
    sess, rec, eng = await new_session(llm=llm, stt=stt)
    await speak(sess)
    await wait_for(lambda: rec.of("assistant_done"))
    final = rec.of("user_final")[0]
    assert final["text"] == "Yesterday I bring my son to school"
    assert final["correction"]["strategy"] == "recast"
    # the coach note reached the LLM, attached to the latest user message only
    sent = llm.requests[-1]
    assert "the natural form is \"I took\"" in sent[-1]["content"]
    # audio was sentence-streamed with the turn id, in order
    frames = [unpack_audio(a) for a in rec.audio]
    assert {f[0] for f in frames} == {final["turn_id"]}
    assert [f[3] for f in frames] == sorted(f[3] for f in frames)
    sentences = [m["text"] for m in rec.of("assistant_sentence")]
    assert sentences == ["Oh, you took your son to school yesterday.", "How did he get on?"]
    # client reports playback; metrics are emitted
    tid = final["turn_id"]
    await sess.handle_json({"type": "playback", "event": "started", "turn_id": tid})
    await sess.handle_json({"type": "playback", "event": "ended", "turn_id": tid})
    m = rec.of("metrics")[-1]["turn"]
    for k in ("eot_ms", "stt_wait_ms", "llm_ttft_ms", "tts_first_audio_ms", "e2e_server_ms", "e2e_audible_ms"):
        assert k in m, k
    assert m["stt_speculative_hit"] is True  # final transcript was started during the pause
    assert sess.state == "listening"
    assert [t.role for t in sess.conv.turns][-2:] == ["user", "assistant"]


async def test_opening_line_is_generated_and_hidden_from_transcript():
    llm = MockLLM(replies=["Hiya! How's your week going?"])
    sess, rec, _ = await new_session(llm=llm, opening=True)
    await wait_for(lambda: rec.of("assistant_done"))
    assert "start the conversation" in llm.requests[0][-1]["content"]
    assert [t["role"] for t in sess.conv.transcript()] == ["assistant"]


async def test_typed_input_uses_same_path():
    sess, rec, _ = await new_session(llm=MockLLM(replies=["Fair enough!"]))
    await sess.handle_json({"type": "text_input", "text": "I am agree"})
    await wait_for(lambda: rec.of("assistant_done"))
    assert rec.of("user_final")[0]["issues"][0]["suggestion"] == "I agree"


async def test_barge_in_stops_audio_first_and_cancels_generation():
    long_reply = " ".join(f"This is sentence number {i}." for i in range(40))
    llm = MockLLM(replies=[long_reply], token_delay_s=0.01)
    tts = MockTTS(delay_s=0.0, ms_per_char=60)
    sess, rec, _ = await new_session(llm=llm, stt=MockSTT(["Tell me a story", "Sorry, can I ask something"]), tts=tts)
    await speak(sess)
    await wait_for(lambda: rec.audio)  # coach has started talking
    first_tid = unpack_audio(rec.audio[0])[0]
    await sess.handle_json({"type": "playback", "event": "started", "turn_id": first_tid})
    await asyncio.sleep(0.05)
    assert sess.state == "speaking"
    n_audio_before = len(rec.audio)

    # The user talks over the coach.
    for _ in range(4):
        await sess.handle_audio(to_pcm([speech_frame()]))
    stop = rec.of("stop_audio")
    assert stop and stop[0]["turn_id"] == first_tid and stop[0]["reason"] == "barge-in"
    # stop_audio went out before anything else about the barge-in
    idx_stop = next(i for i, (k, m) in enumerate(rec.order) if k == "json" and m.get("type") == "stop_audio")
    later_audio = [m for k, m in rec.order[idx_stop:] if k == "bytes" and unpack_audio(m)[0] == first_tid]
    assert later_audio == [], "stale audio was sent after stop_audio"
    bi = rec.of("barge_in")[0]["metrics"]
    assert bi["bargein_cancel_ms"] < 50
    await asyncio.sleep(0.1)
    assert llm.cancelled_count == 1  # generation stopped early, not run to completion
    assert len(rec.audio) == n_audio_before
    # History keeps only what was heard, marked as interrupted.
    last_ai = [t for t in sess.conv.turns if t.role == "assistant"][-1]
    assert last_ai.interrupted and last_ai.text.endswith("—")
    assert "sentence number 39" not in last_ai.text

    # The user's interrupting speech becomes the next turn.
    await speak(sess, n_speech=4, n_silence=14)
    await wait_for(lambda: len(rec.of("user_final")) == 2)
    assert rec.of("user_final")[1]["text"] == "Sorry, can I ask something"
    await wait_for(lambda: len(rec.of("assistant_done")) == 1)


async def test_speech_while_thinking_merges_with_unanswered_utterance():
    llm = MockLLM(replies=["first reply should be cancelled", "Right, got it."], ttft_s=0.3)
    stt = MockSTT(["I was thinking that", "maybe we could go to the park"])
    sess, rec, _ = await new_session(llm=llm, stt=stt)
    await speak(sess)
    await wait_for(lambda: sess.state == "thinking")
    await speak(sess)  # user carries on before any audio was produced
    await wait_for(lambda: rec.of("assistant_done"))
    texts = [m["text"] for m in rec.of("user_final")]
    assert texts[-1] == "I was thinking that maybe we could go to the park"
    users = [t.text for t in sess.conv.user_turns()]
    assert users == ["I was thinking that maybe we could go to the park"]


async def test_speakers_mode_rejects_echo_without_cancelling():
    s = make_settings(barge_in_mode="speakers")
    reply = "Brilliant, that sounds like a lovely day out with the kids. " * 3
    stt = MockSTT(["We went to the beach", "lovely day out with the kids"])
    sess, rec, _ = await new_session(settings=s, llm=MockLLM(replies=[reply]), stt=stt)
    await speak(sess)
    await wait_for(lambda: rec.audio)
    tid = unpack_audio(rec.audio[0])[0]
    await sess.handle_json({"type": "playback", "event": "started", "turn_id": tid})
    for _ in range(8):  # our own voice leaking from the speakers into the mic
        await sess.handle_audio(to_pcm([speech_frame()]))
    await wait_for(lambda: [m for m in rec.of("duck") if m["on"] is False])
    assert rec.of("duck")[0]["on"] is True
    unduck = [m for m in rec.of("duck") if m["on"] is False][0]
    assert unduck["echo"] is True
    assert not rec.of("stop_audio")


async def test_speakers_mode_real_interruption_cancels():
    s = make_settings(barge_in_mode="speakers")
    reply = "Brilliant, that sounds like a lovely day out with the kids. " * 3
    stt = MockSTT(["We went to the beach", "hang on what time is it"])
    sess, rec, _ = await new_session(settings=s, llm=MockLLM(replies=[reply]), stt=stt)
    await speak(sess)
    await wait_for(lambda: rec.audio)
    tid = unpack_audio(rec.audio[0])[0]
    await sess.handle_json({"type": "playback", "event": "started", "turn_id": tid})
    for _ in range(8):
        await sess.handle_audio(to_pcm([speech_frame()]))
    await wait_for(lambda: rec.of("stop_audio"))
    assert rec.of("stop_audio")[0]["turn_id"] == tid


async def test_half_duplex_mode_ignores_mic_while_speaking():
    s = make_settings(barge_in_mode="off")
    sess, rec, _ = await new_session(settings=s, llm=MockLLM(replies=["A long answer. " * 20]))
    await sess.handle_json({"type": "text_input", "text": "hello"})
    await wait_for(lambda: rec.audio)
    tid = unpack_audio(rec.audio[0])[0]
    await sess.handle_json({"type": "playback", "event": "started", "turn_id": tid})
    await speak(sess)
    assert not rec.of("stop_audio") and len(rec.of("user_final")) == 1


async def test_new_turn_after_playback_finished_is_not_an_interruption():
    sess, rec, _ = await new_session(llm=MockLLM(replies=["Short.", "Second."]), stt=MockSTT(["hi", "again"]))
    await sess.handle_json({"type": "text_input", "text": "hi"})
    await wait_for(lambda: rec.of("assistant_done"))
    tid = rec.of("assistant_done")[0]["turn_id"]
    await sess.handle_json({"type": "playback", "event": "started", "turn_id": tid})
    await sess.handle_json({"type": "playback", "event": "ended", "turn_id": tid})
    await speak(sess)
    await wait_for(lambda: len(rec.of("assistant_done")) == 2)
    assert not rec.of("stop_audio") and not rec.of("assistant_interrupted")


async def test_idle_nudge_once():
    s = make_settings(idle_prompt_s=0.15)
    llm = MockLLM(replies=["Hi there.", "Still with me? How's your day been?"])
    sess, rec, _ = await new_session(settings=s, llm=llm)
    await sess.handle_json({"type": "text_input", "text": "hi"})
    await wait_for(lambda: rec.of("assistant_done"))
    tid = rec.of("assistant_done")[0]["turn_id"]
    await sess.handle_json({"type": "playback", "event": "ended", "turn_id": tid})
    await wait_for(lambda: len(rec.of("assistant_done")) == 2, timeout=2)
    assert "gone quiet" in llm.requests[-1][-1]["content"]


async def test_end_session_review_and_stop():
    sess, rec, _ = await new_session(llm=MockLLM(replies=["Oh lovely. How did he get on?"]))
    await sess.handle_json({"type": "text_input", "text": "Yesterday I bring my son to school"})
    await wait_for(lambda: rec.of("assistant_done"))
    await sess.handle_json({"type": "end_session", "enrich": True})
    review = rec.of("review")[0]["review"]
    assert review["mistakes"][0]["you_said"] == "I bring"
    assert review["enriched_by"] == "local-llm"
    assert sess.state == "idle"


def test_word_overlap():
    assert word_overlap("lovely day out", "What a lovely day out with the kids") == 1.0
    assert word_overlap("what time is it", "lovely day out") == 0.0
    assert word_overlap("", "x") == 0.0


@pytest.mark.parametrize("mode", ["headphones", "speakers", "off"])
async def test_config_message(mode):
    sess, rec, _ = await new_session()
    await sess.handle_json({"type": "config", "config": {"mode": "tutor", "barge_in_mode": mode, "end_of_turn_ms": 700,
                                                         "voice": "mock", "sensitivity": "high"}})
    cfg = rec.of("config")[-1]["config"]
    assert cfg["mode"] == "tutor" and cfg["barge_in_mode"] == mode and cfg["end_of_turn_ms"] == 700
    assert sess.policy.tutor and cfg["sensitivity"] == "high"


async def test_barge_in_after_reply_fully_generated_marks_interrupted():
    tts = MockTTS(ms_per_char=80)  # ~5 s of audio, so it is still playing when the user cuts in
    sess, rec, _ = await new_session(llm=MockLLM(replies=["Right then, how was the school run this morning?"]), tts=tts)
    await sess.handle_json({"type": "text_input", "text": "hello"})
    await wait_for(lambda: rec.of("assistant_done"))
    tid = rec.of("assistant_done")[0]["turn_id"]
    await sess.handle_json({"type": "playback", "event": "started", "turn_id": tid})
    for _ in range(4):
        await sess.handle_audio(to_pcm([speech_frame()]))
    assert rec.of("stop_audio")[0]["turn_id"] == tid
    last_ai = [t for t in sess.conv.turns if t.role == "assistant"][-1]
    assert last_ai.interrupted
