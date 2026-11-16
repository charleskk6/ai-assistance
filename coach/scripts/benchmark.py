#!/usr/bin/env python3
"""Reproducible latency benchmark. Run it on the target machine; nothing here is estimated.

Component benchmark (loads the engines configured in .env / COACH_* variables):
    .venv/bin/python scripts/benchmark.py components --runs 5
    .venv/bin/python scripts/benchmark.py components --profiles small medium large   # compare LLM sizes

End-to-end benchmark against a running server (./scripts/start.sh in another terminal):
    .venv/bin/python scripts/benchmark.py e2e --url ws://127.0.0.1:8765/ws --turns 6

Both write JSON to benchmarks/results/<host>-<mode>-<timestamp>.json and print a table.
Test speech is synthesised locally (macOS `say`, else espeak-ng, else the configured
TTS), so runs are repeatable without a microphone.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from britcoach.audio.pcm import FRAME_SAMPLES, resample_linear  # noqa: E402

UTTERANCES = [
    "Yesterday I bring my son to school and he was a bit nervous.",
    "I work as a software engineer and we have a big release next week.",
    "At the weekend we went to the park with the kids.",
    "My neighbour took in a parcel for me, which was really kind.",
    "I'm not sure what to do for half term this year.",
    "The weather has been terrible, it rained all day.",
]


def machine_info() -> dict:
    info = {"host": socket.gethostname(), "system": platform.system(), "release": platform.release(),
            "machine": platform.machine(), "python": platform.python_version()}
    if platform.system() == "Darwin":
        for key, name in (("machdep.cpu.brand_string", "cpu"), ("hw.memsize", "mem_bytes"), ("hw.model", "model")):
            try:
                info[name] = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
            except OSError:
                pass
        try:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
            info["power"] = "battery" if "Battery Power" in out else "ac"
        except OSError:
            pass
    else:
        try:
            info["cpu"] = next(line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name"))
        except Exception:  # noqa: BLE001
            pass
    try:
        import psutil

        info["mem_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
        info["cpu_count"] = psutil.cpu_count()
    except ImportError:
        pass
    return info


def synth_utterance(text: str) -> np.ndarray:
    """16 kHz float32 speech for ``text``."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "u.wav"
        if shutil.which("say"):
            subprocess.run(["say", "-v", "Daniel", "-o", str(path), "--file-format=WAVE", "--data-format=LEI16@16000", text], check=True)
        elif shutil.which("espeak-ng"):
            subprocess.run(["espeak-ng", "-v", "en-gb", "-s", "160", "-w", str(path), text], check=True)
        else:
            raise SystemExit("Need macOS `say` or espeak-ng to synthesise test speech.")
        with wave.open(str(path)) as w:
            sr = w.getframerate()
            a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    a = resample_linear(a, sr, 16000)
    # Trim leading/trailing silence so "end of speech" is the end of the array.
    frames = [float(np.sqrt(np.mean(a[i:i + 160] ** 2))) for i in range(0, len(a) - 160, 160)]
    voiced = [i for i, r in enumerate(frames) if r > 0.01]
    if voiced:
        a = a[voiced[0] * 160:(voiced[-1] + 1) * 160]
    return a


def stats(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    s = sorted(xs)
    return {"n": len(s), "median": round(statistics.median(s), 1), "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 1),
            "min": round(s[0], 1), "max": round(s[-1], 1)}


def word_accuracy(ref: str, hyp: str) -> float:
    import re

    r = re.findall(r"[a-z']+", ref.lower())
    h = set(re.findall(r"[a-z']+", hyp.lower()))
    return round(sum(1 for w in r if w in h) / len(r), 3) if r else 0.0


def print_table(title: str, rows: dict) -> None:
    print(f"\n{title}")
    print(f"{'metric':38} {'median':>9} {'p90':>9} {'min':>9} {'max':>9} {'n':>4}")
    for k, v in rows.items():
        if isinstance(v, dict) and "median" in v:
            print(f"{k:38} {v['median']:>9} {v['p90']:>9} {v['min']:>9} {v['max']:>9} {v['n']:>4}")


def save(mode: str, data: dict) -> Path:
    out = ROOT / "benchmarks" / "results" / f"{socket.gethostname()}-{mode}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2))
    return out


# ====================================================================== components
async def bench_components(args) -> None:
    from britcoach.config import get_settings
    from britcoach.conversation.prompts import system_prompt
    from britcoach.llm.base import CancelToken, GenParams
    from britcoach.llm.factory import create_llm
    from britcoach.pipeline.engines import Engines
    from britcoach.pipeline.metrics import memory_snapshot

    s = get_settings()
    eng = Engines(s)
    t0 = time.perf_counter()
    await eng.start(warmup=True)
    result: dict = {"mode": "components", "machine": machine_info(), "engines": eng.status,
                    "startup_s": round(time.perf_counter() - t0, 2), "settings": s.public_dict()}
    print(json.dumps(eng.status, indent=1))
    audio = {u: synth_utterance(u) for u in UTTERANCES}

    # --- VAD ---------------------------------------------------------------------------
    vad = eng.vad.clone()
    times = []
    for a in audio.values():
        vad.reset()
        for i in range(0, len(a) - FRAME_SAMPLES, FRAME_SAMPLES):
            t = time.perf_counter()
            vad.prob(a[i:i + FRAME_SAMPLES])
            times.append((time.perf_counter() - t) * 1000)
    result["vad_frame_ms"] = stats(times)

    # --- STT ----------------------------------------------------------------------------
    if eng.status.get("stt", {}).get("ok"):
        lat, rtf, acc = [], [], []
        for _ in range(args.runs):
            for text, a in audio.items():
                r = eng.stt.transcribe(a, final=True)
                lat.append(r.latency_s * 1000)
                rtf.append(r.latency_s / r.audio_s)
                acc.append(word_accuracy(text, r.text))
        result["stt_ms"] = stats(lat)
        result["stt_rtf_x1000"] = stats([x * 1000 for x in rtf])
        result["stt_word_accuracy"] = round(sum(acc) / len(acc), 3)

    # --- TTS -----------------------------------------------------------------------------
    if eng.status.get("tts", {}).get("ok"):
        first, rtf = [], []
        voice = eng.default_voice()
        for _ in range(args.runs):
            for text in ["Oh, you took your son to school yesterday.", "How did he get on?", "Fancy a cuppa after drop-off?"]:
                t = time.perf_counter()
                total = 0.0
                ft = None
                for chunk in eng.tts.synthesize(text, voice):
                    ft = ft or time.perf_counter() - t
                    total += chunk.duration_s
                first.append(ft * 1000)
                rtf.append((time.perf_counter() - t) / max(total, 1e-3))
        result["tts_first_audio_ms"] = stats(first)
        result["tts_rtf_x1000"] = stats([x * 1000 for x in rtf])

    # --- LLM (one or more profiles) -------------------------------------------------------
    async def bench_llm(llm, label: str) -> dict:
        sp = system_prompt("free", "school_gate", "B2")
        history = [{"role": "system", "content": sp}]
        ttft, tps, total, reused = [], [], [], []
        for i in range(args.runs):
            for text in UTTERANCES[:3]:
                msgs = history + [{"role": "user", "content": text}]
                t = time.perf_counter()
                first = None
                out = []
                st = {}
                async for c in llm.stream(msgs, GenParams(max_tokens=s.llm_max_tokens, temperature=0.7), CancelToken()):
                    if c.done:
                        st = c.stats or {}
                        break
                    first = first or time.perf_counter()
                    out.append(c.text)
                ttft.append(((first or time.perf_counter()) - t) * 1000)
                total.append((time.perf_counter() - t) * 1000)
                if st.get("tokens_per_s"):
                    tps.append(st["tokens_per_s"])
                if st.get("reused_prefix_tokens") is not None:
                    reused.append(st["reused_prefix_tokens"])
                history += [{"role": "user", "content": text}, {"role": "assistant", "content": "".join(out)}]
                history = [history[0]] + history[-8:]
        r = {"model": llm.model_id, "backend": llm.name, "llm_ttft_ms": stats(ttft), "llm_tokens_per_s": stats(tps),
             "llm_reply_total_ms": stats(total), "sample_reply": "".join(out)[:200], "memory": memory_snapshot()}
        if reused:
            r["reused_prefix_tokens"] = stats(reused)
        if getattr(llm, "last_stats", None):
            r["peak_memory_gb"] = llm.last_stats.get("peak_memory_gb")
        print_table(f"LLM {label}: {llm.model_id}", r)
        return r

    result["llm"] = {}
    if args.profiles:
        await eng.llm.unload() if eng.llm else None
        for prof in args.profiles:
            llm = create_llm(s, profile=prof)
            try:
                await llm.load()
                await llm.warmup()
                result["llm"][prof] = await bench_llm(llm, prof)
            except Exception as e:  # noqa: BLE001
                result["llm"][prof] = {"error": f"{type(e).__name__}: {e}"}
                print(f"LLM {prof} failed: {e}")
            finally:
                await llm.unload()
    elif eng.status.get("llm", {}).get("ok"):
        result["llm"][s.llm_profile] = await bench_llm(eng.llm, s.llm_profile)

    # Estimated budget for one turn, from the measured medians (not a measurement of a real turn):
    def med(d, k):
        return (d.get(k) or {}).get("median")

    first_llm = next(iter(result["llm"].values()), {})
    parts = {"end_of_turn_wait_ms": s.end_of_turn_ms, "stt_ms": med(result, "stt_ms"),
             "llm_ttft_ms": med(first_llm, "llm_ttft_ms"), "tts_first_audio_ms": med(result, "tts_first_audio_ms")}
    result["turn_budget_from_medians"] = {**parts, "note": "STT usually overlaps the end-of-turn wait (speculative transcription); "
                                          "first audio also waits for the first ~8-12 tokens. Use the e2e mode for the real number."}
    result["memory"] = memory_snapshot()
    print_table("Components", {k: v for k, v in result.items() if isinstance(v, dict)})
    print(f"\nSaved {save('components', result)}")
    eng.shutdown()


# ====================================================================== end-to-end
async def bench_e2e(args) -> None:
    import websockets

    audio = [synth_utterance(u) for u in UTTERANCES]
    rows, bargeins = [], []
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "start", "config": {"barge_in_mode": args.barge_in_mode}}))
        pending_audio: dict[int, float] = {}
        events: asyncio.Queue = asyncio.Queue()

        async def reader():
            async for msg in ws:
                now = time.perf_counter()
                if isinstance(msg, bytes):
                    tid = int.from_bytes(msg[:4], "little")
                    if tid not in pending_audio:
                        pending_audio[tid] = now
                        await events.put(("first_audio", tid, now))
                else:
                    m = json.loads(msg)
                    await events.put((m["type"], m, now))

        rtask = asyncio.create_task(reader())

        async def wait_for(kind: str, timeout: float = 60):
            end = time.perf_counter() + timeout
            while True:
                ev = await asyncio.wait_for(events.get(), max(0.01, end - time.perf_counter()))
                if ev[0] == kind:
                    return ev

        async def stream(a: np.ndarray, trailing_s: float = 1.5) -> float:
            """Send audio in real time; return the wall time the last speech frame was sent."""
            pad = np.zeros(int(16000 * trailing_s), np.float32)
            data = np.concatenate([np.zeros(8000, np.float32), a, pad])
            speech_end_idx = 8000 + len(a)
            t_end = None
            start = time.perf_counter()
            for n, i in enumerate(range(0, len(data) - FRAME_SAMPLES + 1, FRAME_SAMPLES)):
                await ws.send((data[i:i + FRAME_SAMPLES] * 32767).astype("<i2").tobytes())
                if t_end is None and i + FRAME_SAMPLES >= speech_end_idx:
                    t_end = time.perf_counter()
                await asyncio.sleep(max(0, start + (n + 1) * FRAME_SAMPLES / 16000 - time.perf_counter()))
            return t_end or time.perf_counter()

        # Opening line: let it finish.
        ev = await wait_for("assistant_done")
        await asyncio.sleep(1.0)
        for i in range(args.turns):
            a = audio[i % len(audio)]
            send_task = asyncio.create_task(stream(a))
            ev = await wait_for("first_audio", 60)
            tid, t_first = ev[1], ev[2]
            # Behave like the browser: report playback as soon as the first audio arrives.
            await ws.send(json.dumps({"type": "playback", "event": "started", "turn_id": tid}))
            t_speech_end = await send_task
            done = await wait_for("assistant_done", 60)
            rows.append({"turn": i, "client_e2e_first_audio_ms": round((t_first - t_speech_end) * 1000, 1),
                         "reply": done[1]["text"][:120]})
            if args.bargein and i % 2 == 1:
                # Interrupt this reply mid-playback with the next utterance.
                await asyncio.sleep(0.4)
                t_bi = time.perf_counter()
                b_task = asyncio.create_task(stream(audio[(i + 1) % len(audio)], trailing_s=1.0))
                stop = await wait_for("stop_audio", 10)
                bargeins.append({"client_speech_start_to_stop_ms": round((stop[2] - t_bi - 0.5) * 1000, 1)})
                await b_task
                await wait_for("assistant_done", 60)
            await ws.send(json.dumps({"type": "playback", "event": "ended", "turn_id": tid}))
            await asyncio.sleep(1.0)
        await ws.send(json.dumps({"type": "stop"}))
        rtask.cancel()

    import urllib.request

    http = args.url.replace("ws://", "http://").replace("/ws", "")
    server = json.loads(urllib.request.urlopen(f"{http}/api/metrics").read())
    health = json.loads(urllib.request.urlopen(f"{http}/api/health").read())
    result = {"mode": "e2e", "machine": machine_info(), "engines": health.get("engines"), "turns": rows,
              "client_e2e_first_audio_ms": stats([r["client_e2e_first_audio_ms"] for r in rows]),
              "bargein": bargeins, "server_summary": server["summary"], "memory": server.get("memory"),
              "note": "client_e2e_first_audio_ms = last speech frame sent -> first reply audio byte received. "
                      "It includes the configured end-of-turn silence (COACH_END_OF_TURN_MS)."}
    print_table("End-to-end (client clock)", {"client_e2e_first_audio_ms": result["client_e2e_first_audio_ms"]})
    print_table("Server-side stages", server["summary"])
    if bargeins:
        print("\nBarge-in (speech onset -> stop_audio received):", [b["client_speech_start_to_stop_ms"] for b in bargeins])
    print(f"\nSaved {save('e2e', result)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    c = sub.add_parser("components")
    c.add_argument("--runs", type=int, default=3)
    c.add_argument("--profiles", nargs="*", choices=["small", "medium", "large"])
    e = sub.add_parser("e2e")
    e.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    e.add_argument("--turns", type=int, default=6)
    e.add_argument("--barge-in-mode", default="headphones")
    e.add_argument("--no-bargein", dest="bargein", action="store_false")
    args = ap.parse_args()
    asyncio.run(bench_components(args) if args.mode == "components" else bench_e2e(args))


if __name__ == "__main__":
    main()
