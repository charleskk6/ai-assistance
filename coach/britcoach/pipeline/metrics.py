"""Per-turn latency metrics and system memory sampling."""

from __future__ import annotations

import os
import statistics
from dataclasses import asdict, dataclass, field


def ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 1)


@dataclass
class TurnMetrics:
    turn_id: int = 0
    kind: str = "user"
    vad_frame_ms: float | None = None  # mean VAD inference time per 32 ms frame
    eot_ms: float | None = None  # last voiced frame -> end-of-turn decision
    stt_wait_ms: float | None = None  # time on the critical path waiting for the final transcript
    stt_compute_ms: float | None = None  # engine compute time for the final transcript
    stt_speculative_hit: bool | None = None
    stt_audio_s: float | None = None
    llm_ttft_ms: float | None = None
    llm_tokens: int | None = None
    llm_tokens_per_s: float | None = None
    llm_prompt_tokens: int | None = None
    llm_reused_prefix_tokens: int | None = None
    first_sentence_ms: float | None = None  # LLM request -> first speakable chunk
    tts_first_audio_ms: float | None = None  # first chunk -> its audio synthesised
    tts_rtf: float | None = None  # synthesis time / audio duration (first chunk)
    e2e_server_ms: float | None = None  # end of user speech -> first audio frame sent
    e2e_audible_ms: float | None = None  # end of user speech -> client reports playback started
    bargein_detect_ms: float | None = None  # first voiced frame -> barge-in confirmed
    bargein_cancel_ms: float | None = None  # barge-in confirmed -> stop sent and work cancelled
    bargein_client_stop_ms: float | None = None  # stop received -> audio silenced (client clock)
    echo_rejected: int = 0
    rss_mb: float | None = None
    sys_available_gb: float | None = None
    llm_peak_memory_gb: float | None = None
    cancelled: bool = False
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


def memory_snapshot() -> dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"rss_mb": round(psutil.Process(os.getpid()).memory_info().rss / 2**20, 1),
                "sys_available_gb": round(vm.available / 2**30, 2), "sys_total_gb": round(vm.total / 2**30, 2),
                "sys_percent": vm.percent}
    except Exception:  # pragma: no cover
        return {}


SUMMARY_KEYS = ["eot_ms", "stt_wait_ms", "stt_compute_ms", "llm_ttft_ms", "llm_tokens_per_s", "first_sentence_ms",
                "tts_first_audio_ms", "e2e_server_ms", "e2e_audible_ms", "bargein_detect_ms", "bargein_cancel_ms",
                "bargein_client_stop_ms", "vad_frame_ms"]


def summarise(rows: list[dict]) -> dict:
    out = {}
    for k in SUMMARY_KEYS:
        vals = sorted(r[k] for r in rows if r.get(k) is not None)
        if vals:
            out[k] = {"n": len(vals), "median": round(statistics.median(vals), 1),
                      "p90": round(vals[min(len(vals) - 1, int(0.9 * len(vals)))], 1),
                      "min": round(vals[0], 1), "max": round(vals[-1], 1)}
    return out
