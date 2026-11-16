#!/usr/bin/env python3
"""Download and verify the local models.

    .venv/bin/python scripts/setup_models.py                 # recommended defaults
    .venv/bin/python scripts/setup_models.py --verify-only   # check availability + licences, download nothing
    .venv/bin/python scripts/setup_models.py --profile all --voices en_GB-cori-medium en_GB-alan-medium

What it fetches (into ./models unless noted):
  * Silero VAD ONNX   - extracted from the MIT-licensed `silero-vad` wheel on PyPI (no torch needed)
  * Whisper           - faster-whisper (CTranslate2) or whisper.cpp ggml weights from Hugging Face
  * Piper voices      - British English voices from rhasspy/piper-voices
  * LLM               - MLX weights into the Hugging Face cache (where mlx-lm looks), or `ollama pull`

Every Hugging Face repository is looked up first and its declared licence printed,
so nothing is assumed to exist.
"""

from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def catalog() -> dict:
    with open(ROOT / "config" / "models.toml", "rb") as fh:
        return tomllib.load(fh)


def hf_info(repo: str) -> tuple[bool, str]:
    """(available, licence-or-reason). Distinguishes 'does not exist' from 'could not check'."""
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        return False, "huggingface_hub is not installed (pip install -r requirements.txt)"
    try:
        info = HfApi().model_info(repo)
        lic = (info.card_data or {}).get("license") if info.card_data else None
        return True, str(lic or "see model card")
    except RepositoryNotFoundError:
        return False, "repository does not exist"
    except Exception as e:  # noqa: BLE001
        return False, f"could not check ({type(e).__name__}) - network or proxy problem?"


# ------------------------------------------------------------------ VAD
def setup_vad() -> None:
    dest = MODELS / "vad" / "silero_vad.onnx"
    if dest.exists():
        print(f"[vad] already present: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print("[vad] downloading the silero-vad wheel from PyPI (no dependencies)")
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "-q", "-d", tmp, "silero-vad"], check=True)
        wheel = glob.glob(f"{tmp}/silero_vad-*.whl")[0]
        with zipfile.ZipFile(wheel) as z, z.open("silero_vad/data/silero_vad.onnx") as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    print(f"[vad] Silero VAD (MIT) -> {dest}")


# ------------------------------------------------------------------ STT
def setup_stt(engine: str, model: str, verify_only: bool) -> None:
    cat = catalog()["stt"]
    if engine == "faster_whisper":
        repo = cat["faster_whisper"][model]
        ok, lic = hf_info(repo)
        print(f"[stt] {repo}: {'available' if ok else 'UNAVAILABLE'} ({lic})")
        if ok and not verify_only:
            from huggingface_hub import snapshot_download

            snapshot_download(repo, local_dir=MODELS / "stt" / f"faster-whisper-{model}")
            print(f"[stt] -> models/stt/faster-whisper-{model}")
    else:
        repo, fname = cat["whispercpp"]["repo"], cat["whispercpp"][model]
        ok, lic = hf_info(repo)
        print(f"[stt] {repo}/{fname}: {'available' if ok else 'UNAVAILABLE'} ({lic})")
        if ok and not verify_only:
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo, fname, local_dir=MODELS / "stt")
            print(f"[stt] -> models/stt/{fname}")


# ------------------------------------------------------------------ TTS
def setup_voices(voices: list[str], verify_only: bool) -> None:
    known = {v["id"]: v for v in catalog()["tts"]["piper_voices"]}
    ok, lic = hf_info("rhasspy/piper-voices")
    print(f"[tts] rhasspy/piper-voices: {'available' if ok else 'UNAVAILABLE'} (repo licence: {lic}; "
          "each voice's MODEL_CARD lists its dataset licence)")
    if not ok or verify_only:
        return
    from huggingface_hub import hf_hub_download

    out = MODELS / "tts"
    out.mkdir(parents=True, exist_ok=True)
    for vid in voices:
        if vid not in known:
            print(f"[tts] unknown voice {vid}; add it to config/models.toml")
            continue
        base = known[vid]["path"]
        for suffix in (".onnx", ".onnx.json"):
            p = hf_hub_download("rhasspy/piper-voices", f"{base}/{vid}{suffix}")
            shutil.copy(p, out / f"{vid}{suffix}")
        try:
            card = Path(hf_hub_download("rhasspy/piper-voices", f"{base}/MODEL_CARD")).read_text()
            lic_line = next((l for l in card.splitlines() if "licen" in l.lower()), "").strip()
        except Exception:  # noqa: BLE001
            lic_line = ""
        print(f"[tts] {vid} -> models/tts/{vid}.onnx {lic_line}")


# ------------------------------------------------------------------ LLM
def setup_llm(backend: str, profiles: list[str], verify_only: bool) -> None:
    cat = catalog()["llm"]["profiles"]
    for prof in profiles:
        p = cat[prof]
        if backend == "mlx":
            candidates = [p["mlx"], *p.get("mlx_alternatives", [])]
            chosen = None
            for repo in candidates:
                ok, lic = hf_info(repo)
                print(f"[llm:{prof}] {repo}: {'available' if ok else 'UNAVAILABLE'} (licence: {lic})")
                if ok:
                    chosen = repo
                    break
            if chosen and chosen != p["mlx"]:
                print(f"[llm:{prof}] primary id unavailable; set COACH_LLM_MODEL={chosen} in .env")
            if chosen and not verify_only:
                from huggingface_hub import snapshot_download

                snapshot_download(chosen, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.jinja"])
                print(f"[llm:{prof}] {chosen} downloaded to the Hugging Face cache")
        elif backend == "ollama":
            tag = p["ollama"]
            if verify_only:
                print(f"[llm:{prof}] would run: ollama pull {tag}")
            elif shutil.which("ollama"):
                subprocess.run(["ollama", "pull", tag], check=False)
            else:
                print(f"[llm:{prof}] Ollama not installed. `brew install ollama`, then `ollama pull {tag}`")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--profile", default="medium", choices=["small", "medium", "large", "all"])
    ap.add_argument("--llm-backend", default="mlx", choices=["mlx", "ollama", "none"])
    ap.add_argument("--stt", default="faster_whisper", choices=["faster_whisper", "whispercpp"])
    ap.add_argument("--stt-model", default="base.en", choices=["tiny.en", "base.en", "small.en"])
    ap.add_argument("--voices", nargs="*", default=["en_GB-cori-medium", "en_GB-northern_english_male-medium"])
    ap.add_argument("--vad-only", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        setup_vad()
    if args.vad_only:
        return
    setup_stt(args.stt, args.stt_model, args.verify_only)
    setup_voices(args.voices, args.verify_only)
    if not args.skip_llm and args.llm_backend != "none":
        profiles = ["small", "medium", "large"] if args.profile == "all" else [args.profile]
        setup_llm(args.llm_backend, profiles, args.verify_only)
    print("\nDone." + ("" if args.verify_only else " Start the app with ./scripts/start.sh"))


if __name__ == "__main__":
    main()
