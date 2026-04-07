"""Text-to-speech layer.

Default backend is **VibeVoice-Realtime-0.5B** which Microsoft reports as
running at realtime on NVIDIA T4 and Apple M4 Pro with ~300 ms first audible
latency. Spanish is one of the experimental multilingual voices added Dec 16,
2025.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

VIBEVOICE_REPO = Path(__file__).resolve().parents[1] / "external" / "VibeVoice"
TTS_BACKEND = os.environ.get("CONSTELLA_TTS_BACKEND", "vibevoice")
TTS_OUTPUT_DIR = Path(os.environ.get("CONSTELLA_TTS_OUTPUT_DIR", "/tmp/constella_tts"))

# Map (language, register) -> VibeVoice experimental speaker name
# These are placeholders pending verification with `download_experimental_voices.sh`.
SPEAKER_BY_LANG = {
    ("en", "formal"): "Carter",
    ("en", "informal"): "Carter",
    ("es", "formal"): "Sofia_es",
    ("es", "informal"): "Sofia_es",
    ("mix", "spanglish"): "Sofia_es",
}


@dataclass
class SynthesisResult:
    audio_path: Path
    backend: str
    latency_ms: float
    speaker: str
    language: str


def synthesize(
    text: str,
    *,
    language: Literal["en", "es", "mix"] = "en",
    register: Literal["formal", "informal", "spanglish"] = "formal",
    out_path: Path | None = None,
) -> SynthesisResult:
    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (TTS_OUTPUT_DIR / f"tts_{int(time.time() * 1000)}.wav")
    speaker = SPEAKER_BY_LANG.get((language, register), "Carter")
    t0 = time.perf_counter()
    if TTS_BACKEND == "vibevoice":
        _synth_vibevoice(text, speaker, out_path)
    else:
        raise NotImplementedError(f"unknown TTS backend: {TTS_BACKEND}")
    latency_ms = (time.perf_counter() - t0) * 1000
    return SynthesisResult(
        audio_path=out_path,
        backend=TTS_BACKEND,
        latency_ms=latency_ms,
        speaker=speaker,
        language=language,
    )


def _synth_vibevoice(text: str, speaker: str, out_path: Path) -> None:
    """Call the VibeVoice realtime demo as a subprocess.

    TODO: refactor to direct Python imports once we read demo/realtime_model_inference_from_file.py
    and pull out the model loader + inference function.
    """
    if not VIBEVOICE_REPO.exists():
        raise RuntimeError(
            f"VibeVoice repo not found at {VIBEVOICE_REPO}. See SETUP.md step 4."
        )
    # The demo script reads text from a file
    txt_path = out_path.with_suffix(".txt")
    txt_path.write_text(text, encoding="utf-8")
    cmd = [
        "python",
        str(VIBEVOICE_REPO / "demo" / "realtime_model_inference_from_file.py"),
        "--model_path",
        "microsoft/VibeVoice-Realtime-0.5B",
        "--txt_path",
        str(txt_path),
        "--speaker_name",
        speaker,
        "--output_path",
        str(out_path),  # NOTE: parameter name to be verified against demo source
    ]
    log.info("vibevoice_tts cmd=%s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=VIBEVOICE_REPO)
    if proc.returncode != 0:
        log.error("vibevoice_tts failed: %s", proc.stderr[-500:])
        raise RuntimeError(f"vibevoice_tts exit={proc.returncode}")
    if not out_path.exists():
        raise RuntimeError(f"VibeVoice did not produce {out_path}")
