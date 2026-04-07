"""Speech recognition layer.

Default backend is **VibeVoice-ASR-7B** which natively handles code-switching
across 50+ languages with no language parameter. On Macs without a CUDA GPU,
falls back to distil-whisper-large-v3 for development convenience.

Production target: VibeVoice-ASR-7B on a single A10 or T4 in an NVIDIA Deep
Learning Container 24.10+.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

VIBEVOICE_REPO = Path(__file__).resolve().parents[1] / "external" / "VibeVoice"
ASR_BACKEND = os.environ.get("CONSTELLA_ASR_BACKEND", "auto")  # auto | vibevoice | distil_whisper


@dataclass
class TranscriptionResult:
    text: str
    backend: str
    latency_ms: float
    raw: dict | None = None


def transcribe(audio_path: Path) -> TranscriptionResult:
    backend = _select_backend()
    t0 = time.perf_counter()
    if backend == "vibevoice":
        text, raw = _transcribe_vibevoice(audio_path)
    else:
        text, raw = _transcribe_distil_whisper(audio_path)
    latency_ms = (time.perf_counter() - t0) * 1000
    return TranscriptionResult(text=text, backend=backend, latency_ms=latency_ms, raw=raw)


def _select_backend() -> str:
    if ASR_BACKEND in {"vibevoice", "distil_whisper"}:
        return ASR_BACKEND
    # auto: prefer VibeVoice if the repo is present and torch sees CUDA
    if VIBEVOICE_REPO.exists():
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "vibevoice"
        except ImportError:
            pass
    return "distil_whisper"


def _transcribe_vibevoice(audio_path: Path) -> tuple[str, dict]:
    """Call the VibeVoice ASR demo as a subprocess.

    TODO: refactor to import from vibevoice package directly once we read the
    actual Python API. For the 2-day sprint we shell out to the documented CLI.
    """
    if not VIBEVOICE_REPO.exists():
        raise RuntimeError(
            f"VibeVoice repo not found at {VIBEVOICE_REPO}. See SETUP.md step 4."
        )
    cmd = [
        "python",
        str(VIBEVOICE_REPO / "demo" / "vibevoice_asr_inference_from_file.py"),
        "--model_path",
        "microsoft/VibeVoice-ASR",
        "--audio_files",
        str(audio_path),
    ]
    log.info("vibevoice_asr cmd=%s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=VIBEVOICE_REPO)
    if proc.returncode != 0:
        log.error("vibevoice_asr failed: %s", proc.stderr[-500:])
        raise RuntimeError(f"vibevoice_asr exit={proc.returncode}")
    text = _extract_text_from_vibevoice_stdout(proc.stdout)
    return text, {"stdout_tail": proc.stdout[-500:], "stderr_tail": proc.stderr[-500:]}


def _extract_text_from_vibevoice_stdout(stdout: str) -> str:
    """The VibeVoice ASR demo prints structured output. We grep for the transcript line.

    TODO: replace with a proper parse once we read the demo source.
    """
    # Pragmatic placeholder: assume the demo prints "Transcript: <text>"
    for line in stdout.splitlines():
        if line.lower().startswith("transcript:"):
            return line.split(":", 1)[1].strip()
    # Fallback: last non-empty line
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _transcribe_distil_whisper(audio_path: Path) -> tuple[str, dict]:
    """Lightweight Mac-friendly fallback ASR."""
    try:
        from transformers import pipeline  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "distil-whisper backend requires transformers. "
            "Run `uv pip install transformers torch` or use `CONSTELLA_ASR_BACKEND=vibevoice`."
        ) from e
    pipe = _get_distil_whisper_pipe(pipeline)
    out = pipe(str(audio_path), return_timestamps=False)
    return out["text"].strip(), {"backend_model": "distil-whisper/distil-large-v3"}


_distil_pipe = None


def _get_distil_whisper_pipe(pipeline_factory):  # cached
    global _distil_pipe
    if _distil_pipe is None:
        log.info("loading distil-whisper-large-v3 (one-time download)")
        _distil_pipe = pipeline_factory(
            "automatic-speech-recognition",
            model="distil-whisper/distil-large-v3",
            chunk_length_s=15,
            batch_size=4,
        )
    return _distil_pipe


# Used by tests to verify ffmpeg is around when we need to convert audio
def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH; install via `brew install ffmpeg`.")
