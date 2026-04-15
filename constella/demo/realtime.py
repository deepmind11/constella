"""Realtime voice demo — auto-commit on silence via fastrtc + Cartesia streaming TTS.

Per-turn pipeline:
    mic → fastrtc VAD → Groq Whisper → run_turn() (constellation) → Cartesia → speaker

Unlike the push-to-talk Gradio demo in `app.py`, this:
  - detects end-of-utterance automatically (no click-to-stop)
  - starts Ana's voice ~1 s after the patient's last word instead of ~2.5 s
  - uses Cartesia Sonic-3 (multilingual) in place of gTTS for natural prosody

Usage:
    constella-realtime                # localhost:7860, open in browser
    constella-realtime --port 8080
"""
from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from constella.orchestrator import run_turn
from constella.realtime.audio import cartesia_pcm_to_numpy, numpy_to_wav_tempfile
from constella.realtime.tts import SAMPLE_RATE as TTS_SAMPLE_RATE, stream_tts
from constella.schemas import ConversationState, PatientContext

log = logging.getLogger(__name__)

DEFAULT_PATIENT = (
    Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "patient_maria.json"
)

# One ConversationState per Python process. Good enough for a single-user demo;
# a prod deployment would key state by fastrtc session_id.
_state: ConversationState | None = None


def _load_patient() -> PatientContext:
    return PatientContext.model_validate_json(DEFAULT_PATIENT.read_text())


def _transcribe(wav_path: Path) -> str:
    """Groq Whisper. Batch — fastrtc only calls the handler on VAD pause, so
    the audio is already a complete utterance, typically 2-5 seconds."""
    from openai import OpenAI

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY required for ASR in the realtime demo.")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    with open(wav_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="text",
        )
    return str(resp).strip()


def _handler(audio: tuple[int, np.ndarray]) -> Iterator[tuple[int, np.ndarray]]:
    """fastrtc invokes this once per turn, with the full utterance since last pause."""
    global _state
    if _state is None:
        _state = ConversationState(patient=_load_patient())

    sample_rate, np_audio = audio
    wav_path = numpy_to_wav_tempfile(sample_rate, np_audio)
    try:
        patient_text = _transcribe(wav_path)
    finally:
        wav_path.unlink(missing_ok=True)

    if not patient_text:
        log.info("empty transcription, skipping turn")
        return

    log.info("patient: %s", patient_text)
    result = run_turn(_state, patient_text)
    _state = result.state
    log.info(
        "nurse [%s, %.0f ms]: %s",
        result.action.kind,
        result.total_latency_ms,
        result.nurse_text,
    )

    # Cartesia language hint: "mix" isn't a valid value, so drop to None
    # and let Sonic-3 auto-handle code-switched text.
    lang_tag = result.language.primary_lang if result.language else None
    cartesia_lang = lang_tag if lang_tag in ("en", "es") else None

    for pcm_chunk in stream_tts(result.nurse_text, language=cartesia_lang):
        yield cartesia_pcm_to_numpy(pcm_chunk, TTS_SAMPLE_RATE)


def build_stream():
    from fastrtc import ReplyOnPause, Stream

    return Stream(
        handler=ReplyOnPause(_handler),
        modality="audio",
        mode="send-receive",
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    stream = build_stream()
    stream.ui.launch(server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
