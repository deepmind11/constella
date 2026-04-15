"""Realtime voice demo — auto-commit on silence via fastrtc + Cartesia.

Per-turn pipeline:
    mic → fastrtc VAD → Groq Whisper → run_turn() → Cartesia streaming TTS → speaker

UI is a Gradio Blocks layout: mic + text input + examples on the left, Ana's
reply + running transcript on the right, specialist verdict JSON at the bottom.

Usage:
    constella-realtime                # localhost:7860
    constella-realtime --port 8080
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Literal

import gradio as gr
import numpy as np

from constella.orchestrator import run_turn
from constella.realtime.audio import cartesia_pcm_to_numpy, numpy_to_wav_tempfile
from constella.realtime.tts import SAMPLE_RATE as TTS_SAMPLE_RATE, stream_tts
from constella.schemas import ConversationState, PatientContext

log = logging.getLogger(__name__)

DEFAULT_PATIENT = (
    Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "patient_maria.json"
)

EXAMPLE_LINES = [
    "Hi, yes this is Maria.",
    "I checked my sugar this morning and it was 142.",
    "Hola, soy María. ¿Cómo está usted?",
    "Pues mi blood sugar esta mañana estaba en 165, no muy mal.",
    "Sí, tomé el metformin con breakfast pero el insulin lo olvidé last night.",
    "Tengo un dolor fuerte en el pecho y me duele el brazo izquierdo.",
]

_SPANISH_FUNCTION_WORDS = frozenset({
    "hola", "está", "sí", "bueno", "pero", "porque", "cómo", "qué",
    "dónde", "entonces", "muy", "mucho", "también", "señora", "usted",
    "soy", "eres", "tomó", "gracias", "bien", "tiene", "desde", "hospital",
    "medicamento", "azúcar", "insulina", "doctor", "doctora", "ahora",
    "poco", "ayer", "mañana", "preocupes", "quiero", "puedes",
})


def _load_patient() -> PatientContext:
    return PatientContext.model_validate_json(DEFAULT_PATIENT.read_text())


def _detect_reply_language(text: str) -> Literal["en", "es"] | None:
    """Heuristic language detection on Ana's reply text.

    We can't use the language specialist's verdict as the TTS hint because it
    runs on the PATIENT's utterance — Ana may reply in a different language
    (her prompt tells her to match the patient's register, which often means
    Spanish even when the patient said a greeting in English). Cartesia needs
    the language of the TEXT IT IS SPEAKING, which is Ana's reply.
    """
    if any(c in text for c in "ñ¿¡"):
        return "es"
    lower = text.lower()
    if any(c in lower for c in "áéíóú"):
        return "es"
    words = set(lower.split())
    if len(words & _SPANISH_FUNCTION_WORDS) >= 2:
        return "es"
    return "en"


def _transcribe(wav_path: Path) -> str:
    """Groq Whisper — batch, but fastrtc only calls us once per VAD pause so
    the clip is already short (typically 2-5 s)."""
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


def _format_transcript(state: ConversationState) -> str:
    if not state.history:
        return "_(start of call)_"
    lines = []
    for turn in state.history[-20:]:
        who = "**María:**" if turn.speaker == "patient" else "**Nurse Ana:**"
        lines.append(f"{who} {turn.text}")
    return "\n\n".join(lines)


def _build_verdict(result) -> str:
    verdict = {
        "action": result.action.kind,
        "latency_ms": {
            "primary": round(result.primary_latency_ms),
            "specialists_parallel": round(result.specialist_latency_ms),
            "total": round(result.total_latency_ms),
        },
        "language": result.language.model_dump() if result.language else None,
        "medication": result.medication.model_dump() if result.medication else None,
        "labs": result.labs.model_dump() if result.labs else None,
        "escalation": result.escalation.model_dump() if result.escalation else None,
    }
    return json.dumps(verdict, indent=2, default=str)


def _compute_turn(state: ConversationState | None, patient_text: str):
    state = state or ConversationState(patient=_load_patient())
    result = run_turn(state, patient_text)
    return result


def _voice_handler(audio, state_dict):
    """fastrtc handler — runs on every VAD pause."""
    from fastrtc import AdditionalOutputs

    state = (
        ConversationState.model_validate(state_dict)
        if state_dict
        else ConversationState(patient=_load_patient())
    )

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
    result = _compute_turn(state, patient_text)
    log.info(
        "nurse [%s, %.0f ms]: %s",
        result.action.kind,
        result.total_latency_ms,
        result.nurse_text,
    )

    reply_lang = _detect_reply_language(result.nurse_text)

    # Emit text updates first so the UI shows the reply while audio streams.
    yield AdditionalOutputs(
        result.nurse_text,
        _build_verdict(result),
        _format_transcript(result.state),
        result.state.model_dump(),
    )

    for pcm_chunk in stream_tts(result.nurse_text, language=reply_lang):
        yield cartesia_pcm_to_numpy(pcm_chunk, TTS_SAMPLE_RATE)


def _text_handler(patient_text: str, state_dict):
    """Text-only fallback path: no audio playback, just text updates."""
    if not patient_text or not patient_text.strip():
        return "", "{}", "_(start of call)_", state_dict

    state = (
        ConversationState.model_validate(state_dict)
        if state_dict
        else ConversationState(patient=_load_patient())
    )
    result = _compute_turn(state, patient_text.strip())
    return (
        result.nurse_text,
        _build_verdict(result),
        _format_transcript(result.state),
        result.state.model_dump(),
    )


def build_ui() -> gr.Blocks:
    from fastrtc import ReplyOnPause, WebRTC

    with gr.Blocks(title="Constella — bilingual constellation voice agent") as demo:
        gr.Markdown(
            "# Constella\n"
            "You are **María González** — 58-year-old diabetic patient, just out of hospital. "
            "**Nurse Ana** will follow up. Speak or type. English, Spanish, or both mid-sentence.\n\n"
            "> **Voice path:** click **Record** and speak; Ana answers the moment you pause.  \n"
            "> **Text path:** type below and press Enter."
        )

        state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### You (María)")
                webrtc = WebRTC(mode="send-receive", modality="audio", label="Speak")
                text_in = gr.Textbox(
                    placeholder="or type here...",
                    label="Text",
                    lines=2,
                )
                send_btn = gr.Button("Send text", variant="secondary", size="sm")
                gr.Examples(
                    examples=[[ex] for ex in EXAMPLE_LINES],
                    inputs=[text_in],
                    label="Try these",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Nurse Ana")
                nurse_text = gr.Textbox(
                    label="Latest reply",
                    lines=4,
                    interactive=False,
                )
                transcript = gr.Markdown(
                    value="_(start of call)_",
                    label="Conversation transcript",
                )

        verdict_out = gr.Code(
            label="What the 4 specialists saw (action + latency breakdown)",
            language="json",
            lines=18,
        )

        # Voice path — audio in, audio out, text/verdict/transcript as additional outputs.
        webrtc.stream(
            fn=ReplyOnPause(_voice_handler),
            inputs=[webrtc, state],
            outputs=[webrtc],
            time_limit=60,
        )
        webrtc.on_additional_outputs(
            lambda nt, v, t, s: (nt, v, t, s),
            outputs=[nurse_text, verdict_out, transcript, state],
            queue=False,
            show_progress="hidden",
        )

        # Text path (no audio).
        text_in.submit(
            _text_handler,
            inputs=[text_in, state],
            outputs=[nurse_text, verdict_out, transcript, state],
        )
        send_btn.click(
            _text_handler,
            inputs=[text_in, state],
            outputs=[nurse_text, verdict_out, transcript, state],
        )

    return demo


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    provider_override = os.environ.get("CONSTELLA_PROVIDER", "").lower()
    if os.environ.get("GROQ_API_KEY") and provider_override != "openrouter":
        log.warning(
            "Using Groq for LLM inference. Free-tier rate limits (6000 TPM on 8B) "
            "will cause multi-second backoff on realtime bursts — Constella fires 5 "
            "concurrent calls per turn. If you see latency > 3 s per turn, either "
            "set CONSTELLA_PROVIDER=openrouter in .env or upgrade to Groq Dev tier "
            "at https://console.groq.com/settings/billing."
        )
    if not os.environ.get("CARTESIA_API_KEY"):
        log.error(
            "CARTESIA_API_KEY is not set. The voice path will fail on the first "
            "TTS call. Add it to .env and restart."
        )
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    build_ui().launch(server_port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
