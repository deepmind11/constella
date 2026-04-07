"""Gradio voice demo.

Voice pipeline:
    Mic → Groq Whisper-large-v3-turbo (ASR) → constellation → gTTS (TTS) → speaker

Text fallback also available (type as patient, read nurse reply).

VibeVoice end-to-end (the production story) is in asr.py / tts.py and requires
local GPU + repo clone per SETUP.md. This demo uses the API-only path so it
works on any machine with a Groq key.

Usage:
    constella-demo               # port 7860
    constella-demo --port 8080
    constella-demo --share       # public Gradio share URL
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path

import gradio as gr

from constella.orchestrator import run_turn
from constella.schemas import ConversationState, PatientContext

DEFAULT_PATIENT = (
    Path(__file__).resolve().parents[1] / "eval" / "scenarios" / "patient_maria.json"
)

log = logging.getLogger(__name__)


# ── ASR: Groq Whisper ────────────────────────────────────────────────────────

def _transcribe(audio_path: str) -> str:
    """Transcribe patient audio using Groq Whisper-large-v3-turbo.

    Uses GROQ_API_KEY directly (not the CONSTELLA_PROVIDER setting) because
    OpenRouter doesn't expose audio transcription endpoints.
    """
    from openai import OpenAI
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required for voice input even when using OpenRouter "
            "for LLM calls. Add it to .env."
        )
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="text",
        )
    # Groq returns the text directly as a string in "text" response_format
    return str(resp).strip()


# ── TTS: gTTS ────────────────────────────────────────────────────────────────

def _synthesize(text: str, lang: str) -> str | None:
    """Synthesize nurse audio using gTTS. Returns path to MP3 or None."""
    try:
        from gtts import gTTS
    except ImportError:
        log.warning("gTTS not installed — text-only output")
        return None
    lang_code = "es" if lang in ("es", "mix") else "en"
    tts = gTTS(text=text, lang=lang_code, slow=False)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(tmp.name)
    return tmp.name


# ── Patient / state helpers ──────────────────────────────────────────────────

def _load_patient() -> PatientContext:
    return PatientContext.model_validate_json(DEFAULT_PATIENT.read_text())


def _run(patient_text: str, state_dict: dict | None):
    """Core handler shared by voice and text paths."""
    state = (
        ConversationState(patient=_load_patient())
        if state_dict is None
        else ConversationState.model_validate(state_dict)
    )
    result = run_turn(state, patient_text)

    lang_tag = result.language.primary_lang if result.language else "en"
    audio_out = _synthesize(result.nurse_text, lang_tag)

    verdicts = {
        "patient_said": patient_text,
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
    return (
        audio_out,
        result.nurse_text,
        json.dumps(verdicts, indent=2, default=str),
        result.state.model_dump(),
    )


def voice_handler(audio_path: str | None, state_dict: dict | None):
    if not audio_path:
        return None, "", "{}", state_dict
    try:
        patient_text = _transcribe(audio_path)
    except Exception as e:
        log.exception("transcription failed")
        return None, f"[transcription error: {e}]", "{}", state_dict
    return _run(patient_text, state_dict)


def text_handler(patient_text: str, state_dict: dict | None):
    if not patient_text or not patient_text.strip():
        return None, "", "{}", state_dict
    return _run(patient_text.strip(), state_dict)


# ── UI ───────────────────────────────────────────────────────────────────────

EXAMPLE_LINES = [
    "Hi, yes this is Maria.",
    "I checked my sugar this morning and it was 142.",
    "Hola, soy María. ¿Cómo está usted?",
    "Pues mi blood sugar esta mañana estaba en 165, no muy mal.",
    "Sí, tomé el metformin con breakfast pero el insulin lo olvidé last night.",
    "Tengo un dolor fuerte en el pecho y me duele el brazo izquierdo.",
]

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Constella — bilingual constellation voice agent") as ui:
        gr.Markdown(
            "# Constella\n"
            "You are **María González** — 58-year-old diabetic patient, just out of hospital.\n"
            "**Nurse Ana** will follow up. Speak or type. Try Spanish, English, or both mid-sentence.\n\n"
            "> Voice path: **click the mic → speak → click stop.** "
            "Nurse Ana replies in audio + text.\n"
            "> Text path: type below and press **Enter** or click **Send**."
        )

        state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### You (María)")
                mic = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Speak",
                    show_download_button=False,
                )
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
                audio_out = gr.Audio(
                    label="Voice response",
                    autoplay=True,
                    type="filepath",
                    show_download_button=False,
                    interactive=False,
                )
                nurse_out = gr.Textbox(label="Text transcript", lines=4, interactive=False)

        verdict_out = gr.Code(
            label="What the 4 specialists saw (action + latency breakdown)",
            language="json",
            lines=20,
        )

        # Voice path: fires when recording stops
        mic.stop_recording(
            voice_handler,
            inputs=[mic, state],
            outputs=[audio_out, nurse_out, verdict_out, state],
        )

        # Text path: Enter key or button
        text_in.submit(text_handler, inputs=[text_in, state],
                       outputs=[audio_out, nurse_out, verdict_out, state])
        send_btn.click(text_handler, inputs=[text_in, state],
                       outputs=[audio_out, nurse_out, verdict_out, state])

    return ui


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="Create public Gradio share URL")
    args = p.parse_args()
    build_ui().launch(server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
