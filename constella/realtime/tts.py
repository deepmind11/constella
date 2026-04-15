"""Streaming TTS via Cartesia Sonic-3.

Yields raw `pcm_f32le` bytes as the model synthesizes. The realtime handler
converts those chunks into fastrtc audio frames so Ana's voice starts playing
within ~100 ms of the first token arriving from the primary agent.

Voices are native per language — a native-Spanish voice is used for Spanish
replies and a native-English voice for English. Using a single multilingual
voice makes the non-native language sound phonetically approximated (e.g.,
American accent on Spanish), which is unusable for a bilingual healthcare
agent. The tradeoff is that Ana's voice identity shifts slightly when she
code-switches — analogous to a bilingual nurse in real life.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Literal

log = logging.getLogger(__name__)

# Native-English female, warm conversational — fits a U.S. nurse making a
# follow-up call. Override via CARTESIA_VOICE_EN.
_DEFAULT_VOICE_EN = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"  # Tessa - Kind Companion

# Native-Latin-American-Spanish female, calm and professional — fits a nurse
# talking to a U.S. Latino patient. Override via CARTESIA_VOICE_ES.
_DEFAULT_VOICE_ES = "3597a26f-80ef-4bd5-8101-9699bc764917"  # Ximena - Calm Navigator

_MODEL_ID = "sonic-3"
SAMPLE_RATE = 44100


def _pick_voice(language: Literal["en", "es"] | None) -> str:
    """Pick a native voice for the given reply language.

    Precedence:
      1. CARTESIA_VOICE_ID — single-voice override (legacy behavior)
      2. CARTESIA_VOICE_ES / CARTESIA_VOICE_EN — per-language overrides
      3. Hard-coded defaults tuned for the healthcare-nurse register
    """
    single = os.environ.get("CARTESIA_VOICE_ID")
    if single:
        return single
    if language == "es":
        return os.environ.get("CARTESIA_VOICE_ES", _DEFAULT_VOICE_ES)
    return os.environ.get("CARTESIA_VOICE_EN", _DEFAULT_VOICE_EN)


def stream_tts(
    text: str,
    *,
    language: Literal["en", "es"] | None = None,
) -> Iterator[bytes]:
    """Yield `pcm_f32le` audio chunks as Cartesia synthesizes `text`.

    `language`: "en" or "es" selects a native voice AND passes the language
    hint to Sonic-3. `None` falls back to the English voice (safest default).
    """
    from cartesia import Cartesia  # lazy import so pytest doesn't pull websockets

    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CARTESIA_API_KEY is required for the realtime demo. Add it to .env."
        )

    voice_id = _pick_voice(language)
    client = Cartesia(api_key=api_key)

    with client.tts.websocket_connect() as conn:
        ctx_kwargs: dict = {
            "model_id": _MODEL_ID,
            "voice": {"mode": "id", "id": voice_id},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_f32le",
                "sample_rate": SAMPLE_RATE,
            },
        }
        if language is not None:
            ctx_kwargs["language"] = language
        ctx = conn.context(**ctx_kwargs)
        ctx.push(text)
        ctx.no_more_inputs()
        for response in ctx.receive():
            if response.type == "chunk" and response.audio:
                yield response.audio
