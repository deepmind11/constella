"""Streaming TTS via Cartesia Sonic-3.

Yields raw `pcm_f32le` bytes as the model synthesizes. The realtime handler
converts those chunks into fastrtc audio frames so Ana's voice starts playing
within ~100 ms of the first token arriving from the primary agent.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import Literal

log = logging.getLogger(__name__)

# Default voice — Cartesia's multilingual example voice, handles en + es and
# reasonable code-switching. Override by setting CARTESIA_VOICE_ID to any
# voice UUID from https://play.cartesia.ai/voices.
_DEFAULT_VOICE_ID = "6ccbfb76-1fc6-48f7-b71d-91ac6298247b"
_MODEL_ID = "sonic-3"
SAMPLE_RATE = 44100


def stream_tts(
    text: str,
    *,
    language: Literal["en", "es"] | None = None,
) -> Iterator[bytes]:
    """Yield `pcm_f32le` audio chunks as Cartesia synthesizes `text`.

    `language`:
      - "en" or "es": hard-hints Cartesia to that language (cleaner prosody
        when the language specialist is confident)
      - None: let Sonic-3 auto-detect (right choice for code-switched output)
    """
    from cartesia import Cartesia  # lazy import so pytest doesn't pull websockets

    api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "CARTESIA_API_KEY is required for the realtime demo. Add it to .env."
        )

    voice_id = os.environ.get("CARTESIA_VOICE_ID", _DEFAULT_VOICE_ID)
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
