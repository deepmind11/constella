"""Audio format conversion for the realtime pipeline.

fastrtc hands us `(sample_rate, np.ndarray)` tuples; Groq Whisper wants a file
path to a WAV; Cartesia streams raw `pcm_f32le` bytes back to us. These helpers
bridge those three representations.
"""
from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np


def numpy_to_wav_tempfile(sample_rate: int, audio: np.ndarray) -> Path:
    """Write an fastrtc-shape audio array to a temp WAV that Whisper accepts.

    fastrtc delivers `(1, N)` mono audio. We squeeze to 1D, coerce to int16,
    and emit a standard 16-bit PCM WAV. Caller is responsible for deleting
    the file once Whisper has finished with it.
    """
    mono = audio.squeeze()
    if mono.dtype != np.int16:
        if mono.dtype.kind == "f":
            mono = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        else:
            mono = mono.astype(np.int16)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(mono.tobytes())
    return Path(tmp.name)


def cartesia_pcm_to_numpy(
    pcm_bytes: bytes, sample_rate: int
) -> tuple[int, np.ndarray]:
    """Convert a Cartesia `pcm_f32le` chunk into the `(sr, (1, N))` shape fastrtc expects."""
    arr = np.frombuffer(pcm_bytes, dtype=np.float32)
    return (sample_rate, arr.reshape(1, -1))
