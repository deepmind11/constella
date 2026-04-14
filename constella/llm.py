"""Thin LLM client wrapper.

Supports two inference backends, auto-detected from environment:

    1. Groq (preferred when Dev tier available)
       - export GROQ_API_KEY=gsk_...
       - Primary: llama-3.3-70b-versatile  (~250 ms TTFT)
       - Specialists: llama-3.1-8b-instant (~80 ms TTFT)
       - Free tier: 6000 TPM on 8B is the painful ceiling for fan-out

    2. OpenRouter (fallback when Groq dev tier is unavailable)
       - export OPENROUTER_API_KEY=sk-or-...
       - Primary: meta-llama/llama-3.3-70b-instruct
       - Specialists: meta-llama/llama-3.1-8b-instruct
       - OpenRouter is OpenAI-compatible, so we use the `openai` SDK pointed
         at https://openrouter.ai/api/v1

Precedence: if GROQ_API_KEY is set AND CONSTELLA_PROVIDER is not "openrouter",
Groq wins. Otherwise we fall back to OpenRouter. Set CONSTELLA_PROVIDER=openrouter
to force OpenRouter even when a Groq key is present.

Anthropic Claude is supported as a fallback for higher-quality LLM-as-judge in
the eval harness (not wired into primary/specialist paths).
"""
from __future__ import annotations

import os
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# ---- Provider selection ---------------------------------------------------

_PROVIDER_OVERRIDE = os.environ.get("CONSTELLA_PROVIDER", "").lower()


def _select_provider() -> str:
    if _PROVIDER_OVERRIDE in ("groq", "openrouter"):
        return _PROVIDER_OVERRIDE
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    raise RuntimeError(
        "No LLM provider configured. Set GROQ_API_KEY or OPENROUTER_API_KEY in .env. "
        "See SETUP.md step 3."
    )


# ---- Model names per provider ---------------------------------------------

_MODELS_BY_PROVIDER = {
    "groq": {
        "primary": "llama-3.3-70b-versatile",
        "specialist": "llama-3.1-8b-instant",
    },
    "openrouter": {
        "primary": "meta-llama/llama-3.3-70b-instruct",
        "specialist": "meta-llama/llama-3.1-8b-instruct",
    },
}


def _resolve_models() -> tuple[str, str]:
    provider = _select_provider()
    defaults = _MODELS_BY_PROVIDER[provider]
    primary = os.environ.get("CONSTELLA_PRIMARY_MODEL", defaults["primary"])
    specialist = os.environ.get("CONSTELLA_SPECIALIST_MODEL", defaults["specialist"])
    return primary, specialist


# Resolved once at import time; override via CONSTELLA_PRIMARY_MODEL / CONSTELLA_SPECIALIST_MODEL
PRIMARY_MODEL, SPECIALIST_MODEL = _resolve_models()

# ---- Client singletons ----------------------------------------------------

_client: OpenAI | None = None


def _client_singleton() -> OpenAI:
    """Return a cached OpenAI-compatible client pointed at the active provider."""
    global _client
    if _client is not None:
        return _client
    provider = _select_provider()
    if provider == "groq":
        _client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            max_retries=5,
        )
    elif provider == "openrouter":
        _client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            max_retries=5,
            default_headers={
                # OpenRouter recommends these for attribution / rate-limit tiering
                "HTTP-Referer": "https://github.com/deepmind11/constella",
                "X-Title": "Constella",
            },
        )
    else:  # pragma: no cover - guarded by _select_provider
        raise RuntimeError(f"unknown provider: {provider}")
    return _client


# ---- Chat primitives ------------------------------------------------------


def chat(
    *,
    system: str,
    user: str,
    model: str = PRIMARY_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 512,
    response_format: dict | None = None,
) -> str:
    """Single-turn chat completion. Returns the assistant message text."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = _client_singleton().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _schema_hint(schema: type[BaseModel]) -> str:
    """Produce a compact human-readable field list for a Pydantic schema.

    Resolves $defs references so nested models (e.g. LanguageSegment inside
    LanguageVerdict) get their enum values documented — otherwise Llama 8B
    fills those fields with natural-language strings that fail Pydantic validation.
    Typically 80-200 tokens; far cheaper than the full JSON schema dump.
    """
    full = schema.model_json_schema()
    defs = full.get("$defs", {})

    def _resolve(spec: dict) -> dict:
        if "$ref" in spec:
            name = spec["$ref"].split("/")[-1]
            return defs.get(name, spec)
        return spec

    def _describe(name: str, spec: dict, indent: int = 0) -> list[str]:
        spec = _resolve(spec)
        pad = "  " * indent
        t = spec.get("type", "")
        enum = spec.get("enum")
        if enum:
            t = f"one of {enum}"
        lines = [f"{pad}{name} ({t})"]
        # recurse into array items
        if t == "array" and "items" in spec:
            item_spec = _resolve(spec["items"])
            for sub_name, sub_spec in item_spec.get("properties", {}).items():
                lines.extend(_describe(sub_name, sub_spec, indent + 1))
        # recurse into object properties
        elif "properties" in spec:
            for sub_name, sub_spec in spec["properties"].items():
                lines.extend(_describe(sub_name, sub_spec, indent + 1))
        return lines

    lines: list[str] = []
    for prop_name, prop_spec in full.get("properties", {}).items():
        lines.extend(_describe(prop_name, prop_spec))
    return "\n".join(lines)


def structured_chat(
    *,
    system: str,
    user: str,
    schema: type[T],
    model: str = SPECIALIST_MODEL,
    max_retries: int = 3,
) -> T | None:
    """Specialist-style chat that returns a Pydantic model.

    Uses native JSON mode (`response_format={"type": "json_object"}`) so we get
    guaranteed JSON output without pasting the full schema in the prompt.
    Pydantic still validates the result. On validation error we retry up to
    max_retries with a compact repair hint. If we still can't get valid JSON
    we return None and the caller treats it as 'no verdict'.
    """
    base_system = (
        f"{system}\n\n"
        f"Respond with a SINGLE JSON object. Required fields:\n"
        f"{_schema_hint(schema)}\n"
        f"No prose. No markdown fences. Only the JSON object."
    )
    last_err: str | None = None
    for attempt in range(max_retries):
        prompt = user if attempt == 0 else (
            f"{user}\n\nYour previous response had this error: {last_err}\n"
            f"Return ONLY valid JSON."
        )
        raw = chat(
            system=base_system,
            user=prompt,
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        # Strip common drift: stray markdown fences
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            return schema.model_validate_json(cleaned)
        except (ValidationError, ValueError) as e:
            last_err = str(e)[:300]
    return None
