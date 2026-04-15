"""Primary Conversational Agent.

The stateful "constellation primary" — analogous to Polaris's 70B conversational
LLM. Holds the dialogue, references the patient's discharge plan, and produces
a single nurse utterance per turn. Specialists then verify it in parallel.

We deliberately keep the primary's job *narrow*: be warm, be brief, follow the
discharge plan checklist. Safety / accuracy / language verification all live in
the specialists so the primary can stay fast.
"""
from __future__ import annotations

from typing import Literal

from constella.llm import PRIMARY_MODEL, chat
from constella.schemas import ConversationState, Turn

_SPANISH_MARKERS = frozenset("ñ¿¡áéíóú")
_SPANISH_FUNCTION_WORDS = frozenset({
    "hola", "está", "estás", "estas", "esta", "sí", "bueno", "buenos",
    "buenas", "dias", "días", "tardes", "noches", "pero", "porque",
    "como", "cómo", "qué", "que", "dónde", "donde", "entonces", "muy",
    "mucho", "mucha", "también", "tambien", "señora", "señor", "usted",
    "soy", "eres", "tomó", "gracias", "bien", "tiene", "tengo", "tienes",
    "desde", "hospital", "medicamento", "azúcar", "insulina", "doctor",
    "doctora", "ahora", "poco", "ayer", "mañana", "preocupes", "quiero",
    "puedes", "ese", "con", "por", "para", "del", "las", "los", "nada",
    "algo", "todo", "toda", "hace", "hacer", "estoy", "siento", "duele",
})
_ENGLISH_FUNCTION_WORDS = frozenset({
    "the", "and", "is", "are", "was", "were", "i", "my", "you", "your",
    "this", "that", "have", "has", "had", "with", "for", "but", "from",
    "of", "in", "on", "to", "it", "a", "an", "not", "don't", "didn't",
})


def detect_utterance_language(text: str) -> Literal["en", "es", "mix"]:
    """Heuristic language detector for a patient utterance.

    Purpose: give the primary agent an unambiguous signal about what language
    the patient JUST spoke, so Ana replies in the same language instead of
    defaulting to the patient's profile language. Used before the language
    specialist has a chance to weigh in (specialists run after primary).

    Returns "mix" when both languages are present in detectable density.
    """
    lower = text.lower()
    has_spanish_markers = any(c in lower for c in _SPANISH_MARKERS)
    words = set(lower.split())
    spanish_word_hits = len(words & _SPANISH_FUNCTION_WORDS)
    english_word_hits = len(words & _ENGLISH_FUNCTION_WORDS)
    looks_spanish = has_spanish_markers or spanish_word_hits >= 2
    looks_english = english_word_hits >= 2
    if looks_spanish and looks_english:
        return "mix"
    if looks_spanish:
        return "es"
    return "en"


_LANGUAGE_DIRECTIVE = {
    "en": "The patient just spoke in English. Reply in English.",
    "es": "La paciente acaba de hablar en español. Responde en español.",
    "mix": (
        "The patient code-switched between English and Spanish. Match their "
        "register and feel free to code-switch naturally."
    ),
}

PRIMARY_SYSTEM_PROMPT = """You are Nurse Ana, a warm, experienced registered nurse making a post-discharge follow-up call.

Your job on each turn:
1. Listen to what the patient just said.
2. Respond in ONE to THREE short sentences (spoken aloud, so keep it conversational).
3. Cover the next item on the discharge follow-up checklist (medications, symptoms, vitals, appointments) — but only one topic at a time.
4. Match the patient's language. If they speak Spanish, reply in Spanish. If they code-switch (English + Spanish in one sentence), match their register naturally.
5. Never give a new prescription, dose, or diagnosis. You are following an existing discharge plan.
6. If the patient reports anything that sounds dangerous, ask a clarifying question — DO NOT minimize it. The escalation specialist will decide whether to hand off to a human RN.

Tone: warm, calm, attentive. Use the patient's first name. No medical jargon unless the patient uses it first. No long lists. No disclaimers.

The discharge follow-up checklist for THIS patient will be provided in the user message."""


def build_user_prompt(state: ConversationState, last_patient_text: str) -> str:
    history_lines = []
    for turn in state.history[-20:]:  # last 20 turns (~10 exchanges)
        speaker = "Patient" if turn.speaker == "patient" else "Nurse Ana"
        history_lines.append(f"{speaker}: {turn.text}")
    history_block = "\n".join(history_lines) if history_lines else "(start of call)"

    med_block = "\n".join(
        f"  - {m.name} {m.dose_mg or ''}mg, {m.frequency} ({m.indication or 'no indication'})"
        for m in state.patient.medications
    )

    detected_lang = detect_utterance_language(last_patient_text)
    directive = _LANGUAGE_DIRECTIVE[detected_lang]

    return f"""Patient profile:
- Name: {state.patient.name}
- Age: {state.patient.age}
- Pronouns: {state.patient.pronouns}
- Preferred language at home: {state.patient.primary_language} (but ignore this if they just spoke a different language — see directive below)
- Conditions: {", ".join(state.patient.conditions)}
- Discharge summary: {state.patient.discharge_summary}
- Medications:
{med_block}

Recent conversation:
{history_block}

Patient just said:
"{last_patient_text}"

LANGUAGE DIRECTIVE (mandatory): {directive}

Respond as Nurse Ana. One to three short sentences."""


def primary_respond(state: ConversationState, last_patient_text: str) -> str:
    """Run one turn of the primary agent. Returns the raw nurse utterance text."""
    user = build_user_prompt(state, last_patient_text)
    return chat(
        system=PRIMARY_SYSTEM_PROMPT,
        user=user,
        model=PRIMARY_MODEL,
        temperature=0.4,
        max_tokens=200,
    ).strip()


def rewrite_with_hint(
    state: ConversationState,
    last_patient_text: str,
    original: str,
    hint: str,
) -> str:
    """Re-run the primary with a corrective hint from a specialist."""
    base_user = build_user_prompt(state, last_patient_text)
    user = (
        f"{base_user}\n\n"
        f"Your first attempt was: \"{original}\"\n"
        f"A safety reviewer flagged it: {hint}\n"
        f"Please rewrite. Keep it one to three sentences. Match the patient's language."
    )
    return chat(
        system=PRIMARY_SYSTEM_PROMPT,
        user=user,
        model=PRIMARY_MODEL,
        temperature=0.3,
        max_tokens=200,
    ).strip()


def append_turn(state: ConversationState, speaker: str, text: str, **kw) -> ConversationState:
    """Immutably append a turn and return a new ConversationState."""
    new_history = list(state.history) + [Turn(speaker=speaker, text=text, **kw)]  # type: ignore[arg-type]
    return state.model_copy(update={"history": new_history, "turn_count": state.turn_count + 1})
