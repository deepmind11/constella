"""Primary Conversational Agent.

The stateful "constellation primary" — analogous to Polaris's 70B conversational
LLM. Holds the dialogue, references the patient's discharge plan, and produces
a single nurse utterance per turn. Specialists then verify it in parallel.

We deliberately keep the primary's job *narrow*: be warm, be brief, follow the
discharge plan checklist. Safety / accuracy / language verification all live in
the specialists so the primary can stay fast.
"""
from __future__ import annotations

from constella.llm import PRIMARY_MODEL, chat
from constella.schemas import ConversationState, Turn

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
    for turn in state.history[-6:]:  # last 3 exchanges
        speaker = "Patient" if turn.speaker == "patient" else "Nurse Ana"
        history_lines.append(f"{speaker}: {turn.text}")
    history_block = "\n".join(history_lines) if history_lines else "(start of call)"

    med_block = "\n".join(
        f"  - {m.name} {m.dose_mg or ''}mg, {m.frequency} ({m.indication or 'no indication'})"
        for m in state.patient.medications
    )

    return f"""Patient profile:
- Name: {state.patient.name}
- Age: {state.patient.age}
- Pronouns: {state.patient.pronouns}
- Primary language: {state.patient.primary_language}
- Conditions: {", ".join(state.patient.conditions)}
- Discharge summary: {state.patient.discharge_summary}
- Medications:
{med_block}

Recent conversation:
{history_block}

Patient just said:
"{last_patient_text}"

Respond as Nurse Ana. One to three short sentences. Match the patient's language."""


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
