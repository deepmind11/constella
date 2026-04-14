"""Constellation orchestrator.

Implements the Polaris-style fan-out:

    patient utterance
          │
          ▼
    +-----------------+
    | Primary (70B)   |  generates a candidate nurse response
    +--------+--------+
             │
             ▼
    +-----------------+
    |  Parallel fan-out: 4 specialists run concurrently
    |    - language       (8B)
    |    - medication     (8B)
    |    - labs/vitals    (8B)
    |    - escalation     (8B)
    +--------+--------+
             │
             ▼
    +-----------------+
    | Merge → NextAction:                       |
    |   Escalate  (any specialist demands it)    |
    |   Rewrite   (medication.harm)              |
    |   Append    (labs followup)                |
    |   Emit      (default)                      |
    +-----------------+

We use a thread pool for fan-out instead of a graph framework so the orchestration
stays in a single file and is easy to follow end-to-end.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from constella.primary import append_turn, primary_respond, rewrite_with_hint
from constella.schemas import (
    AppendFollowup,
    ConversationState,
    Emit,
    Escalate,
    EscalationVerdict,
    LabsVerdict,
    LanguageVerdict,
    MedicationVerdict,
    NextAction,
    Rewrite,
)
from constella.specialists import (
    escalation_specialist,
    labs_specialist,
    language_specialist,
    medication_specialist,
)

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    """Everything we learn from a single patient -> nurse turn."""

    state: ConversationState
    patient_text: str
    primary_draft: str
    nurse_text: str
    action: NextAction
    language: LanguageVerdict | None = None
    medication: MedicationVerdict | None = None
    labs: LabsVerdict | None = None
    escalation: EscalationVerdict | None = None
    primary_latency_ms: float = 0.0
    specialist_latency_ms: float = 0.0
    rewrite_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    rewrite_count: int = 0
    notes: list[str] = field(default_factory=list)


def _run_specialists(
    patient_text: str,
    nurse_draft: str,
    state: ConversationState,
) -> tuple[LanguageVerdict | None, MedicationVerdict | None, LabsVerdict | None, EscalationVerdict | None]:
    """Fire all 4 specialists in parallel. Return their verdicts (any may be None)."""
    results: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {
            ex.submit(language_specialist, patient_text): "language",
            ex.submit(medication_specialist, nurse_draft, state.patient): "medication",
            ex.submit(labs_specialist, patient_text): "labs",
            ex.submit(escalation_specialist, patient_text, state.patient): "escalation",
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                log.exception("specialist %s failed", name)
                results[name] = None
    return (
        results.get("language"),  # type: ignore[return-value]
        results.get("medication"),  # type: ignore[return-value]
        results.get("labs"),  # type: ignore[return-value]
        results.get("escalation"),  # type: ignore[return-value]
    )


def _merge_verdicts(
    primary_text: str,
    language: LanguageVerdict | None,
    medication: MedicationVerdict | None,
    labs: LabsVerdict | None,
    escalation: EscalationVerdict | None,
) -> NextAction:
    """Combine specialist verdicts into a single NextAction.

    Priority order (highest first):
        1. Escalation (urgency in {urgent, emergency}) → Escalate
        2. Medication harm → Rewrite
        3. Labs requires_followup → AppendFollowup
        4. Default → Emit
    """
    # 1. Escalation wins
    if escalation and escalation.escalate and escalation.urgency in ("urgent", "emergency"):
        return Escalate(
            reason=escalation.reason or "Red flag detected by escalation specialist.",
            urgency=escalation.urgency,
        )

    # 2. Medication harm
    if medication and not medication.safe:
        harm_issues = [i for i in medication.issues if i.severity == "harm"]
        if harm_issues:
            issue = harm_issues[0]
            hint = (
                f"{issue.description}. "
                f"Suggested correction: {issue.suggested_correction or 'remove or fix the unsafe statement'}."
            )
            return Rewrite(original_text=primary_text, hint=hint)

    # 3. Labs follow-up
    lang_tag = language.primary_lang if language else "en"
    if labs and labs.requires_followup and labs.followup_suggestion:
        return AppendFollowup(
            text=primary_text,
            appended=labs.followup_suggestion,
            language_tag=lang_tag,
        )

    # 4. Default emit
    return Emit(text=primary_text, language_tag=lang_tag)


def run_turn(
    state: ConversationState,
    patient_text: str,
    *,
    max_rewrites: int = 1,
) -> TurnResult:
    """Run one full constellation turn.

    Steps:
        1. primary draft (70B)
        2. fan out 4 specialists in parallel (8B each)
        3. merge verdicts -> NextAction
        4. if Rewrite, run primary again with hint (max once by default)
        5. append nurse turn to state and return TurnResult
    """
    t0 = time.perf_counter()

    # 1. primary draft
    t_p0 = time.perf_counter()
    primary_draft = primary_respond(state, patient_text)
    primary_latency_ms = (time.perf_counter() - t_p0) * 1000

    # 2. specialists
    t_s0 = time.perf_counter()
    language, medication, labs, escalation = _run_specialists(patient_text, primary_draft, state)
    specialist_latency_ms = (time.perf_counter() - t_s0) * 1000

    # 3. merge
    action = _merge_verdicts(primary_draft, language, medication, labs, escalation)

    nurse_text = primary_draft
    rewrite_latency_ms = 0.0
    rewrite_count = 0

    # 4. handle rewrite
    if isinstance(action, Rewrite) and max_rewrites > 0:
        t_r0 = time.perf_counter()
        nurse_text = rewrite_with_hint(state, patient_text, primary_draft, action.hint)
        rewrite_latency_ms = (time.perf_counter() - t_r0) * 1000
        rewrite_count = 1
        # Re-merge after rewrite for the FINAL action (now likely Emit)
        # Note: we trust the rewrite — we don't re-run specialists in this MVP.
        action = Emit(
            text=nurse_text,
            language_tag=language.primary_lang if language else "en",
        )
    elif isinstance(action, Emit):
        nurse_text = action.text
    elif isinstance(action, AppendFollowup):
        nurse_text = f"{action.text} {action.appended}".strip()
    elif isinstance(action, Escalate):
        nurse_text = (
            "I want to make sure you get the right help right away. "
            "I'm going to connect you with one of our nurses now. Please stay on the line."
        )

    # 5. update state
    new_state = append_turn(
        state,
        speaker="patient",
        text=patient_text,
        language_tag=language.primary_lang if language else "en",
    )
    new_state = append_turn(
        new_state,
        speaker="nurse",
        text=nurse_text,
        language_tag=language.primary_lang if language else "en",
        latency_ms=(time.perf_counter() - t0) * 1000,
    )
    if isinstance(action, Escalate):
        new_state = new_state.model_copy(
            update={"escalated": True, "escalation_reason": action.reason}
        )

    total_latency_ms = (time.perf_counter() - t0) * 1000

    return TurnResult(
        state=new_state,
        patient_text=patient_text,
        primary_draft=primary_draft,
        nurse_text=nurse_text,
        action=action,
        language=language,
        medication=medication,
        labs=labs,
        escalation=escalation,
        primary_latency_ms=primary_latency_ms,
        specialist_latency_ms=specialist_latency_ms,
        rewrite_latency_ms=rewrite_latency_ms,
        total_latency_ms=total_latency_ms,
        rewrite_count=rewrite_count,
    )
