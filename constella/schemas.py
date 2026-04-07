"""Pydantic schemas shared across primary, specialists, and orchestrator.

Every specialist returns a structured verdict; the orchestrator merges them into a
single NextAction that the loop executes.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---- Patient and conversation state ---------------------------------------


class Medication(BaseModel):
    name: str
    dose_mg: float | None = None
    frequency: str  # e.g. "twice daily with meals"
    indication: str | None = None


class PatientContext(BaseModel):
    """Synthetic patient profile loaded from eval/scenarios/patient_*.json."""

    name: str
    age: int
    pronouns: str
    primary_language: Literal["en", "es"]
    secondary_language: Literal["en", "es", "none"] = "none"
    conditions: list[str]
    medications: list[Medication]
    discharge_summary: str
    last_labs: dict[str, float] = Field(default_factory=dict)


class Turn(BaseModel):
    speaker: Literal["patient", "nurse"]
    text: str
    language_tag: Literal["en", "es", "mix"] = "en"
    latency_ms: float | None = None


class ConversationState(BaseModel):
    patient: PatientContext
    history: list[Turn] = Field(default_factory=list)
    turn_count: int = 0
    escalated: bool = False
    escalation_reason: str | None = None


# ---- Specialist verdicts ---------------------------------------------------


class LanguageSegment(BaseModel):
    text: str
    lang: Literal["en", "es", "unk"]
    confidence: float


class LanguageVerdict(BaseModel):
    primary_lang: Literal["en", "es", "mix"]
    segments: list[LanguageSegment]
    code_switch_count: int = 0
    cultural_register: Literal["formal", "informal", "spanglish"] = "formal"


class MedicationIssue(BaseModel):
    description: str
    severity: Literal["info", "warn", "harm"]
    suggested_correction: str | None = None


class MedicationVerdict(BaseModel):
    safe: bool
    issues: list[MedicationIssue] = Field(default_factory=list)


class Measurement(BaseModel):
    metric: str  # e.g. "glucose", "systolic_bp", "weight_kg"
    value: float
    unit: str
    out_of_range: bool = False
    range_note: str | None = None


class LabsVerdict(BaseModel):
    extracted_values: list[Measurement] = Field(default_factory=list)
    out_of_range_count: int = 0
    requires_followup: bool = False
    followup_suggestion: str | None = None


class EscalationVerdict(BaseModel):
    escalate: bool
    reason: str | None = None
    urgency: Literal["none", "routine", "urgent", "emergency"] = "none"
    red_flags: list[str] = Field(default_factory=list)


# ---- Orchestrator decisions ------------------------------------------------


class Emit(BaseModel):
    """Emit the primary agent's response as-is."""

    kind: Literal["emit"] = "emit"
    text: str
    language_tag: Literal["en", "es", "mix"] = "en"


class Rewrite(BaseModel):
    """Ask the primary agent to rewrite its response with a hint."""

    kind: Literal["rewrite"] = "rewrite"
    original_text: str
    hint: str


class AppendFollowup(BaseModel):
    """Emit the primary's response with an appended follow-up question."""

    kind: Literal["append"] = "append"
    text: str
    appended: str
    language_tag: Literal["en", "es", "mix"] = "en"


class Escalate(BaseModel):
    """Hand off to a human RN immediately."""

    kind: Literal["escalate"] = "escalate"
    reason: str
    urgency: Literal["urgent", "emergency"] = "urgent"


NextAction = Emit | Rewrite | AppendFollowup | Escalate


# ---- Eval scoring ----------------------------------------------------------


class TurnScore(BaseModel):
    scenario_id: str
    turn_index: int
    medical_accuracy: int  # 0-3
    harm_rate: int  # 0-3 (3 = safe)
    language_correctness: int  # 0-3
    escalation_correctness: int  # 0-3
    latency_pass: bool  # under 2.2 sec
    notes: str | None = None
