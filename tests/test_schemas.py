"""Schema sanity tests. Run without GROQ_API_KEY."""
from __future__ import annotations

import json
from pathlib import Path

from constella.orchestrator import _merge_verdicts
from constella.schemas import (
    AppendFollowup,
    Emit,
    Escalate,
    EscalationVerdict,
    LabsVerdict,
    LanguageVerdict,
    Measurement,
    MedicationIssue,
    MedicationVerdict,
    PatientContext,
    Rewrite,
)

PATIENT_FILE = (
    Path(__file__).resolve().parents[1]
    / "constella"
    / "eval"
    / "scenarios"
    / "patient_maria.json"
)


def test_patient_loads():
    p = PatientContext.model_validate_json(PATIENT_FILE.read_text())
    assert p.name == "María González"
    assert p.primary_language == "es"
    assert any(m.name == "Metformin" for m in p.medications)


def test_merge_default_emit():
    action = _merge_verdicts(
        primary_text="Hola María, ¿cómo se siente hoy?",
        language=LanguageVerdict(primary_lang="es", segments=[], code_switch_count=0),
        medication=MedicationVerdict(safe=True, issues=[]),
        labs=LabsVerdict(),
        escalation=EscalationVerdict(escalate=False, urgency="none", red_flags=[]),
    )
    assert isinstance(action, Emit)
    assert action.language_tag == "es"


def test_merge_escalation_wins():
    action = _merge_verdicts(
        primary_text="Tomate la pastilla.",
        language=LanguageVerdict(primary_lang="es", segments=[]),
        medication=MedicationVerdict(safe=True, issues=[]),
        labs=LabsVerdict(),
        escalation=EscalationVerdict(
            escalate=True,
            reason="chest pain",
            urgency="emergency",
            red_flags=["chest pain"],
        ),
    )
    assert isinstance(action, Escalate)
    assert action.urgency == "emergency"


def test_merge_medication_harm_triggers_rewrite():
    action = _merge_verdicts(
        primary_text="Take 100 units of insulin tonight.",
        language=LanguageVerdict(primary_lang="en", segments=[]),
        medication=MedicationVerdict(
            safe=False,
            issues=[
                MedicationIssue(
                    description="Dose 5x prescribed amount",
                    severity="harm",
                    suggested_correction="Patient is on 20u qhs; do not change dose without provider order.",
                )
            ],
        ),
        labs=LabsVerdict(),
        escalation=EscalationVerdict(escalate=False, urgency="none", red_flags=[]),
    )
    assert isinstance(action, Rewrite)
    assert "20u qhs" in action.hint


def test_merge_labs_followup_appends():
    action = _merge_verdicts(
        primary_text="Glad to hear it.",
        language=LanguageVerdict(primary_lang="en", segments=[]),
        medication=MedicationVerdict(safe=True, issues=[]),
        labs=LabsVerdict(
            extracted_values=[
                Measurement(metric="glucose", value=287, unit="mg/dL", out_of_range=True)
            ],
            out_of_range_count=1,
            requires_followup=True,
            followup_suggestion="Have you eaten anything in the last 2 hours?",
        ),
        escalation=EscalationVerdict(escalate=False, urgency="none", red_flags=[]),
    )
    assert isinstance(action, AppendFollowup)
    assert "eaten" in action.appended


def test_scenario_files_parse():
    """All scenario JSON files should parse cleanly and reference a valid patient."""
    scenarios_dir = PATIENT_FILE.parent
    files = [p for p in scenarios_dir.glob("*.json") if not p.name.startswith("patient_")]
    assert len(files) >= 5
    for fp in files:
        data = json.loads(fp.read_text())
        assert "scenario_id" in data
        assert "patient_turns" in data
        assert len(data["patient_turns"]) >= 1
        assert (scenarios_dir / data["patient_file"]).exists()
