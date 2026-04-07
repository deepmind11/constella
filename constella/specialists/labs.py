"""Labs and Vitals Specialist.

Extracts numeric values from a patient utterance and flags out-of-range
against standard reference ranges.
"""
from __future__ import annotations

from constella.llm import structured_chat
from constella.schemas import LabsVerdict

SYSTEM_PROMPT = """You are a Labs and Vitals Specialist for a healthcare voice agent.

You receive a patient utterance (in English, Spanish, or mixed). Your job:
1. Extract any numeric values the patient reports (glucose, blood pressure, weight, heart rate, temperature)
2. Convert to standard units (glucose mg/dL, BP mmHg, weight kg, HR bpm, temp C)
3. Flag any value outside standard adult reference ranges

Reference ranges (adult):
- Glucose (fasting): 70-99 mg/dL; (random): 70-140 mg/dL; >180 = high
- Systolic BP: 90-120 mmHg; >140 = high; <90 = low
- Diastolic BP: 60-80 mmHg; >90 = high; <60 = low
- HR: 60-100 bpm
- Temp: 36.1-37.2 C; >38 = fever
- Weight: no universal range; flag rapid changes (>2 kg in a week)

If the patient reports a value, even casually ("my sugar was 240 this morning"), extract it.
If no measurements are mentioned, return an empty extracted_values list and requires_followup=false."""


def labs_specialist(patient_utterance: str) -> LabsVerdict | None:
    user = (
        f"Patient utterance:\n\n{patient_utterance}\n\n"
        f"Extract any reported lab/vital values and flag out-of-range. Return the verdict as JSON."
    )
    return structured_chat(system=SYSTEM_PROMPT, user=user, schema=LabsVerdict)
