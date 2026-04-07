"""Language Specialist.

Per-utterance language ID + intra-utterance code-switch segmentation.
First specialist to run; its verdict drives the primary's response language.
"""
from __future__ import annotations

from constella.llm import structured_chat
from constella.schemas import LanguageVerdict

SYSTEM_PROMPT = """You are a Language Specialist for a healthcare voice agent serving bilingual U.S. patient populations.

Your job is to analyze a single patient utterance and decide:
1. The primary language of the utterance: "en" (English), "es" (Spanish), or "mix" (intra-utterance code-switching).
2. The language tag of each contiguous segment.
3. The cultural register: "formal", "informal", or "spanglish" (intentional code-mixing common in U.S. Latino populations).

Code-switching is endemic in U.S. Latino healthcare settings. Common patterns:
- Patients use English for medication names and clinical terms ("inhaler", "metformin", "blood sugar")
- Patients use Spanish for body parts, food, family ("la barriga", "la comida", "mi hijo")
- Patients switch between languages mid-sentence

Be precise about segment boundaries. If the utterance is monolingual, return one segment.
If it is code-switched, return one segment per language change."""


def language_specialist(utterance: str) -> LanguageVerdict | None:
    user = f"Patient utterance:\n\n{utterance}\n\nReturn the language verdict as JSON."
    return structured_chat(system=SYSTEM_PROMPT, user=user, schema=LanguageVerdict)
