"""Specialist verifier agents that fan out in parallel after the primary turn."""

from constella.specialists.escalation import escalation_specialist
from constella.specialists.labs import labs_specialist
from constella.specialists.language import language_specialist
from constella.specialists.medication import medication_specialist

__all__ = [
    "language_specialist",
    "medication_specialist",
    "labs_specialist",
    "escalation_specialist",
]
