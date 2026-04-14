"""Terminal chat — run the constellation from the command line.

Usage:
    constella-chat
    constella-chat --patient eval/scenarios/patient_maria.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from constella.orchestrator import run_turn
from constella.schemas import ConversationState, PatientContext

DEFAULT_PATIENT = (
    Path(__file__).resolve().parent / "eval" / "scenarios" / "patient_maria.json"
)


def main() -> int:
    p = argparse.ArgumentParser(description="Constella terminal chat")
    p.add_argument(
        "--patient",
        type=Path,
        default=DEFAULT_PATIENT,
        help="Path to patient JSON file",
    )
    p.add_argument("--verbose", action="store_true", help="Show specialist verdicts")
    args = p.parse_args()

    patient = PatientContext.model_validate_json(args.patient.read_text())
    state = ConversationState(patient=patient)

    print(f"\nConstella — post-discharge follow-up")
    print(f"Patient: {patient.name}, {patient.age}yo | {', '.join(patient.conditions)}")
    print("Type as the patient. Ctrl-C or 'quit' to exit.\n")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCall ended.")
            return 0

        if user_input.lower() in ("quit", "exit", "q"):
            print("Call ended.")
            return 0

        if not user_input:
            continue

        result = run_turn(state, user_input)
        state = result.state

        print(f"\nNurse Ana: {result.nurse_text}")

        if args.verbose:
            print(f"\n  [action={result.action.kind} | "
                  f"total={result.total_latency_ms:.0f}ms | "
                  f"primary={result.primary_latency_ms:.0f}ms | "
                  f"specialists={result.specialist_latency_ms:.0f}ms]")
            if result.language:
                print(f"  [lang={result.language.primary_lang} "
                      f"register={result.language.cultural_register} "
                      f"switches={result.language.code_switch_count}]")
            if result.escalation and result.escalation.escalate:
                print(f"  [ESCALATE urgency={result.escalation.urgency} "
                      f"reason={result.escalation.reason}]")

        if state.escalated:
            print("\n*** Call escalated to human RN. Session ended. ***")
            return 0


if __name__ == "__main__":
    sys.exit(main())
