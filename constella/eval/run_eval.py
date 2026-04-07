"""Eval runner.

    constella-eval                  # run all scenarios in eval/scenarios/
    constella-eval --scenario 03    # run a single scenario by id prefix
    constella-eval --quick          # 3 baseline scenarios only

Writes a markdown report to eval/results/<timestamp>.md and a JSON dump.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from constella.eval.rubric import aggregate, score_turn
from constella.orchestrator import run_turn
from constella.schemas import ConversationState, PatientContext

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
RESULTS_DIR = Path(__file__).parent / "results"

# Free-tier Groq has a 6000 TPM budget on llama-3.1-8b-instant that refills
# at ~100 tok/sec. A quick eval scenario consumes ~3000-4000 tokens in a burst,
# so we need ~30s between scenarios for Groq free tier. OpenRouter doesn't have
# this problem — default to 0 there. Override with CONSTELLA_EVAL_INTER_SCENARIO_SLEEP.
_provider = os.environ.get("CONSTELLA_PROVIDER", "groq").lower()
_default_sleep = "0" if _provider == "openrouter" else "30"
INTER_SCENARIO_SLEEP_S = float(os.environ.get("CONSTELLA_EVAL_INTER_SCENARIO_SLEEP", _default_sleep))

log = logging.getLogger(__name__)


def load_patient(scenario: dict) -> PatientContext:
    patient_path = SCENARIOS_DIR / scenario["patient_file"]
    with patient_path.open() as f:
        return PatientContext.model_validate_json(f.read())


def load_scenarios(filter_prefix: str | None, quick: bool) -> list[dict]:
    files = sorted(p for p in SCENARIOS_DIR.glob("*.json") if not p.name.startswith("patient_"))
    out: list[dict] = []
    for fp in files:
        with fp.open() as f:
            data = json.load(f)
        if filter_prefix and not data["scenario_id"].startswith(filter_prefix):
            continue
        out.append(data)
    if quick:
        out = [s for s in out if s["scenario_id"] in ("01_english_baseline", "02_spanish_baseline", "03_code_switch_inhaler")]
    return out


def run_scenario(scenario: dict) -> dict:
    patient = load_patient(scenario)
    state = ConversationState(patient=patient)
    turn_results = []
    scores = []
    for i, patient_text in enumerate(scenario["patient_turns"]):
        result = run_turn(state, patient_text)
        state = result.state
        score = score_turn(scenario["scenario_id"], i, result, scenario.get("expected", {}))
        turn_results.append(result)
        scores.append(score)
    return {
        "scenario_id": scenario["scenario_id"],
        "category": scenario.get("category", "uncategorized"),
        "turns": turn_results,
        "scores": scores,
        "summary": aggregate(scores),
    }


def write_report(all_results: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# Constella eval — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"**Scenarios:** {len(all_results)}")
    lines.append("")

    # Roll-up
    flat_scores = [s for r in all_results for s in r["scores"]]
    overall = aggregate(flat_scores)
    lines.append("## Overall")
    lines.append("")
    for k, v in overall.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Per scenario
    lines.append("## Per scenario")
    lines.append("")
    for r in all_results:
        lines.append(f"### `{r['scenario_id']}` ({r['category']})")
        for k, v in r["summary"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        for i, (turn, score) in enumerate(zip(r["turns"], r["scores"])):
            lines.append(f"**Turn {i}** — patient: _{turn.patient_text}_")
            lines.append(f"  - nurse: _{turn.nurse_text}_")
            lines.append(
                f"  - action={turn.action.kind} | "
                f"med={score.medical_accuracy} harm={score.harm_rate} "
                f"lang={score.language_correctness} esc={score.escalation_correctness} "
                f"latency_pass={score.latency_pass} ({turn.total_latency_ms:.0f}ms)"
            )
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("wrote report: %s", out_path)


def write_json_dump(all_results: list[dict], out_path: Path) -> None:
    """JSON dump of scores only (turns contain non-serializable LLM objects)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for r in all_results:
        payload.append({
            "scenario_id": r["scenario_id"],
            "category": r["category"],
            "summary": r["summary"],
            "scores": [s.model_dump() for s in r["scores"]],
        })
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Constella eval runner")
    p.add_argument("--scenario", help="Filter by scenario id prefix (e.g. '03')")
    p.add_argument("--quick", action="store_true", help="Run 3 baseline scenarios only")
    args = p.parse_args()

    scenarios = load_scenarios(args.scenario, args.quick)
    if not scenarios:
        print("No scenarios matched.", file=sys.stderr)
        return 1
    print(f"Running {len(scenarios)} scenarios (inter-scenario sleep: {INTER_SCENARIO_SLEEP_S:.0f}s)...")
    all_results = []
    for i, scenario in enumerate(scenarios):
        if i > 0 and INTER_SCENARIO_SLEEP_S > 0:
            print(f"  ... sleeping {INTER_SCENARIO_SLEEP_S:.0f}s to let Groq TPM refill ...")
            time.sleep(INTER_SCENARIO_SLEEP_S)
        print(f"[{i + 1}/{len(scenarios)}] {scenario['scenario_id']}")
        all_results.append(run_scenario(scenario))

    ts = time.strftime("%Y%m%d_%H%M%S")
    write_report(all_results, RESULTS_DIR / f"{ts}.md")
    write_json_dump(all_results, RESULTS_DIR / f"{ts}.json")

    overall = aggregate([s for r in all_results for s in r["scores"]])
    print()
    print("=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
