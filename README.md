# Constella

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

> Multi-agent voice constellation for bilingual healthcare follow-up calls, with first-class English-Spanish code-switching support.

---

## What This Is

Constella is a runnable implementation of a Polaris-style safety constellation for voice-based healthcare AI. One stateful primary conversational agent ("Nurse Ana") drives the dialogue; four stateless specialist verifier agents (language, medication, labs, escalation) run in parallel after every primary turn, and an orchestrator merges their verdicts into a single next action.

The use case is a post-discharge follow-up call with a bilingual Type 2 diabetic patient who code-switches between English and Spanish mid-utterance. The voice I/O layer targets [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) for GPU deployments; the portable Gradio demo uses Groq-hosted Whisper for ASR and Google gTTS for TTS (see [Tech Stack](#tech-stack) for the full model / provider breakdown).

## Quick Start

```bash
# Clone
git clone https://github.com/deepmind11/constella.git
cd constella

# Install
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# Configure — add your Groq API key (free tier works)
echo 'GROQ_API_KEY=your_key_here' > .env

# Launch the Gradio voice demo (mic or text input)
uv run constella-demo

# Or run the eval harness
uv run constella-eval
```

See [SETUP.md](SETUP.md) for the VibeVoice local-GPU path and the full bootstrap.

## Architecture

```
                                                       ┌──────────────────────────┐
                                                       │  Eval harness            │
                                                       │  (5 scenarios)           │
                                                       └────────────┬─────────────┘
                                                                    │
                                                                    ▼
┌───────────┐    audio    ┌──────────────┐  text    ┌──────────────────────────────┐
│  Patient  │────────────▶│  ASR:        │─────────▶│   Primary Conversational     │
│  speaks   │             │  Whisper     │          │   Agent (Llama 3.3 70B)      │
│  (mic or  │             │  large-v3    │          │   "Nurse Ana"                │
│   file)   │             │  turbo       │          │   - manages dialogue state   │
└───────────┘             │  (Groq API)  │          │   - generates next utterance │
      ▲                   └──────────────┘          └──────────────┬───────────────┘
      │                                                            │
      │                                            (parallel fan-out)
      │                                                            │
      │            ┌──────────────────┬──────────────────┬─────────┴────────┐
      │            │                  │                  │                  │
      │            ▼                  ▼                  ▼                  ▼
      │     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │     │ Language     │  │ Medication   │  │ Labs & Vitals│  │ Escalation   │
      │     │ Specialist   │  │ Specialist   │  │ Specialist   │  │ Specialist   │
      │     │ - lang ID    │  │ - dosage     │  │ - extract    │  │ - red-flag   │
      │     │ - code-switch│  │ - drug-drug  │  │   numbers    │  │   symptoms   │
      │     │ - route to   │  │ - timing     │  │ - check range│  │ - human XFR  │
      │     │   ES/EN/MIX  │  │              │  │              │  │              │
      │     └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
      │            │                  │                 │                 │
      │            └──────────────────┴─────────────────┴─────────────────┘
      │                                       │
      │                                       ▼
      │                       ┌──────────────────────────────┐
      │                       │  Orchestrator merges          │
      │                       │  specialist verdicts.         │
      │                       │  - If escalation: hard XFR    │
      │                       │  - If med error: rewrite turn │
      │                       │  - If labs followup: append   │
      │                       │  - Else: emit primary's turn  │
      │                       └──────────────┬───────────────┘
      │                                      │
      │                                      ▼
      │                       ┌──────────────────────────────┐
      │                       │  TTS: gTTS (Google API)      │
      │                       │  multilingual, including     │
      │                       │  Spanish and English          │
      │                       └──────────────┬───────────────┘
      │                                      │
      └──────────────────────────────────────┘
```

- The **primary** is a stateful Llama 3.3 70B on Groq. It holds the conversation, follows the discharge-plan checklist, and produces a single short nurse utterance per turn.
- The **four specialists** are stateless Llama 3.1 8B calls. They fan out in parallel via a `ThreadPoolExecutor` so the specialist tax is `max(specialist_latencies)`, not their sum.
- The **orchestrator** merges the four verdicts into one of four actions with the priority order `Escalate > Rewrite > AppendFollowup > Emit`. Safety beats correctness beats completeness beats friendliness.
- **VibeVoice is the production target, not the current demo path.** The Gradio demo uses Groq-hosted Whisper and Google gTTS so it runs on any laptop with a Groq key. Swapping in `VibeVoice-ASR-7B` and `VibeVoice-Realtime-0.5B` for self-hosted GPU deployments is listed in [Tradeoffs](#tradeoffs-and-what-id-do-with-more-time).

See [how_it_works.md](how_it_works.md) for the full design rationale.

## Specialists

| Specialist | Role | Model |
|---|---|---|
| **Language** | Per-utterance language ID + intra-utterance code-switch segmentation. Routes downstream prompts to EN, ES, or MIX. | Llama 3.1 8B |
| **Medication** | Verifies dosage, timing, and contraindications in the nurse's draft reply against the patient's med list. | Llama 3.1 8B |
| **Labs & Vitals** | Extracts numeric values from patient utterances ("my sugar was 240 this morning") and flags out-of-range. | Llama 3.1 8B |
| **Escalation** | Detects red-flag symptoms (chest pain, severe hypo/hyperglycemia, stroke signs, suicidal ideation) and triggers human handoff. | Llama 3.1 8B |

Every specialist returns a Pydantic-validated verdict. Malformed JSON is retried up to three times with a repair hint; persistent failures surface as "no verdict" and the orchestrator logs a warning.

## Eval Harness

5 synthetic post-discharge scenarios, one per bucket:

| # | Scenario file | Bucket |
|---|---|---|
| 01 | `01_english_baseline.json` | English-only baseline |
| 02 | `02_spanish_baseline.json` | Spanish-only baseline |
| 03 | `03_code_switch_inhaler.json` | Intra-utterance code-switching (Spanglish) |
| 04 | `04_high_glucose_followup.json` | Labs follow-up |
| 05 | `05_chest_pain_escalation.json` | Escalation / emergency |

Each turn is scored on 5 dimensions: `medical_accuracy`, `harm_rate`, `language_correctness`, `escalation_correctness`, and `latency_pass` (2.2 s end-to-end budget).

The synthetic patient is **María González**, a 58-year-old on metformin + insulin glargine + lisinopril + atorvastatin. Her full profile lives at `constella/eval/scenarios/patient_maria.json`.

Run the full sweep:

```bash
uv run constella-eval
```

Results write to `constella/eval/results/<timestamp>.md` (human-readable report) and `.json` (machine-readable score dump).

## Tech Stack

**Models and providers** — Groq is the LLM inference provider for everything text-related; it also hosts Whisper for ASR. gTTS is Google's TTS. VibeVoice is self-hosted on GPU for the production path.

| Pipeline stage | Model | Provider |
|---|---|---|
| ASR (speech → text), demo | `whisper-large-v3-turbo` | Groq API |
| ASR (speech → text), prod target | `VibeVoice-ASR-7B` | self-hosted (GPU) |
| Primary LLM ("Nurse Ana") | `llama-3.3-70b-versatile` | Groq API |
| Specialist LLMs (×4) | `llama-3.1-8b-instant` | Groq API |
| TTS (text → speech), demo | `gTTS` | Google API |
| TTS (text → speech), prod target | `VibeVoice-Realtime-0.5B` | self-hosted (GPU) |

Provider for the LLMs is overridable via `CONSTELLA_PROVIDER=openrouter`, which swaps Groq for OpenRouter-hosted Llama variants; the Gradio mic path still needs `GROQ_API_KEY` because OpenRouter doesn't expose audio transcription.

**Other:**
- **Language:** Python 3.11
- **Schemas / validation:** Pydantic v2
- **Fan-out:** stdlib `concurrent.futures.ThreadPoolExecutor` (no graph framework)
- **Demo UI:** Gradio
- **Packaging:** uv + hatchling

## Why VibeVoice

VibeVoice-ASR-7B is one of the few openly available ASR models that handles intra-utterance code-switching without a language parameter. That property matters for U.S. Latino patient populations, where patients commonly use English for clinical terms ("inhaler", "metformin") inside otherwise-Spanish sentences. On the TTS side, VibeVoice-Realtime-0.5B reports ~300 ms first-audible latency on a single T4 or M4 Pro.

## Why Four Specialists

Pushing all safety checks into the primary's system prompt works until the first hard contradiction between checks (e.g., "stay in Spanish" vs. "ask clinical questions precisely"). Giving each check its own prompt, schema, and retry loop means the orchestrator can merge them with an explicit, auditable priority order.

Fan-out is parallel, not pipelined: serializing four ~150 ms specialist calls adds ~600 ms per turn; parallelizing drops the tax to the slowest single call (~200 ms), which keeps the turn under the 2.2 s budget.

## Project Structure

```
constella/
├── README.md
├── SETUP.md
├── how_it_works.md
├── pyproject.toml
├── constella/
│   ├── primary.py            # Nurse Ana (stateful 70B)
│   ├── orchestrator.py       # parallel fan-out + merge → NextAction
│   ├── schemas.py            # Pydantic schemas for state + verdicts
│   ├── llm.py                # provider-agnostic chat + structured_chat
│   ├── cli.py                # terminal chat (constella-chat)
│   ├── specialists/          # 4 verifier agents
│   │   ├── language.py
│   │   ├── medication.py
│   │   ├── labs.py
│   │   └── escalation.py
│   ├── eval/
│   │   ├── rubric.py         # 5-dimension per-turn scoring
│   │   ├── run_eval.py       # constella-eval entrypoint
│   │   ├── scenarios/        # 5 scenario JSONs + patient_maria.json
│   │   └── results/          # run reports
│   └── demo/
│       └── app.py            # Gradio voice UI (constella-demo)
└── tests/
    └── test_schemas.py       # schema + orchestrator merge tests (no network)
```

## Tradeoffs and What I'd Do With More Time

- **Re-verify rewrites.** The MVP trusts the rewrite after a medication-harm flag; a production version would re-run the specialists on the rewrite and escalate on a second failure.
- **Expand the scenario set.** 5 scenarios is enough to exercise every branch of the orchestrator; a real eval would extend into the low hundreds with LLM-as-judge scoring and a sampled clinician spot-check.
- **Wire the Gradio demo through VibeVoice.** The demo uses Groq Whisper + gTTS for portability; adding a self-hosted VibeVoice ASR/TTS path (`VibeVoice-ASR-7B` + `VibeVoice-Realtime-0.5B`) would be the next step for GPU deployments.
- **Observability.** Per-call structured logs exist but there's no dashboard. A lightweight trace viewer (LangSmith or a local Streamlit board) would make the specialist verdicts legible at a glance.
- **More specialists.** Privacy/PHI redaction, scheduling, and social-determinants specialists are the obvious next additions.

## Acknowledgements

- Polaris safety-constellation pattern — Mukherjee et al., [arXiv:2403.13313](https://arxiv.org/abs/2403.13313)
- Microsoft VibeVoice — [github.com/microsoft/VibeVoice](https://github.com/microsoft/VibeVoice)
- Groq LPU inference for Llama 3.1 / 3.3

## License

MIT. See [LICENSE](LICENSE).
