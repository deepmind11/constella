# Constella

> A multi-agent voice constellation for bilingual healthcare follow-up calls, with first-class English-Spanish code-switching support. Built on Microsoft VibeVoice end-to-end.

**Status:** Active build (started 2026-04-07). 2-day sprint.
**Author:** Harshit Ghosh
**License:** Apache 2.0

---

## What this is

Constella is a miniature, runnable, evaluable implementation of a Polaris-style safety constellation for voice-based healthcare AI agents. One stateful primary conversational agent is flanked by four specialist verifier agents (medication, labs and vitals, language detection, escalation) running in parallel after every primary turn. The whole pipeline uses **Microsoft VibeVoice** for both ASR (`VibeVoice-ASR-7B`) and TTS (`VibeVoice-Realtime-0.5B`).

The use case is intentionally narrow: post-discharge follow-up calls for a Type 2 diabetic patient who code-switches between English and Spanish.

## Why I built this

I read the [Polaris paper](https://arxiv.org/abs/2403.13313) (Mukherjee et al., 2024) and the [WellSpan colorectal cancer screening study](https://www.medrxiv.org/content/10.1101/2024.12.16.24318586v1) (Bhimani and Baker, 2024) and noticed two things:

1. **Polaris 3.0 reports 99.83% Spanish accuracy** and supports 14 languages with real-time multilingual auto-switch. Hippocratic AI's Spanish-language voice agent achieved a **2.6x higher FIT-test opt-in rate** among Spanish-speaking patients vs English-speaking patients in the WellSpan study. Bilingual healthcare AI is producing real clinical-economic impact, not just demos.
2. **Intra-utterance code-switching** (where a bilingual speaker mixes English and Spanish *within a single sentence*: "¿Tomo el inhaler dos veces al día, but only when I feel short of breath?") is endemic in U.S. Latino patient populations, but I could not find an explicit treatment of it in any of the published Polaris papers.

Around the same time, Microsoft released [VibeVoice](https://github.com/microsoft/VibeVoice), the first open-source ASR model that natively handles **code-switching across 50+ languages with no language parameter required**. I wanted to know: what happens if you build a Polaris-style constellation entirely on VibeVoice? Does it handle code-switching gracefully? How does end-to-end latency compare to Hippocratic's published 2.2-sec target?

Constella is my attempt to answer these questions in a 2-day build.

## Architecture

```
                                                       ┌──────────────────────────┐
                                                       │  Eval harness            │
                                                       │  (15 scenarios)          │
                                                       └────────────┬─────────────┘
                                                                    │
                                                                    ▼
┌───────────┐    audio    ┌──────────────┐  text    ┌──────────────────────────────┐
│  Patient  │────────────▶│  VibeVoice    │─────────▶│   Primary Conversational     │
│  speaks   │             │  ASR-7B       │          │   Agent (Llama 3.1 70B)      │
│  (mic or  │             │  (50+ langs,  │          │   "Nurse Constella"          │
│   file)   │             │  code-switch  │          │   - manages dialogue state   │
└───────────┘             │  native)      │          │   - generates next utterance │
      ▲                   └──────────────┘          └──────────────┬───────────────┘
      │                                                            │
      │                                            (parallel fan-out)
      │                                                            │
      │            ┌──────────────────┬──────────────────┬─────────┴────────┬──────────────────┐
      │            │                  │                  │                  │                  │
      │            ▼                  ▼                  ▼                  ▼                  ▼
      │     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │     │ Language     │  │ Medication   │  │ Labs & Vitals│  │ Escalation   │  │ (future:     │
      │     │ Specialist   │  │ Specialist   │  │ Specialist   │  │ Specialist   │  │ Privacy /    │
      │     │ - lang ID    │  │ - dosage     │  │ - extract    │  │ - red-flag   │  │ EHR / etc)   │
      │     │ - code-switch│  │ - drug-drug  │  │   numbers    │  │   symptoms   │  │              │
      │     │ - route to   │  │ - timing     │  │ - check range│  │ - human XFR  │  │              │
      │     │   ES/EN/MIX  │  │              │  │              │  │              │  │              │
      │     └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────────┘
      │            │                  │                 │                 │
      │            └──────────────────┴─────────────────┴─────────────────┘
      │                                       │
      │                                       ▼
      │                       ┌──────────────────────────────┐
      │                       │  Orchestrator merges          │
      │                       │  specialist verdicts.         │
      │                       │  - If escalation: hard XFR    │
      │                       │  - If med error: rewrite turn │
      │                       │  - Else: emit primary's turn  │
      │                       └──────────────┬───────────────┘
      │                                      │
      │                                      ▼
      │                       ┌──────────────────────────────┐
      │                       │  VibeVoice Realtime-0.5B TTS │
      │                       │  (~300 ms first audible      │
      │                       │  latency, multilingual,      │
      │                       │  Spanish experimental)        │
      │                       └──────────────┬───────────────┘
      │                                      │
      └──────────────────────────────────────┘
```

## The 4 specialist agents

| Specialist | Role | Model |
|---|---|---|
| **Language** | Per-utterance language ID + intra-utterance code-switch segmentation. Routes downstream prompts to ES, EN, or MIX. | Llama 3.1 8B (Groq) |
| **Medication** | Verifies dosage, timing, contraindications against the synthetic patient's med list. | Llama 3.1 8B (Groq) |
| **Labs & Vitals** | Extracts numeric values from patient utterances ("My sugar was 240 this morning") and flags out-of-range. | Llama 3.1 8B (Groq) |
| **Escalation** | Detects red-flag symptoms (chest pain, severe hypoglycemia, DKA, suicidal ideation) and triggers human handoff. | Llama 3.1 8B (Groq) |

## Quick start

See [SETUP.md](SETUP.md) for the full bootstrap. The short version:

```bash
# 1. Clone Constella
git clone https://github.com/deepmind11/constella.git
cd constella

# 2. Set up Python env with uv
uv venv
source .venv/bin/activate
uv pip install -e .

# 3. Clone and install VibeVoice as a sibling repo
mkdir -p external && cd external
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -e .[streamingtts]
bash demo/download_experimental_voices.sh   # downloads Spanish experimental speakers
cd ../..

# 4. Set Groq API key
echo 'GROQ_API_KEY=your_key_here' > .env

# 5. Run a single scenario end-to-end
uv run python -m constella.demo.app --scenario eval/scenarios/03_codeswitch_metformin.json

# 6. Run the full eval harness
uv run constella-eval
```

## Use case

Post-discharge follow-up call for **María González**, a fictional 58-year-old Type 2 diabetic patient recently discharged with a metformin + insulin regimen. María prefers Spanish but switches to English for medication names — a pattern documented in the bilingual healthcare communication literature. Her medication list, lab values, and discharge summary are synthetic and committed in `eval/scenarios/patient_maria.json`.

## Eval harness

15 synthetic post-discharge scenarios in three buckets:

- **English-only** (scenarios 01-05): baseline accuracy + latency
- **Spanish-only** (scenarios 06-10): Spanish accuracy + latency
- **Code-switching** (scenarios 11-13): the key bucket
- **Escalation** (scenario 14): patient mentions chest pain in Spanish
- **Medication safety** (scenario 15): patient asks about double-dosing in mixed language

Each turn is scored on 5 dimensions: medical accuracy, harm rate, language correctness, escalation correctness, latency.

Results table (will be filled in as the eval runs):

```
| Bucket          | Single LLM | Constella | Δ      |
|-----------------|-----------:|----------:|-------:|
| English-only    |        TBD |       TBD | TBD    |
| Spanish-only    |        TBD |       TBD | TBD    |
| Code-switching  |        TBD |       TBD | TBD    |
| Median latency  |        TBD |       TBD | TBD    |
```

## Hardware

- **Dev (Mac M-series):** VibeVoice-Realtime-0.5B for TTS runs on M4 Pro at realtime per [Microsoft's docs](https://github.com/microsoft/VibeVoice/blob/main/docs/vibevoice-realtime-0.5b.md). For ASR, the dev path uses a smaller fallback (distil-whisper-large-v3) with VibeVoice-ASR-7B as the production target.
- **Prod / cloud:** A single A10 or T4 with VibeVoice-ASR-7B for end-to-end VibeVoice inference. NVIDIA Deep Learning Container 24.10+ recommended per Microsoft docs.

## What this is NOT

- **Production-grade.** A 2-day build cannot match the operational rigor of a $3.5B unicorn running on H200s with TensorRT-LLM, NVIDIA NIM, and SageMaker HyperPod.
- **A medical product.** Constella is a non-diagnostic engineering demo with synthetic patients. **No PHI ever touches the system.**
- **A novel research contribution.** Constella builds on the constellation pattern from Mukherjee et al. (Polaris, 2024), the multilingual code-switching capability from Microsoft VibeVoice (2025), and the WellSpan colorectal study from Bhimani and Baker (2024).

## What I would love to discuss

1. How does Polaris handle intra-utterance code-switching in production today?
2. The Language Specialist routing pattern used here: does it map to anything in the Polaris stack?
3. The Grove AI acquisition opens a new product surface (clinical trial recruitment). My background is in oncology cfDNA pipeline engineering at BillionToOne. How is the new Life Sciences Division thinking about integrating bioinformatics data into Grace?

## References

1. **Polaris 1.0:** Mukherjee, S. et al. (2024). *Polaris: A Safety-focused LLM Constellation Architecture for Healthcare.* arXiv:2403.13313. https://arxiv.org/abs/2403.13313
2. **WellSpan colorectal Spanish study:** Bhimani, M. and Baker, J. (2024). *Multilingual AI Care Agent for Colorectal Cancer Screening.* medRxiv 2024.12.16.24318586. https://www.medrxiv.org/content/10.1101/2024.12.16.24318586v1
3. **Polaris 3.0 announcement:** Hippocratic AI (2025). *Polaris 3.0: A 4.2 Trillion Parameter Suite of 22 LLMs.* https://hippocraticai.com/polaris-3/
4. **VibeVoice:** Microsoft (2025). *VibeVoice: Open-Source Frontier Voice AI.* https://github.com/microsoft/VibeVoice
5. **RWE-LLM evaluation framework:** Hippocratic AI (2025). *Real-World Evaluation of LLMs in Healthcare.* medRxiv 10.1101/2025.03.17.25324157.

## Author

**Harshit Ghosh** — bilingual bioinformatics engineer with three years building production data pipelines and multi-agent systems in regulated healthcare environments. Previously at BillionToOne (oncology cfDNA, NorthStar Select Assay) and Columbia University (computational biology). Spanish at professional working level. Reach me at harshitghosh@gmail.com or [LinkedIn](https://linkedin.com/in/harshit-ghosh).
