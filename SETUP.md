# Setup Guide

> This file walks you from a clean Mac to a runnable Constella in ~30 minutes (excluding model downloads).

## Prerequisites

- macOS (Apple Silicon recommended for VibeVoice-Realtime-0.5B) or Linux with NVIDIA GPU
- Python 3.11 (NOT 3.13 yet — VibeVoice deps lag)
- Git
- ~30 GB free disk (VibeVoice model weights are large)
- A [Groq API key](https://console.groq.com) (free tier works for the entire build)

## Step 1 — Install uv (fast Python package manager)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

## Step 2 — Set up the Constella Python environment

```bash
cd /Users/hgz/Projects/constella
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

This installs Pydantic, the Groq / OpenAI clients, Gradio, and the eval/test deps. It does NOT yet install VibeVoice — that goes in step 4.

## Step 3 — Set environment variables

Create `.env` in the project root:

```bash
cat > .env << 'EOF'
GROQ_API_KEY=your_groq_key_here
ANTHROPIC_API_KEY=optional_for_llm_judge_eval
HF_HOME=/Users/hgz/.cache/huggingface
EOF
```

## Step 4 — Install VibeVoice (the critical step)

VibeVoice is installed as a sibling editable git repo so we can read its source and follow demo scripts.

```bash
cd /Users/hgz/Projects/constella
mkdir -p external
cd external
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
```

### 4a — Install the streaming TTS dependencies

```bash
pip install -e ".[streamingtts]"
```

### 4b — Download experimental Spanish speakers

```bash
bash demo/download_experimental_voices.sh
```

This downloads the experimental multilingual speakers added Dec 16, 2025, including Spanish. The script downloads to `voices/` inside the VibeVoice repo.

### 4c — Verify VibeVoice TTS works

Run the Microsoft-provided realtime inference smoke test:

```bash
python demo/realtime_model_inference_from_file.py \
  --model_path microsoft/VibeVoice-Realtime-0.5B \
  --txt_path demo/text_examples/1p_vibevoice.txt \
  --speaker_name Carter
```

Expected output: a `.wav` file in the current directory containing English speech. First-audible-latency should be ~300 ms on M4 Pro.

### 4d — (Optional) Verify VibeVoice ASR works

The ASR model is 7B params and ideally needs a GPU. If you have a CUDA box:

```bash
python demo/vibevoice_asr_inference_from_file.py \
  --model_path microsoft/VibeVoice-ASR \
  --audio_files <some_audio.wav>
```

If you're on Mac and can't run the 7B ASR locally, the Gradio demo still works — it uses Groq-hosted Whisper for ASR via API, so no local GPU is required. VibeVoice-ASR-7B is documented as the production target for future self-hosted deployments.

## Step 5 — Verify Constella imports

```bash
cd /Users/hgz/Projects/constella
source .venv/bin/activate
uv run python -c "from constella.orchestrator import run_turn; print('Constella imports OK')"
```

## Step 6 — Run the smoke test

```bash
uv run pytest tests/test_schemas.py -v
```

This runs schema validation and the orchestrator merge-logic tests (no network calls, safe to run without API keys). If all tests pass, the Pydantic plumbing and merge priority are wired up correctly.

## Step 7 — Run a single end-to-end scenario

```bash
uv run constella-eval --scenario 03
```

This walks through one synthetic post-discharge call end-to-end (the `03_code_switch_inhaler` scenario by default). Expected output per turn:
- Language specialist verdict (with code-switch segments)
- Primary agent's response
- 4 specialist verdicts run in parallel
- Orchestrator decision (emit / rewrite / append / escalate)
- Latency breakdown

Or run the Gradio voice demo to interact live:

```bash
uv run constella-demo
```

## Step 8 — Run the full eval

```bash
uv run constella-eval
```

This runs all 5 scenarios, scores them with the rubric, and writes to `constella/eval/results/<timestamp>.md` (human-readable report) and `constella/eval/results/<timestamp>.json` (machine-readable score dump).

## Troubleshooting

### "Groq API rate limit"
The free tier has a tokens-per-minute ceiling that a full eval run can hit in bursts. The eval harness pauses between scenarios by default (30 s on Groq, 0 s on OpenRouter). To change it, set `CONSTELLA_EVAL_INTER_SCENARIO_SLEEP=45` in `.env`.

### "Specialist returns malformed JSON"
Llama 3.1 8B occasionally drifts off-schema. The specialist code wraps Pydantic with retries. If it fails 3x, the specialist returns a "no verdict" result and the orchestrator logs a warning.

### "VibeVoice Spanish sounds unnatural"
Spanish is "experimental" per Microsoft's docs. Document this honestly in the README. The point of Constella is the architecture, not the voice quality.

## Next steps after setup

1. Run the eval harness and paste results into the README results table
2. Read [how_it_works.md](how_it_works.md) for a detailed technical walkthrough of each architectural decision
3. Record a demo video (`demo/demo.mp4`) of a code-switching scenario end-to-end
