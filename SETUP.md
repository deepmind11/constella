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

This installs LangGraph, Pydantic, Groq client, Gradio, and the eval/test deps. It does NOT yet install VibeVoice — that goes in step 4.

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

If you're on Mac and can't run the 7B ASR locally, Constella will fall back to `distil-whisper-large-v3` for development. The README acknowledges this and frames VibeVoice-ASR as the production target.

## Step 5 — Verify Constella imports

```bash
cd /Users/hgz/Projects/constella
source .venv/bin/activate
uv run python -c "from constella.orchestrator import build_graph; print('Constella imports OK')"
```

## Step 6 — Run the smoke test

```bash
uv run pytest tests/test_specialists.py -v
```

This runs each specialist against a known input/output pair. If all 4 specialists pass, the LLM plumbing works.

## Step 7 — Run a single end-to-end scenario

```bash
uv run python -m constella.demo.app --scenario eval/scenarios/03_codeswitch_metformin.json
```

This walks through one synthetic post-discharge call end-to-end. Expected output:
- Whisper/VibeVoice ASR transcription of the patient utterance
- Language specialist verdict (with code-switch segments)
- Primary agent's response
- 4 specialist verdicts in parallel
- Orchestrator decision (emit / rewrite / escalate)
- VibeVoice TTS audio file
- Latency log entry

## Step 8 — Run the full eval

```bash
uv run constella-eval --output eval/results/run_$(date +%Y%m%d_%H%M%S)
```

This runs all 15 scenarios, scores them with the rubric, and writes:
- `eval_results.csv` — per-turn detailed scores
- `eval_summary.md` — aggregated by bucket, ready to paste into the README

## Troubleshooting

### "VibeVoice import fails on Mac"
Mac doesn't have CUDA. VibeVoice-Realtime-0.5B should work on MPS (Apple Silicon). If it doesn't:
1. Make sure you're on PyTorch 2.5+ with MPS support
2. Try `PYTORCH_ENABLE_MPS_FALLBACK=1 python ...`
3. Fall back to `distil-whisper` for ASR by setting `CONSTELLA_ASR_BACKEND=distil_whisper` in `.env`

### "Groq API rate limit"
Free tier is 30 req/min. Eval harness rate-limits itself; if you hit issues, set `CONSTELLA_GROQ_RPM=20` in `.env`.

### "Specialist returns malformed JSON"
Llama 3.1 8B occasionally drifts off-schema. The specialist code wraps Pydantic with retries. If it fails 3x, the specialist returns a "no verdict" result and the orchestrator logs a warning.

### "VibeVoice Spanish sounds unnatural"
Spanish is "experimental" per Microsoft's docs. Document this honestly in the README. The point of Constella is the architecture, not the voice quality.

## Next steps after setup

1. Read [how_it_works.md](how_it_works.md) (private, gitignored) — your interview prep doc
2. Run the eval harness, paste results into the README
3. Record a 60-second demo video to `demo/demo.mp4`
4. Push to https://github.com/deepmind11/constella
5. Reference Constella in:
   - Hippocratic AI Engineer cover letter
   - Cold InMail to Margaret Urban (after verifying her current employer)
   - Cold InMail to Subho Mukherjee (CSO)
   - Recruiter messages
