# Realtime Voice — Backend Architecture

> **Status:** design doc. Nothing in this document is implemented yet on this branch (`feat/realtime-voice`). This is the plan we're reviewing before writing code.

## 1. Goal

Replace the current record-send-reply Gradio UX with a phone-call feel:

- Audio streams continuously in both directions over WebRTC.
- Server-side VAD decides when the patient has stopped talking.
- Ana's voice starts speaking 300-500 ms after the patient stops — not 2-3 s.
- The patient can interrupt Ana mid-sentence (barge-in).
- The **existing 4-specialist constellation is preserved** — it's Constella's whole point. We just swap out the I/O layer.

What this is **not**: a rewrite into an end-to-end speech-to-speech model (OpenAI Realtime, Gemini Live). Those collapse the text LLM layer and would bypass the constellation entirely.

## 2. Target stack

| Stage | Today | Target | Why the swap |
|---|---|---|---|
| Transport | Gradio `gr.Audio` push-to-talk | WebRTC (Pipecat small-webrtc transport) | Bidirectional streaming, barge-in support |
| VAD | None — user clicks stop | Silero VAD (local, runs in Pipecat) | No click, automatic turn boundaries |
| ASR | Groq Whisper (batch) | Deepgram `nova-3` streaming | Streaming partials, ~150 ms finalization after EOT |
| Primary LLM | Groq Llama 3.3 70B (sync) | Groq Llama 3.3 70B (streaming) | Stream tokens into TTS instead of waiting for full reply |
| Specialists | Groq Llama 3.1 8B ×4, parallel | **unchanged** — Groq Llama 3.1 8B ×4, parallel | Constellation logic stays identical |
| TTS | gTTS (batch MP3) | Cartesia Sonic-2 multilingual (streaming) | ~90 ms first-audible, real multilingual (en/es) |
| Orchestration | single `run_turn()` call | Pipecat `Pipeline` of `FrameProcessor`s | Native streaming, interruption, turn-taking |

**Keeping Groq for LLMs** because: (a) the constellation pattern has zero reason to change providers, (b) Groq's LPU latencies (~250 ms TTFT 70B, ~80 ms TTFT 8B) are what make streaming worth doing.

## 3. Pipeline topology

Pipecat organizes the runtime as a linear chain of `FrameProcessor`s. Audio frames, transcription frames, LLM frames, and TTS audio frames flow through in order. Our chain:

```
  Mic (WebRTC)
       │ AudioRawFrame (20 ms chunks)
       ▼
  ┌─────────────────────────┐
  │ SileroVADAnalyzer       │  decides when user started/stopped speaking
  └─────────────┬───────────┘
                │ UserStartedSpeakingFrame / UserStoppedSpeakingFrame
                ▼
  ┌─────────────────────────┐
  │ DeepgramSTTService       │  streaming ASR, emits partial + final
  └─────────────┬───────────┘
                │ TranscriptionFrame (final, on VAD stop)
                ▼
  ┌─────────────────────────────────────────────────┐
  │ ConstellationProcessor  ← THE NEW THING          │
  │                                                  │
  │  1. Append patient turn to ConversationState     │
  │  2. Fan out in parallel:                         │
  │       - primary.primary_respond() (streaming)    │
  │       - language_specialist()                    │
  │       - labs_specialist()                        │
  │       - escalation_specialist()                  │
  │       - (medication waits on primary's draft)    │
  │  3. Await medication specialist on primary draft │
  │  4. Merge verdicts → NextAction                  │
  │  5. Rewrite if needed (blocks TTS)               │
  │  6. Emit final TextFrame(s) for TTS              │
  │  7. Append nurse turn to ConversationState       │
  └─────────────┬───────────────────────────────────┘
                │ TextFrame (nurse reply)
                ▼
  ┌─────────────────────────┐
  │ CartesiaTTSService       │  streaming TTS, emits audio chunks
  └─────────────┬───────────┘
                │ TTSAudioRawFrame
                ▼
  Speaker (WebRTC)
```

The **critical design decision** is step 2: parallelism between primary and specialists.

### 3a. Where the constellation plugs in

Today `run_turn()` in `orchestrator.py` is a single synchronous function. In Pipecat it becomes a `FrameProcessor` subclass (`ConstellationProcessor`). The internals do the same work — primary + 4 specialists + merge — but the return is streamed as `TextFrame`s instead of returned as a string.

```
constella/
├── orchestrator.py          # unchanged — run_turn() stays, used by eval harness
├── realtime/                # NEW
│   ├── __init__.py
│   ├── pipeline.py          # Pipecat pipeline setup + transport wiring
│   ├── constellation.py     # ConstellationProcessor (FrameProcessor subclass)
│   └── session.py           # Per-call ConversationState, lifecycle hooks
└── demo/
    ├── app.py               # existing Gradio demo — stays, labelled "legacy"
    └── realtime.py          # NEW constella-realtime entrypoint
```

Why a separate folder instead of modifying `orchestrator.py`: the eval harness (`constella-eval`) calls `run_turn()` synchronously and writes per-turn latency to a JSON report. Keeping it sync-friendly means eval keeps working without any changes.

## 4. Running the specialists: parallelism strategy

Three options, in increasing complexity:

### Option A — constellation-then-speak (MVP)

Run the whole `run_turn()` to completion, then stream the final text into TTS. Simplest mental model; easiest rollback if specialists hang.

**Perceived latency (best case, Groq):**

| Stage | Time |
|---|---|
| Deepgram ASR finalization after EOT | 150 ms |
| Primary (70B, full response) | 500 ms |
| Specialists in parallel (max) | 250 ms (overlaps with primary, effectively 0 ms added) |
| Rewrite if needed | +400 ms (rare path) |
| Cartesia TTS first audible | 90 ms |
| **Total EOT → Ana's first audio** | **~740 ms happy path** |

That's 3× better than today (~2.5 s) and below the 1 s threshold where conversation stops feeling snappy.

### Option B — stream primary, run specialists concurrently, interrupt on flag

Start TTS on primary's first tokens. Specialists run in parallel with TTS playback. If escalation/medication fires a flag, cut TTS mid-word with `BotInterruptionFrame` and replace with rewrite/escalation text.

**Perceived latency:** ~300-400 ms EOT → first audio. Near-Gemini-Live feel.

**Cost:** patient briefly hears the unsafe draft before it gets cut off. For a medication harm case (e.g., "take 200 mg of …" [cut off]), that's arguably worse than silence — it teaches the wrong dose for half a second.

### Option C — hybrid: gate TTS on safety-critical specialists only

- Start specialists on patient utterance immediately (language, labs, escalation run on patient text — no dependency on primary).
- Primary streams into a buffer, not directly to TTS.
- TTS starts as soon as **medication specialist returns safe** AND **escalation specialist returns non-urgent**.
- Labs followup appended at end, no TTS gate needed.

Roughly: if max(medication, escalation) < primary generation time, zero added latency over Option B — but with safety preserved.

### Recommendation

**Ship Option A first.** It's a 4× UX improvement with the lowest implementation risk and no new safety questions. Once that's stable, Option C is a latency optimization that can be added incrementally — the `ConstellationProcessor` interface doesn't change.

## 5. Session state

Today: `ConversationState` lives in Gradio's `gr.State` and gets serialized/deserialized per turn.

New: `ConversationState` lives inside the `ConstellationProcessor` instance. One processor = one call. When the WebRTC session ends, the processor is destroyed. For multi-call history (not in scope for the MVP) we'd add a Redis-backed session store.

The recent `[-20:]` history-window change applies automatically — the processor just calls `primary_respond(state, utterance)` with the accumulating state, same as today.

## 6. Interruption / barge-in

Pipecat handles this natively:

1. While TTS is playing, VAD detects user speech.
2. Pipeline emits `UserStartedSpeakingFrame`.
3. Cartesia TTS receives `BotInterruptionFrame`, cancels current synthesis, flushes audio buffer.
4. Speaker goes silent within ~100 ms.
5. Deepgram starts transcribing the new utterance.

No custom code needed beyond wiring these standard frames. The `ConstellationProcessor` needs to handle being mid-generation when interrupted — it should cancel the in-flight primary/specialist calls (their HTTP connections) and move on to the new patient utterance.

## 7. Failure modes

| Failure | Behavior |
|---|---|
| Deepgram WebSocket drops | Pipecat auto-reconnects; transient loss of a partial transcript, final still fires on EOT |
| Specialist returns `None` (structured validation failed 3x) | Same as today — treat as "no verdict", orchestrator logs warning |
| Primary LLM timeout | Fall back to a hard-coded "I didn't catch that — could you repeat?" safety phrase, don't surface an error to the user |
| Cartesia TTS fails | Fall back to gTTS synchronously (slow, but better than silence); log alert |
| VAD false trigger on background noise | Deepgram gets empty audio; the final `TranscriptionFrame` is `""` → `ConstellationProcessor` skips the turn |

## 8. Env vars / API keys

| Key | Provider | Purpose | Free tier? |
|---|---|---|---|
| `GROQ_API_KEY` | Groq | LLMs (unchanged) | yes |
| `DEEPGRAM_API_KEY` | Deepgram | Streaming ASR | yes, $200 free credit |
| `CARTESIA_API_KEY` | Cartesia | Streaming TTS | yes, free tier covers dev |

Both Deepgram and Cartesia sign-ups are ~2 minutes each. You'll hand me those keys before Phase 3 below.

## 9. Known tradeoffs vs today

- **Spanglish code-switching on ASR.** Deepgram's `language=multi` detects one language per utterance, not mid-utterance switches. This is a regression vs. Groq Whisper, which handles Spanglish well. Mitigation: use Deepgram streaming for the live feel, then re-transcribe the final audio buffer with Groq Whisper for the canonical transcript that feeds specialists. Costs ~200 ms on EOT; acceptable. Alternative: evaluate Gladia Solaria, which markets Spanglish as a first-class feature.
- **No multilingual gTTS-style free path.** Cartesia is paid (but cheap). gTTS stays as a fallback.
- **Gradio text-input fallback.** The current demo's "type instead of speak" path is useful for dev. The new realtime page will have a text-input box too, but it bypasses ASR and feeds directly into the `ConstellationProcessor`.

## 10. Implementation phases

1. **Scaffold Pipecat dependencies.** Add `pipecat-ai[deepgram,cartesia,silero,webrtc]` to pyproject, pin versions, lock uv.
2. **Wire `ConstellationProcessor`.** Wrap `run_turn()`. Keep Option A (no streaming primary) for v1. Unit-test with a fake Pipecat pipeline that injects `TranscriptionFrame`s.
3. **Stand up the WebRTC transport.** Pipecat's `SmallWebRTCTransport` serves a static HTML page that opens a WebRTC peer connection to the Python server.
4. **Replace `constella-demo` default route.** `constella-demo` points at the new realtime page; `constella-demo --legacy` keeps the current Gradio app.
5. **Smoke-test server boot + text-input path.** No mic needed — verify the pipeline processes a typed patient utterance end-to-end and produces audio out.
6. **User validates the live voice loop.** You speak, listen, report. I iterate on VAD thresholds, TTS voice selection, etc.
7. **(Later) Option C latency optimization.** Gate TTS on medication + escalation specialists, not the full merge.

## 11. What stays the same

- `constella/orchestrator.py`, `constella/primary.py`, `constella/specialists/*.py`, `constella/schemas.py` — zero changes.
- `constella/eval/` — zero changes, still batch-synchronous.
- `tests/test_schemas.py` — zero changes.

This is the point: the whole constellation is an I/O-agnostic function. The realtime work is purely a transport + streaming layer around it.

## 12. Open questions for review

1. **Option A vs C for MVP?** I recommend A. Confirms?
2. **Spanglish ASR fallback to Groq Whisper on EOT — acceptable 200 ms overhead?**
3. **Keep the Gradio demo parallel to the new one, or retire it?** I recommend keeping it as `constella-demo --legacy` for dev testing.
4. **Deepgram vs Gladia for ASR?** Deepgram for MVP (larger ecosystem, better docs); revisit if Spanglish accuracy is a real problem.
