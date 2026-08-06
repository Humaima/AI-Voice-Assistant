# Conversational Voice AI Assistant — LangGraph + Groq + ElevenLabs

Voice-in, voice-out WhatsApp assistant. Matches the architecture diagram:
WhatsApp voice note → chunking/buffering → Whisper-large-v3 (Groq) →
LangGraph agent (memory + Llama-3.3-70B) → ElevenLabs TTS → voice reply.

This repo is being built **phase by phase**, matching the implementation
plan. Each phase is a working increment you can run and test before the
next one is layered on.

## Phase status

| Phase | Name                     | Status |
|-------|--------------------------|--------|
| 1     | Project Foundation       | ✅ Done |
| 2     | Voice Input Pipeline     | ✅ Done |
| 3     | Speech Recognition       | ✅ Done |
| 4     | LangGraph Agent          | ✅ Done |
| 5     | Conversational Memory    | ✅ Done |
| 6     | Response Generation      | ✅ Done (this delivery) |
| 7     | Voice Generation         | ✅ Done (this delivery) |
| 8     | Audio Streaming          | ✅ Done (this delivery) |
| 9     | WhatsApp Integration     | ✅ Done (this delivery) |
| 10    | Deployment               | ✅ Done (this delivery) |

## Project structure

```
voice-ai-assistant/
├── app/
│   ├── main.py              # FastAPI entrypoint, health check
│   ├── core/
│   │   ├── config.py        # Settings loaded from .env (pydantic-settings)
│   │   └── logging_config.py
│   ├── api/                 # HTTP routes (webhook lands here in Phase 9)
│   ├── agents/               # LangGraph nodes/graph (Phase 4-6)
│   ├── services/             # Whisper, ElevenLabs, audio processing clients
│   ├── models/                # Pydantic schemas / ORM models
│   └── db/                    # SQLAlchemy session, ChromaDB client
├── tests/
├── docker/Dockerfile
├── docker-compose.yml         # app + postgres + redis + chromadb
├── requirements.txt
└── .env.example
```

## Running Phase 1 locally

1. Copy the env file and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   **Windows PowerShell:** `cp` is aliased to `Copy-Item` but the extension-less-to-named copy sometimes silently no-ops — use explicitly:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Start everything with Docker Compose:
   ```bash
   docker compose up --build
   ```
   **Windows:** requires Docker Desktop to be running first (open it from the Start menu and wait for the whale icon in the system tray to stop animating) and set to **Linux containers** mode. If you see an error like `dockerDesktopLinuxEngine: The system cannot find the file specified`, Docker Desktop isn't fully started yet.
3. Check the health endpoint:
   ```bash
   curl http://localhost:8000/health
   # {"status":"ok","app":"voice-ai-assistant","environment":"development"}
   ```

Or without Docker, for quick local iteration:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
This is enough to test Phases 2 and 3 — they don't touch Postgres/Redis/ChromaDB, so Docker isn't required until later phases (memory, deployment).

**Running tests on Windows:** if `pytest` alone raises `ModuleNotFoundError: No module named 'app'`, either run `python -m pytest tests/ -v` instead, or rely on the `pyproject.toml` in this repo (already configured with `pythonpath = ["."]`) which fixes this for plain `pytest` too.

## What Phase 1 gives you

- FastAPI app with a working `/health` endpoint and lifespan hooks
  ready for DB/Redis/Chroma connection pools (wired up in later phases).
- Centralized, typed settings (`app/core/config.py`) — every API key
  and connection string the whole project needs is already listed in
  `.env.example`, even though most services aren't built yet. This
  keeps you from hunting down config in five different files later.
- Structured logging via stdlib `logging`, correctly leveled.
- Docker Compose stack with Postgres, Redis, and ChromaDB already
  wired up and health-checked, so Phase 5 (memory) and Phase 10
  (deployment) build on infra that's already proven to work.

## Phase 2 — Voice Input Pipeline

`app/services/audio_processor.py` runs the full pipeline:

1. **Validate** — format allow-list (ogg/opus/mp3/wav/m4a/amr), size cap, empty-payload check.
2. **Load & convert** — ffmpeg-backed decode (via pydub) → resample to 16kHz mono (what Whisper expects).
3. **Normalize** — consistent volume regardless of how loud/quiet the original recording was.
4. **Remove silence** — trims leading/trailing/internal silence (`pydub.silence.detect_nonsilent`); rejects clips that are entirely silence.
5. **Chunk** — splits into 30s chunks with 500ms overlap (so words aren't cut at chunk boundaries) — this is the "Chunking Voice Note" step in the diagram.

`app/services/audio_buffer.py` holds the resulting chunks (`AudioBuffer`/`BufferedChunk`), exports each as WAV bytes ready for Whisper, and tracks which chunks have been consumed — this is the "Audio Buffer" box in the diagram, and the consumption tracking is what Phase 8 (streaming) will build on.

### Try it now (before Phase 9's Twilio webhook exists)

A debug endpoint lets you exercise the whole pipeline via file upload:

```bash
curl -X POST http://localhost:8000/audio/process-test \
  -F "file=@/path/to/voice_note.ogg"
```

Or use the Swagger UI at `http://localhost:8000/docs`.

### Tests

```bash
pip install -r requirements.txt
pytest tests/test_audio_processor.py -v
```

12 tests cover format validation, resampling/downmixing, silence trimming, single- vs. multi-chunk splitting, and buffer consumption tracking — using synthetically generated tones so no audio fixture files are needed.

## Phase 3 — Speech Recognition

`app/services/transcription.py` sends each buffered chunk (from Phase 2's `AudioBuffer`) to Whisper-large-v3 on Groq and stitches the results:

1. **`TranscriptionService.transcribe_chunk`** — calls Groq's `/audio/transcriptions` with `response_format="verbose_json"` (so we get back language + segment-level confidence, not just plain text), `temperature=0.0` for deterministic output.
2. **Retry with backoff** — `RateLimitError`, `APIConnectionError`, `APITimeoutError`, and `InternalServerError` are retried with exponential backoff (configurable via `transcription_max_retries` / `transcription_backoff_base_seconds`). `BadRequestError`/auth errors fail immediately since retrying won't fix a malformed request.
3. **Stitching** — chunks were built with a 500ms overlap (Phase 2) so words aren't cut at chunk boundaries. `stitch_transcripts` detects duplicated words at the boundary using fuzzy matching (`difflib`) and trims them, so a two-chunk transcript like `"...know the neural"` + `"the neural maze..."` becomes one clean sentence instead of repeating "the neural".
4. **`TranscriptionService.transcribe_buffer`** — runs the above across every chunk in an `AudioBuffer`, in order, marking each chunk consumed as it succeeds (so a mid-buffer failure doesn't lose progress on already-transcribed chunks).

### Try it now

```bash
curl -X POST http://localhost:8000/audio/transcribe-test \
  -F "file=@/path/to/voice_note.ogg"
```

This chains Phase 2 (validate/convert/chunk) → Phase 3 (transcribe/stitch) and requires a real `GROQ_API_KEY` in `.env` — unlike Phase 2's `/audio/process-test`, this one makes an actual network call.

### Tests

```bash
pytest tests/test_transcription.py -v
```

13 tests cover chunk transcription, retry/backoff behavior, non-retryable failures, and boundary-word stitching — all against a fake Groq client (`FakeGroqClient`), so no network access or real API key is needed to run them.

## Phase 4 — LangGraph Agent

`app/agents/graph.py` builds the "LangGraph" box from the diagram as a 3-node graph:

```
START -> load_memory -> generate_response -> update_memory -> END
```

- **`load_memory` / `update_memory`** — deliberate no-op stubs for now. Memory currently lives entirely in `AgentState["messages"]`, carried in and out by the caller. The nodes exist as real graph steps today so Phase 5 can give them a body (Postgres for full history, ChromaDB for semantic recall) without rewiring the graph — just swapping what happens inside two existing nodes.
- **`generate_response`** — the "Conversational Chain" box: builds `[system prompt, ...history, new transcript]` and calls Llama-3.3-70B on Groq (`app/agents/llm.py`). Uses LangGraph's `add_messages` reducer (`app/agents/state.py`) so nodes append to history rather than needing to manage the full list themselves.
- **`app/agents/prompts.py`** — Ava's persona, written for *spoken* output specifically: no markdown/bullets/emoji (none of that survives TTS), short replies, natural contractions.
- **`AgentService`** (`app/services/agent_service.py`) — the boundary between the graph and the rest of the app; converts between LangChain's `BaseMessage` types and JSON-friendly `ChatMessageModel` for the API layer.

### Try it now

Text-only, fastest way to iterate on the agent/prompt:
```bash
curl -X POST http://localhost:8000/agent/chat-test \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "test-1", "transcript": "Do you know the neural maze podcast?"}'
```
Response includes `history` — pass it back as `history` in your next request to continue the conversation (no server-side memory until Phase 5).

Full pipeline — voice note in, Ava's text reply out (chains Phases 2+3+4):
```bash
curl -X POST http://localhost:8000/audio/converse-test \
  -F "file=@/path/to/voice_note.ogg"
```

### Tests

```bash
pytest tests/test_agent.py -v
```

8 tests cover graph wiring, prompt construction (system prompt + history sent to the LLM in the right order), multi-turn history accumulation, and role conversion — all against LangChain's built-in `FakeListChatModel`, so no network or API key is needed.

## Phase 5 — Conversational Memory

Gives `load_memory`/`update_memory` (Phase 4's stub nodes) real bodies — no changes to the graph's shape, just what happens inside two existing nodes.

**Postgres (short-term/full history)** — `app/db/models.py` + `app/services/memory/conversation_store.py`
- One `messages` table: conversation_id, role, content, created_at
- `PostgresConversationStore.get_recent_messages(conversation_id, limit)` / `.append_messages(...)`, behind a `ConversationMemory` Protocol (same structural-typing pattern as `TranscriptionClient` from Phase 3) so the graph depends on "anything that can get/append messages," not this concrete class

**ChromaDB (long-term/semantic recall)** — `app/services/memory/vector_store.py`
- `ChromaVectorStore.add_memory(...)` / `.search(...)`, behind a `VectorMemory` Protocol
- After each turn, both the user's message and Ava's reply get embedded and indexed, scoped to that conversation
- Before generating a reply, `load_memory` also queries this for anything semantically relevant to the new message — so if someone mentions their dog's name in turn 2 and asks about it in turn 40 (well outside Postgres's recent-N window), it still surfaces, folded in as a system note

**What changed for callers:** you no longer need to pass `history` back and forth — just reuse the same `conversation_id` across calls and the server remembers. (`history` in `/agent/chat-test` still works but is only honored when no memory store is configured — otherwise the store is authoritative and explicit `history` is ignored, to avoid double-counting.)

Both stores are **optional** in `AgentService`/`build_graph` — unset, everything behaves exactly like Phase 4 (caller carries history, nothing persisted). This is what keeps all of Phase 4's original tests passing unchanged, and lets you skip standing up Postgres/ChromaDB entirely for quick prompt iteration.

### One-time setup

```bash
docker compose up -d postgres chromadb   # or the full stack
python -m app.db.bootstrap               # creates the `messages` table
```
Deliberately *not* run automatically at app startup — Phases 2-4 still work with zero DB configured; this only matters once you actually call an endpoint that uses memory.

**Local app + dockerized infra** (the workflow used through Phase 4) — `.env.example`'s defaults already point at `localhost` with the ports docker-compose exposes to your host (Postgres `5433`, ChromaDB `8001`). Postgres defaults to `5433`, not the usual `5432`, specifically to avoid clashing with a natively-installed Postgres service — common on Windows, where installers often set one up as an auto-starting background service on the standard port, which would otherwise silently intercept connections meant for this project's container (same host/port, different credentials — surfaces as a confusing password-auth or connection-refused error). If you're certain nothing else on your machine uses 5432, you can change `POSTGRES_PORT` and `POSTGRES_HOST_PORT` in `.env` back to `5432` — but there's no real downside to leaving it on `5433`. If you instead run the app itself inside `docker compose` too, `docker-compose.yml` overrides `POSTGRES_HOST`/`CHROMA_HOST` to the container-network hostnames automatically — no manual change needed either way.

### Try it now

```bash
curl -X POST http://localhost:8000/agent/chat-test \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "memory-test", "transcript": "My dogs name is Rex"}'

# ...later, no history needed, same conversation_id:
curl -X POST http://localhost:8000/agent/chat-test \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "memory-test", "transcript": "What is my dogs name?"}'
```
Ava should correctly answer "Rex" on the second call, with no `history` passed at all.

### Tests

```bash
pytest tests/test_conversation_store.py tests/test_vector_store.py tests/test_agent.py -v
```

- `test_conversation_store.py` (7 tests) — real SQLAlchemy async queries against in-memory SQLite (not a mock): ordering, the recency limit, conversation isolation, empty-input edge cases.
- `test_vector_store.py` (6 tests) — real `chromadb.EphemeralClient` with a small deterministic hash-based embedding function (avoids needing a real ML model download in CI): similarity search, conversation isolation, empty-query/empty-content handling.
- `test_agent.py` (13 tests, up from 8) — new `TestAgentServiceWithMemory` class covers history persisting across separate calls without the caller passing it, conversation isolation by ID, explicit `history` correctly being ignored once a store is configured, and the semantic-recall system message actually showing up in a later turn.

51 tests pass across the whole project; `pyright` is clean.

## Phase 6 — Response Generation

Two response-quality improvements layered onto the existing `generate_response` node, plus a new node — no changes to Phase 5's memory wiring.

### 1. Spoken-text sanitization (`app/agents/text_processing.py`)

Ava's system prompt already tells the model not to use markdown — but a prompt is guidance, not a guarantee. `sanitize_for_speech()` is a defensive backstop that runs on every response (and every note) before it's stored or returned:

- Strips `**bold**`, `_italic_`, `# headers`, `- bullets`, `1. numbered lists`, `` `inline code` ``, ` ```code fences``` `, and `[markdown links](url)` (keeping the link text)
- Collapses multi-paragraph structure into natural spoken flow (no literal pauses for line breaks a TTS engine can't represent)
- Idempotent — safe to run more than once on the same text without side effects

Also included: `estimate_speech_duration_seconds()` — a rough words-per-minute estimate, useful for logging now and as a sanity-check once Phase 7 reports actual TTS audio duration.

### 2. "Generate Ava's Note" (`_generate_note_node` in `app/agents/graph.py`)

The diagram's caption step: WhatsApp voice notes can't be skimmed the way text can, so a short caption sent alongside the audio lets someone tell what the reply's about without playing it — like a notification preview.

- Runs as its own small LLM call (not asked for in the same call as the main response), so a verbose or off-topic caption attempt can never leak into or distort the actual spoken reply
- Capped at 12 words by its prompt (`NOTE_SYSTEM_PROMPT`)
- **Graceful degradation**: if the note-generation call fails for any reason, falls back to a simple truncation of the actual response rather than failing the whole turn — a missing caption is cosmetic; losing the whole reply over it wouldn't be a reasonable tradeoff

Graph shape is now: `load_memory → generate_response → generate_note → update_memory → END`. `update_memory` is unaffected — `generate_note` doesn't touch `messages`, so its logic for finding "the last 2 messages = this turn" still holds.

### What changed for callers

`AgentResult` (from `/agent/chat-test` and `/audio/converse-test`) now includes a `note` field alongside `response_text`.

### Try it now

```bash
curl -X POST http://localhost:8000/agent/chat-test \
  -H "Content-Type: application/json" \
  -d '{"conversation_id": "note-test", "transcript": "Give me a quick rundown of the weekend weather", "history": []}'
```
Check the response for both `response_text` (the full spoken reply) and `note` (a short caption like "Summarizes the weekend forecast").

### Tests

```bash
pytest tests/test_text_processing.py tests/test_agent.py -v
```

- `test_text_processing.py` (22 tests, new) — every markdown construct sanitization handles, idempotency, whitespace collapsing, and the speech-duration estimate.
- `test_agent.py` (19 tests, up from 13) — new coverage for: the note field being populated, the note-generation call using the right prompt and receiving the response text (not the transcript) as input, markdown getting stripped from both `response` and `note`, and the fallback-to-truncation behavior when note generation fails.

79 tests pass across the whole project; `pyright` is clean.

### Step-by-step manual testing walkthrough

1. **Automated tests first** (no API key needed):
   ```bash
   pytest tests/test_text_processing.py -v
   ```
   Confirms the sanitization logic itself is correct before spending any real API calls on it.

2. **Trigger markdown from the model on purpose**, to see the sanitizer actually catch something (rather than just trusting the prompt worked):
   ```bash
   curl -X POST http://localhost:8000/agent/chat-test \
     -H "Content-Type: application/json" \
     -d '{"conversation_id": "sanitize-test", "transcript": "List 3 tips for better sleep, formatted with markdown bullet points", "history": []}'
   ```
   Even though this deliberately asks for markdown, `response_text` in the reply should come back with no `*`, `#`, or `-` list markers — plain spoken-style prose instead. This is the sanitizer doing its job regardless of what the model attempted.

3. **Check the note makes sense relative to the response** — for the same call above, confirm `note` is a short, accurate one-line summary (e.g. "Shares tips for better sleep") and not a copy of the full response or unrelated text.

4. **Test the note fallback** without needing to actually break anything: read `tests/test_agent.py`'s `TestNoteFallback` class — it simulates a note-generation failure and confirms the main response still comes through successfully with a truncated fallback note. Worth understanding even without re-running it live, since it's the behavior that'll matter if Groq ever has a transient hiccup specifically on the second (note) call of a turn.

5. **Multi-turn + note together**: reuse the same `conversation_id` across two calls (memory persists per Phase 5) and confirm each response comes back with its own distinct, relevant `note` — not a stale one from the previous turn.

## Phase 7 — Voice Generation

Converts Ava's text response into speech via ElevenLabs (`eleven_flash_v2_5`, per the diagram) — this is the first phase where you can actually *hear* Ava reply, completing the voice-in/voice-out loop.

### `app/services/tts.py`

- **`TTSService.synthesize(text)`** — validates, sanitizes, and sends text to ElevenLabs, returns raw audio bytes.
- **Sanitization as defense-in-depth**: runs `sanitize_for_speech` (Phase 6) again internally, so `TTSService` is safe to call directly with arbitrary text, not just text that already passed through the agent graph.
- **Validation**: rejects empty text outright, and rejects text over `tts_max_chars` (2000, configurable) — a safety ceiling, not a truncation. Ava's prompt already keeps replies short, so hitting this in practice would mean something's actually wrong upstream and is worth knowing about loudly rather than silently cutting off audio mid-sentence.
- **Retry/backoff**, same pattern as `TranscriptionService`: network errors (`httpx.ConnectError`/`TimeoutException`) and server-side issues (5xx, 429 rate limits) retry with exponential backoff; 4xx errors (bad API key, invalid `voice_id`) fail immediately since retrying won't fix a malformed request.
- **`TTSClient`** — a structural Protocol (same pattern as `TranscriptionClient`/`ConversationMemory`/`VectorMemory` elsewhere in this project), so tests inject a fake ElevenLabs client instead of hitting the real API.

### Two new endpoints

- **`POST /tts/synthesize-test`** — `{"text": "..."}` in, playable audio out. Isolated from the rest of the pipeline, for testing TTS on its own.
- **`POST /audio/voice-reply-test`** — the full loop: voice note in → transcribe → agent response + note → synthesize speech → **audio out**. First endpoint where you upload your voice and get Ava's voice back.

Since HTTP headers must be ASCII but the transcript/response/note can contain any Unicode character, `/audio/voice-reply-test` returns them base64-encoded in `X-Transcript-B64`, `X-Response-Text-B64`, and `X-Note-B64` response headers alongside the audio body.

### One-time setup

Get a free ElevenLabs API key and pick a voice ID at [elevenlabs.io](https://elevenlabs.io), then in `.env`:
```
ELEVENLABS_API_KEY=your_real_key_here
ELEVENLABS_VOICE_ID=your_chosen_voice_id
```

### Tests

```bash
pytest tests/test_tts.py -v
```

15 tests against a fake ElevenLabs client (`FakeElevenLabsClient`) — no network or real API key needed: successful synthesis, sanitization-before-send, empty/oversized text validation, retry on connection errors/timeouts/rate-limits/5xx, and immediate failure (no retry) on 4xx errors like a bad API key or invalid voice ID.

94 tests pass across the whole project; `pyright` is clean.

### Step-by-step manual testing walkthrough

1. **Automated tests first** (no API key needed):
   ```bash
   pytest tests/test_tts.py -v
   ```

2. **Isolated TTS test** — confirm ElevenLabs itself works before chaining anything onto it. In `/docs`, use `POST /tts/synthesize-test`:
   ```json
   {"text": "Hey, this is a test of the voice generation pipeline."}
   ```
   Swagger UI renders the response with a built-in audio player — click it and confirm you hear clear speech.

3. **Test the character limit guard** — send text over 2000 characters (e.g. paste a long paragraph repeated a few times) and confirm you get a clean `502` with a "too long" message, not a hang or a truncated/garbled audio file.

4. **Test with markdown on purpose** — since `TTSService` sanitizes independently:
   ```json
   {"text": "Here's **the plan**:\n- Wake up early\n- Go for a run"}
   ```
   Listen to the result — it should read naturally with no spoken-aloud asterisks or "dash" sounds, confirming the defense-in-depth sanitization works even calling TTS directly (bypassing the agent graph entirely).

5. **The full loop** — the actual milestone of this phase. Use `POST /audio/voice-reply-test`, upload a real voice recording asking Ava something. Play the returned audio directly in Swagger UI. Then decode the headers to confirm the pipeline's intermediate steps matched what you heard:
   ```bash
   curl -i -X POST http://localhost:8000/audio/voice-reply-test \
     -F "file=@your_voice_note.ogg" \
     -F "conversation_id=voice-loop-test" \
     --output ava_reply.mp3
   ```
   Then decode a header to check the transcript, e.g. in PowerShell:
   ```powershell
   [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String("PASTE_X-TRANSCRIPT-B64_VALUE_HERE"))
   ```
   Play `ava_reply.mp3` and confirm it's actually Ava's voice speaking the `response_text` you decoded.

6. **Multi-turn voice conversation** — call `/audio/voice-reply-test` twice with the same `conversation_id`, referencing something from the first recording in the second. Since Phase 5's memory persists server-side, Ava's second voice reply should correctly reference what you said in the first — the full pipeline (memory + response + note + speech) working together end to end.

7. **Error handling** — try `/audio/voice-reply-test` with a corrupted/empty audio file and confirm you get a clean `422` from the Phase 2 validation layer, never reaching TTS at all (no wasted ElevenLabs API call on input that was never going to work).

## Phase 8 — Audio Streaming

Reduces server-side latency by overlapping "waiting for the rest of Ava's response" with "already synthesizing speech for the part that's done" — instead of Phase 4-7's sequence (wait for the complete text response, THEN synthesize one audio clip), this streams the reply sentence by sentence and kicks off TTS for each sentence as soon as it's ready.

**Important context, stated upfront**: WhatsApp delivers voice notes as complete files, not a live stream — so this doesn't change what an end WhatsApp user experiences once Phase 9 sends the reply; they still get one complete voice note either way. What it *does* change is wall-clock time on the server, and it lays groundwork for any future consumer that can play audio progressively (e.g. a live web interface). The value here is measurable, not audible — streamed and non-streamed audio sound identical; the difference is in timing, which is why this phase's testing tools are about numbers, not listening.

### `app/services/sentence_chunker.py`

`SentenceChunker` buffers streamed text and yields complete sentence-groups as they're ready. Batches short sentences together (`min_chunk_chars`, default 20) before flushing — each flush becomes a separate TTS request, and TTS has fixed per-request latency overhead, so over-chunking on short sentences ("Sure. Yes. Got it.") can hurt more than it helps. Known limitation, documented in the code: sentence-boundary detection is a simple regex, not real NLP segmentation — it'll over-split on abbreviations or decimals. Acceptable here (one extra short TTS clip, not incorrect output); worth remembering if this module is ever reused somewhere that needs precise boundaries.

### `app/services/streaming.py`

- **`stream_response_sentences()`** — streams the LLM token-by-token (`llm.astream`), yielding each completed sentence with timing.
- **`stream_voice_reply()`** — the full fast path: loads memory, streams sentences, synthesizes TTS per-sentence (off the event loop via `asyncio.to_thread` so it doesn't block other requests), then persists the note and full turn to memory once streaming completes.
- **Design choice worth knowing**: this bypasses the LangGraph graph for response generation (LangGraph's node model returns one dict per execution — not a natural fit for progressively yielding partial results). But it directly *reuses* `graph.py`'s `_load_memory_node`/`_generate_note_node`/`_update_memory_node` functions rather than duplicating that logic, since they're plain async functions, not tied to LangGraph internals. Only response generation has a genuinely different implementation between the two paths.
- **Sanitization tradeoff**: each sentence is sanitized independently as it streams (vs. the whole response at once in the non-streaming path) — markdown spanning a sentence-chunk boundary won't be caught. Documented, not hidden.

### Three new endpoints, in increasing order of what they touch

1. **`POST /agent/stream-metrics-test`** — cheapest to test: streams a response and returns per-sentence timing as JSON. No TTS, no memory, no database — isolates exactly the LLM-streaming behavior Phase 8 is built on.
2. **`POST /agent/latency-comparison-test`** — runs the *same* transcript through both the old blocking path and the new streaming path, returns `blocking_total_ms` vs. `streaming_time_to_first_audio_ms` as a direct, repeatable number. Deliberately side-effect-free (no DB writes on either path) so it's safe to run repeatedly as a pure benchmark.
3. **`POST /audio/voice-reply-stream-test`** — true HTTP streaming: audio bytes start arriving as each sentence is synthesized, not after the whole reply is assembled. **Honest limitation**: since HTTP headers must be sent before the body and the full response/note text isn't known until streaming finishes, only the transcript goes in a header (`X-Transcript-B64`) — response text and note are logged server-side instead. Use `/audio/voice-reply-test` (Phase 7) instead if you need those programmatically.

### Tests

```bash
pytest tests/test_sentence_chunker.py tests/test_streaming.py -v
```

- `test_sentence_chunker.py` (15 tests) — sentence splitting, short-sentence batching, multi-character delta feeding, flush edge cases, content-preservation across arbitrary chunk boundaries.
- `test_streaming.py` (10 tests) — sentence timing ordering, per-sentence TTS call ordering, memory persistence after streaming completes, prior-history loading (confirming the graph-function reuse actually works), and correct behavior with no memory stores configured.

119 tests pass across the whole project; `pyright` is clean.

### Step-by-step manual testing walkthrough

Since streamed and non-streamed audio sound identical, **the numbers are the test** — start with the cheap JSON endpoints before touching audio at all.

1. **Automated tests first** (no API key needed):
   ```bash
   pytest tests/test_sentence_chunker.py tests/test_streaming.py -v
   ```

2. **See sentence-level timing directly** — this is the clearest way to *see* streaming happening:
   ```bash
   curl -X POST http://localhost:8000/agent/stream-metrics-test \
     -H "Content-Type: application/json" \
     -d '{"conversation_id": "stream-test", "transcript": "Tell me three interesting facts about space, one full sentence each."}'
   ```
   Look at the `elapsed_seconds` on each sentence in the response — they should increase gradually, showing sentences becoming available progressively rather than all at once.

3. **The actual latency comparison** — this is the number that matters:
   ```bash
   curl -X POST http://localhost:8000/agent/latency-comparison-test \
     -H "Content-Type: application/json" \
     -d '{"conversation_id": "latency-test", "transcript": "Give me a two or three sentence overview of how photosynthesis works."}'
   ```
   Check `streaming_time_to_first_audio_ms` vs `blocking_total_ms` — the streaming number should be meaningfully lower, since it only has to wait for the *first* sentence plus one TTS call, not the entire response plus one TTS call for everything. `speedup_to_first_audio` gives you that ratio directly. Try this a few times and with different-length responses (longer responses → bigger gap, since blocking waits for proportionally more text before doing anything).

4. **Confirm content correctness, not just speed** — for the same request as step 2, check that concatenating all the `sentences[].text` values in order reconstructs a coherent response with nothing dropped or duplicated.

5. **True streaming audio** — upload a real voice recording:
   ```bash
   curl -i -X POST http://localhost:8000/audio/voice-reply-stream-test \
     -F "file=@your_voice_note.ogg" \
     -F "conversation_id=stream-audio-test" \
     --output ava_streamed_reply.mp3
   ```
   Play the result — it should sound identical to Phase 7's `/audio/voice-reply-test` output (same content, different delivery timing). Check your uvicorn console log for the `stream_voice_reply | ... | complete` line to see the full response text and note that isn't available in the response headers.

6. **Watch it happen live** (optional, the most direct way to see the pipelining) — run `/audio/voice-reply-stream-test` while watching your uvicorn console: you should see `sentence ready at X.XXs` debug logs (set `LOG_LEVEL=DEBUG` in `.env` if you don't see them) appearing progressively, each immediately followed by a TTS call, rather than one long pause followed by a burst of activity.

7. **Multi-turn + streaming together** — call `/audio/voice-reply-stream-test` twice with the same `conversation_id`. Since it reuses `graph.py`'s real memory nodes, the second streamed reply should correctly reference the first, same as the non-streaming path.

## Phase 9 — WhatsApp Integration

This is the real product: a genuine WhatsApp voice note in, run through the entire pipeline built across Phases 2-8, replied to with a genuine WhatsApp voice note out. Everything before this phase was debug endpoints for testing pipeline stages in isolation; this phase is what an actual user's phone talks to.

### Architecture

- **`app/api/webhook.py`** — `POST /webhook/whatsapp`, the URL Twilio calls on every inbound WhatsApp message. Parses Twilio's real payload shape (form-encoded, not JSON), validates the request genuinely came from Twilio, then delegates to the handler.
- **`app/services/twilio_client.py`** — `validate_twilio_signature` (Twilio's HMAC-based webhook authentication — tested against the *real* algorithm via `RequestValidator.compute_signature`, not a hand-rolled fake) and `TwilioMediaDownloader` (fetches inbound voice notes from Twilio's protected media URLs using Basic Auth).
- **`app/services/whatsapp_handler.py`** — the actual orchestration: chains audio validation (Phase 2) → transcription (Phase 3) → agent with memory (Phases 4-6) → TTS (Phase 7) → TwiML reply. Every failure point has a graceful fallback reply — a WhatsApp user always gets *something* back, never silence. **Uses the WhatsApp sender's phone number directly as the `conversation_id`** — a natural, stable per-user identity, so Phase 5's memory just works per-contact with zero extra wiring.
- **`app/services/media_store.py`** + **`app/api/media.py`** — Twilio fetches outbound reply audio from a public URL (it can't receive raw bytes in a webhook response), so generated replies are saved to local disk and served back out at `{PUBLIC_BASE_URL}/media/{filename}`.
- **Why not Phase 8's streaming pipeline here**: Twilio delivers outbound WhatsApp media as one fetchable URL, not a live stream — there's no way to progressively deliver audio to a WhatsApp user, so the complete file has to exist before Twilio can fetch any of it. This phase uses the Phase 4-7 non-streaming path.

### Part 1: Twilio account and WhatsApp Sandbox setup

You don't need a paid account or business verification to test this — Twilio's WhatsApp Sandbox is free and works immediately.

1. **Create a Twilio account**: go to [twilio.com/try-twilio](https://www.twilio.com/try-twilio) and sign up (free trial, no credit card required for sandbox use).
2. **Find your credentials**: once logged in, your Twilio Console dashboard ([console.twilio.com](https://console.twilio.com)) shows your **Account SID** and **Auth Token** right on the homepage. Copy both into `.env`:
   ```
   TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   TWILIO_AUTH_TOKEN=your_auth_token_here
   ```
3. **Activate the WhatsApp Sandbox**: in the Console, go to **Messaging → Try it out → Send a WhatsApp message** (or search "WhatsApp Sandbox" in the Console search bar). You'll see a sandbox number (usually `+1 415 523 8886`) and a join code like `join <two-words>`.
4. **Join the sandbox from your own phone**: open WhatsApp, message that sandbox number, and send the exact join phrase shown (e.g. `join happy-tiger`). You'll get a confirmation reply. This links your personal WhatsApp number to the sandbox for testing — **note: sandbox sessions expire after 72 hours of inactivity**, so you may need to rejoin if you come back to this later.
5. Set your Twilio WhatsApp number in `.env`:
   ```
   TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
   ```

### Part 2: Expose your local server with ngrok

Twilio needs a public HTTPS URL to send webhooks to — your `localhost:8000` isn't reachable from the internet.

1. **Install ngrok**: [ngrok.com/download](https://ngrok.com/download) — or `winget install ngrok` on Windows, `brew install ngrok` on Mac.
2. **Sign up and connect your auth token** (free tier is fine): after signing up at [ngrok.com](https://ngrok.com), run:
   ```powershell
   ngrok config add-authtoken YOUR_TOKEN_HERE
   ```
3. **Start your FastAPI server first** (separate terminal, as always):
   ```powershell
   uvicorn app.main:app --reload
   ```
4. **Start the tunnel** (another separate terminal):
   ```powershell
   ngrok http 8000
   ```
   You'll see output with a **Forwarding** URL like `https://a1b2-c3d4-5678.ngrok-free.app`. This is your public URL.
5. **Set it in `.env`**:
   ```
   PUBLIC_BASE_URL=https://a1b2-c3d4-5678.ngrok-free.app
   ```
   **Important**: on ngrok's free tier, this URL **changes every time you restart ngrok**. You'll need to update `.env` and the Twilio Console webhook setting (next step) together each session — this is the single most common source of confusion when testing, so if things stop working after a break, check this first.

### Part 3: Point Twilio at your webhook

1. Back in the Twilio Console's WhatsApp Sandbox settings page, find **"When a message comes in"**.
2. Set it to your ngrok URL + `/webhook/whatsapp`, e.g.:
   ```
   https://a1b2-c3d4-5678.ngrok-free.app/webhook/whatsapp
   ```
3. Method: **POST**.
4. Save.

### Part 4: Finish `.env` setup

You should now have, at minimum:
```
GROQ_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886
PUBLIC_BASE_URL=https://your-current-ngrok-url.ngrok-free.app
```
Plus Postgres/ChromaDB running (`docker compose up -d postgres chromadb`) and `python -m app.db.bootstrap` already run, from Phase 5.

Restart uvicorn after editing `.env` so the new values are actually loaded.

### Tests

```bash
pytest tests/test_twilio_client.py tests/test_media_store.py tests/test_whatsapp_handler.py tests/test_webhook.py -v
```

- **`test_twilio_client.py`** (11 tests) — signature validation tested against Twilio's **real** signing algorithm (`RequestValidator.compute_signature` — the same code Twilio itself runs, not a fake), so these would actually catch a regression; media download tested against the real `TwilioMediaDownloader` class via `httpx.MockTransport` (a fake network transport, not a fake class).
- **`test_media_store.py`** (15 tests) — save/load roundtrips, and **real path-traversal protection tests** (`../../etc/passwd`-style attempts), since the filename in `GET /media/{filename}` comes straight from the URL.
- **`test_whatsapp_handler.py`** (8 tests) — the core business logic, run with **real** audio processing and a **real** compiled agent graph, fakes only at the true external boundaries (Twilio media download, the LLM, TTS). Covers: no-media/non-audio messages get instructions, download/audio-validation/agent/TTS failures all produce graceful fallback replies instead of crashing, a successful flow produces correct TwiML, and — importantly — confirms the WhatsApp phone number is correctly used as the memory `conversation_id`.
- **`test_webhook.py`** (9 tests) — HTTP-layer tests: missing/wrong/tampered signatures all correctly return `403`, missing API keys return clean `500`s, and the media route correctly serves files (or `404`s cleanly, including for path-traversal attempts).

One real bug this testing surfaced and fixed: `TTSService` didn't have a catch-all for unexpected exception types — only specific `httpx` errors and `ApiError` were wrapped as `TTSError`; anything else would propagate raw instead of being caught by `whatsapp_handler.py`'s `except TTSError` fallback. Fixed with a defensive catch-all that wraps any unanticipated exception as `TTSError`, so the graceful-fallback guarantee actually holds regardless of what the underlying client throws.

162 tests pass across the whole project; `pyright` is clean.

### Step-by-step manual testing walkthrough

1. **Automated tests first**:
   ```bash
   pytest tests/test_twilio_client.py tests/test_media_store.py tests/test_whatsapp_handler.py tests/test_webhook.py -v
   ```

2. **The real thing** — from the WhatsApp number you joined the sandbox with, send a voice note to your Twilio sandbox number asking Ava something. Watch your uvicorn console — you should see the `whatsapp | from=... | transcript=...` log line once processing completes, then get a reply in WhatsApp within a few seconds: a short text caption (Ava's note) with a playable voice note attached.

3. **Text-only message** — send a plain text WhatsApp message (no voice note) to the sandbox number. You should get back the "I reply to voice notes" instructions message, not an error or silence.

4. **Send a photo or other non-audio media** — same instructions-reply behavior should trigger; confirms the `media_content_type.startswith("audio/")` check works on real Twilio payloads, not just the format your own testing tools happened to send.

5. **Multi-turn memory over real WhatsApp** — send a voice note mentioning something (e.g. your dog's name), then a follow-up voice note asking about it. Since the WhatsApp number is the `conversation_id`, Ava should correctly remember across separate messages — the same Phase 5 memory you tested via curl, now working through the real product.

6. **Verify data landed in Postgres** — open pgAdmin, query the `messages` table filtered to your WhatsApp number (Twilio's `From` format, e.g. `whatsapp:+15551234567`):
   ```sql
   SELECT role, content, created_at FROM messages
   WHERE conversation_id = 'whatsapp:+15551234567'
   ORDER BY created_at ASC;
   ```

7. **Check generated audio files exist**: look in your project's `media_storage/` folder — you should see `.mp3` files accumulating, one per reply sent. (This is local disk for now — Phase 10 is where this would move to real cloud storage for an actual deployment.)

8. **Test signature validation is actually protecting the endpoint** — try calling your own webhook directly without a valid Twilio signature:
   ```bash
   curl -X POST https://your-ngrok-url.ngrok-free.app/webhook/whatsapp \
     -d "From=whatsapp:+15551234567&NumMedia=0"
   ```
   Should get a clean `403`, confirming random internet traffic can't trigger your pipeline (which would otherwise burn real Groq/ElevenLabs API credits) just by finding your ngrok URL.

9. **Test the ngrok-URL-changed failure mode on purpose** — restart ngrok (getting a new URL) without updating `.env` or the Twilio Console, then send a WhatsApp message. Twilio will get a connection failure (your old tunnel is gone) — this is the single most common real gotcha with sandbox testing, worth seeing once so you recognize it immediately if it happens later.

### Known limitations, stated honestly (as of Phase 9)

- **Sandbox limitations**: Twilio's WhatsApp Sandbox works great for testing but isn't meant for real users — only numbers that've sent the join code can message you, sessions expire after 72 hours, and it caps you at a handful of messages per day. Moving to a real Twilio WhatsApp Sender (business-verified) is a Phase 10 concern — see below.
- **Synchronous processing**: the webhook waits for the entire pipeline before responding to Twilio, risking its webhook timeout. **Fixed in Phase 10** — see below.
- **Local media storage**: reply audio lived on local disk with manual cleanup, not real cloud storage. **Fixed in Phase 10** — see below.

## Phase 10 — Deployment

Takes this from "runs on my machine with ngrok" to an actually deployed service. Two kinds of work: production-hardening the code itself, and the actual deployment walkthrough.

### What changed in the code

**1. Async webhook processing** (`app/api/webhook.py`, `app/services/whatsapp_handler.py`) — directly motivated by real numbers observed in Phase 9 testing: full pipeline processing took up to ~13 seconds, uncomfortably close to Twilio's webhook timeout. The webhook now acknowledges Twilio with an empty TwiML response almost instantly, and processes the voice note as a FastAPI `BackgroundTask` — the real reply gets sent afterward via Twilio's REST API (`TwilioMessageSender`, using `client.messages.create(...)`) rather than the original synchronous TwiML response. Text-only "no voice note attached" replies stay synchronous, since there's no meaningful processing time there.

**2. Cloud media storage** (`app/services/media_store.py`) — this mattered more than a "nice to have": most deployment platforms run containers with **ephemeral local disks**. A file saved locally by one instance can vanish on redeploy, restart, or simply not exist on a different instance if you ever scale beyond one. `MediaStorage` is now a Protocol with two implementations — `LocalDiskMediaStorage` (unchanged, still the default for local dev) and `S3MediaStorage` (new — works with real AWS S3 or any S3-compatible provider: Cloudflare R2, Backblaze B2, MinIO). Switch via `MEDIA_STORAGE_BACKEND=s3` in `.env`.

**3. Debug endpoint protection** (`app/api/debug_auth.py`) — `/audio/*`, `/agent/*`, `/tts/*` each spend real Groq/ElevenLabs credits per call. Fine wide open on your own machine; a real risk on a public deployment, where anyone who finds the URL could run up your bill. When `ENVIRONMENT=production`, these now require an exact `X-Debug-Token` header match against `DEBUG_API_TOKEN` — and refuse access entirely (not weakly) if that token isn't set, rather than the dangerous failure mode of "empty token means anything works." The webhook and `/metrics` routes are deliberately left public, since Twilio has no way to send a custom header and metrics scraping tools expect open access.

**4. Basic observability** (`app/core/metrics.py`, `GET /metrics`) — `prometheus-client` had sat unused in `requirements.txt` since Phase 1. Now tracks WhatsApp message outcomes, per-pipeline-stage errors (so a spike in TTS failures, say, is visible rather than buried in logs), and end-to-end reply latency — the exact metric that matters given why this phase restructured the webhook in the first place.

**5. Alembic migrations** (`migrations/`, `alembic.ini`) — replaces relying solely on `python -m app.db.bootstrap`'s `CREATE TABLE IF NOT EXISTS` approach for anything beyond local quick-start. Real schema versioning: `alembic upgrade head` / `alembic downgrade base`, with an initial migration matching Phase 5's `MessageRecord` exactly (verified by actually running both directions against a test database, not just written and assumed correct). Notably, `migrations/env.py` pulls its connection string from the app's own `Settings.database_url` rather than a second hardcoded URL in `alembic.ini` — Phase 5 hit a real bug from exactly that kind of duplication (`DATABASE_URL` silently drifting out of sync with `POSTGRES_PASSWORD`), so this migration setup was built specifically to not repeat it.

**6. Production Dockerfile** (`docker/Dockerfile.prod`) — separate from the existing dev `docker/Dockerfile` (which stays as-is for local `docker-compose` use). Multi-stage build (build tools like `build-essential` don't ship in the final ~200MB-smaller image), runs as a non-root user, includes a `HEALTHCHECK` matching `/health`, and bundles `migrations/`/`alembic.ini` so migrations can run against the deployed container's actual code version.

### Tests

```bash
pytest tests/test_debug_auth.py tests/test_media_store.py tests/test_webhook.py -v
```

- `test_debug_auth.py` (9 tests, new) — dev mode unaffected, production correctly blocks/allows based on token, confirms the guard applies to exactly the right three routers and not the webhook/metrics/health routes.
- `test_media_store.py` (26 tests, up from 15) — new `S3MediaStorage` coverage using a fake boto3-shaped client (save/load roundtrip, correct bucket/content-type sent, both real-AWS and S3-compatible-endpoint URL formats, clear errors when unconfigured), plus a test confirming the free-function API actually dispatches to S3 when configured, not just that the class works in isolation.
- `test_webhook.py` (12 tests) — includes the async background-task wiring and the TwiML content-type regression test from earlier troubleshooting in this conversation.

192 tests pass across the whole project; `pyright` is clean.

**One thing I can't verify from this environment**: `docker/Dockerfile.prod` itself — building and running a real container isn't possible in the sandbox this project was built in. Test it locally before deploying:
```bash
docker build -f docker/Dockerfile.prod -t voice-ai-assistant:prod .
docker run -p 8000:8000 --env-file .env voice-ai-assistant:prod
curl http://localhost:8000/health
```

### Step-by-step deployment walkthrough (Render)

Render was chosen for this walkthrough because it has native managed Postgres, straightforward Docker-based web services, and covers Web Services + free Postgres **with no credit card required at all**. This walkthrough deliberately skips ChromaDB — Render's free tier doesn't include running a custom Docker image as a Private Service (that requires a paid plan), so we set `ENABLE_SEMANTIC_MEMORY=false` and the app runs on Postgres-based short-term memory alone (see `app/core/config.py`'s docstring on that setting). **Net result: this entire deployment needs no billing information anywhere in the stack** — relevant if you've had to remove a card and are waiting out a re-add restriction, as came up while building this walkthrough.

**One important clarification, since it's an easy mix-up**: ChromaDB and the S3-style storage below are unrelated systems. ChromaDB is semantic/long-term memory (skipped entirely here). The storage below is *only* for hosting generated voice-reply audio files so Twilio can fetch them — there is no way to "migrate ChromaDB to S3 storage," since ChromaDB doesn't work that way; we're simply not using ChromaDB at all in this deployment.

#### Part 1: Backblaze B2 setup (free, no credit card)

1. Sign up at [backblaze.com/sign-up/b2-cloud-storage-backup-archive](https://www.backblaze.com/sign-up/b2-cloud-storage-backup-archive) — confirmed on Backblaze's own signup page: no card required
2. Verify your email, log in to the B2 dashboard
3. **Create a Bucket** — name it something like `voice-ai-media` (must be globally unique), set **Files in Bucket** to **Public** (Twilio needs to fetch files without credentials)
4. Click into the bucket, note the **Endpoint** shown (e.g. `s3.us-west-004.backblazeb2.com`) — you'll need this with `https://` in front
5. Go to **App Keys** → **Add a New Application Key** → name it, restrict access to your specific bucket, type **Read and Write** → **Create New Key**
6. **Copy both values immediately** (the application key is shown only once): `keyID` and `applicationKey`

#### Part 2: Push your code to GitHub

```bash
git init
git add .
git commit -m "Voice AI Assistant — all 10 phases"
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

#### Part 3: Create the Postgres database

Render Dashboard → **New → PostgreSQL** → name it (e.g. `voice-ai-assistant-db`) → plan **Free** → create, and wait for it to finish provisioning. Keep its **Info** tab open — you'll need those connection details next.

#### Part 4: Create the web service

Render Dashboard → **New → Web Service** → connect your GitHub repo:
- **Runtime**: Docker
- **Dockerfile Path**: `docker/Dockerfile.prod`
- **Docker Context**: `.`
- **Plan**: Free
- **Health Check Path**: `/health`

Create — the first deploy will likely fail health checks until you finish the steps below. Expected; keep going.

#### Part 5: Set every environment variable

Web service → **Environment** tab:

```
# Database — copy exact values from your Postgres's Info tab
POSTGRES_USER=<from Postgres Info tab>
POSTGRES_PASSWORD=<from Postgres Info tab>
POSTGRES_HOST=<from Postgres Info tab — use the Internal host if shown>
POSTGRES_PORT=<from Postgres Info tab>
POSTGRES_DB=<from Postgres Info tab>

# Skip ChromaDB entirely — no service to deploy, no cost
ENABLE_SEMANTIC_MEMORY=false

# Backblaze B2 — from Part 1
MEDIA_STORAGE_BACKEND=s3
S3_BUCKET_NAME=voice-ai-media
S3_REGION=us-west-004
S3_ENDPOINT_URL=https://s3.us-west-004.backblazeb2.com
AWS_ACCESS_KEY_ID=<your keyID>
AWS_SECRET_ACCESS_KEY=<your applicationKey>

# Your API keys
GROQ_API_KEY=<your real key>
ELEVENLABS_API_KEY=<your real key>
ELEVENLABS_VOICE_ID=<your chosen voice id>
TWILIO_ACCOUNT_SID=<your account sid>
TWILIO_AUTH_TOKEN=<your auth token>
TWILIO_WHATSAPP_NUMBER=whatsapp:+1XXXXXXXXXX

# Environment + security
ENVIRONMENT=production
DEBUG_API_TOKEN=<run `openssl rand -hex 32` locally and paste the result>
```

Save — triggers a redeploy.

#### Part 6: Get your Render URL, then set `PUBLIC_BASE_URL`

Once deployed, Render shows your URL at the top of the dashboard (e.g. `https://voice-ai-assistant-xyz.onrender.com`). Add it as one more variable:
```
PUBLIC_BASE_URL=https://voice-ai-assistant-xyz.onrender.com
```
Save (one more redeploy — last one needed).

#### Part 7: Run migrations against the production database

Web service → **Shell** tab (a terminal inside the running container):
```bash
alembic upgrade head
alembic current
```
Should show revision `a447c3190db3`.

#### Part 8: Point Twilio at your real deployment

This is also your moment to move beyond the Sandbox's 5-message/day limit, if you haven't already:

- **Still testing** (Sandbox is fine): Twilio Console → **Messaging → Try it out → Send a WhatsApp message** → update the Sandbox's webhook URL to `https://your-render-url.onrender.com/webhook/whatsapp`
- **Ready for real users**: apply for a proper WhatsApp Sender under **Messaging → Senders → WhatsApp senders** — requires Meta Business verification (a few business days for approval) but removes the sandbox's join-code and daily-limit restrictions entirely.

#### Part 9: Verify end to end

```bash
curl https://your-render-url.onrender.com/health
curl https://your-render-url.onrender.com/metrics
```
Then send a real voice note on WhatsApp and confirm you get Ava's voice reply back.

#### Part 10: Confirm debug endpoints are actually locked down

```bash
curl -X POST https://your-render-url.onrender.com/tts/synthesize-test -d '{"text":"test"}'
# Expect: 404

curl -X POST https://your-render-url.onrender.com/tts/synthesize-test \
  -H "X-Debug-Token: your_actual_debug_api_token" -d '{"text":"test"}'
# Expect: past the guard (200, or a different error if ElevenLabs itself has an issue)
```

#### Faster next time: the Blueprint

`render.yaml` in this repo does Parts 3-5's Postgres/web-service creation and most environment variable wiring in one step (**New → Blueprint**, connect your repo, click Apply) — worth using for a second deployment or a fresh environment, now that you understand what it's doing under the hood from doing it manually once.
```

### What's still a known limitation, honestly

Even after Phase 10, a couple of things are deliberately out of scope for this project's stated purpose (learning, hands-on — not a system engineered for scale):

- **`asyncio.to_thread` isn't used for the blocking I/O calls** (transcription, TTS, S3, Twilio's REST client) inside the async background task — consistent with how these were already written in earlier phases, but a genuinely high-concurrency deployment would want these wrapped to avoid blocking the event loop during each call. Noted in `media_store.py`'s docstring, not silently ignored.
- **FastAPI's `BackgroundTasks`, not a real task queue** — fine at this project's scale; a production deployment expecting to process many simultaneous WhatsApp conversations would want Celery, RQ, or arq instead, which survive a server restart mid-task and can be scaled independently of the web process.
- **S3 cleanup relies on your bucket's lifecycle rules**, not application code — deliberate (see `S3MediaStorage.cleanup_expired`'s docstring): a cloud provider's built-in expiry is more reliable than an app-level sweep that only runs while the app happens to be running.

---

That's all ten phases: a WhatsApp voice assistant that listens, remembers, thinks, and talks back — built and tested step by step, from a project-planning doc and an architecture diagram to something you can actually message on your phone.
