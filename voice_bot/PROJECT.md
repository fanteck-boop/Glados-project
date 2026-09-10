# Voice Clone Bot — References

## XTTS v2 Voice Cloning

### Reference audio spec

- **Length**: 6–10 seconds of clean speech.
- **Format**: mono WAV, 24 kHz (XTTS default output), or 16 kHz.
- **Content**: The reference should be the same speaker, same language as the bot's output.
- **No backgroud noise / music / reverb** — cleanest possible.

### Speaker ID vs. voice reference

- **voice_reference** set → cloning from that clip. `speaker_id` is ignored.
- **voice_reference** empty, **speaker_id** set → uses one of XTTS's built-in speakers (0–9).
- Both empty → uses default speaker (usually speaker 0).

### First-run model download

- `TTS` (Coqui) downloads `tts_models/multilingual/multi-dataset/xtts_v2` on first load (~1.5 GB).
- Cache dir: `~/.local/share/coqui/tts/` on Linux, or the Coqui cache dir on Windows.
- On GPU, load with `gpu=True`. On CPU, `gpu=False` — slower but works.

### Performance notes

- GPU: ~0.5–2 s for a short sentence.
- CPU: ~10–60 s for a short sentence (depends on length and CPU).
- If CPU is too slow, use pyttsx3 as the backend instead (no GPU needed, no cloning).

## Wake Word Detection

### Energy-based VAD (MVP)

The wake_word module uses a simple energy threshold + silence-duration heuristic:

- **Energy threshold** (`WAKE_WORD.vad_energy_threshold`): RMS energy in [0, 1] that counts as "speech present". Tune per mic — most mics range 0.005–0.05.
- **Min duration** (`WAKE_WORD.vad_min_duration_sec`): Utterances shorter than this are discarded (prevents clicks/noise from being treated as speech).
- **Silence gap**: When energy drops below threshold for ~0.6–0.8 s, the utterance is considered complete and transcribed.

### Whisper as wake-phrase checker

Every completed utterance is transcribed by Whisper (tiny/base model). The wake phrase is matched against the normalized transcript (lowercase, punctuation removed, spaces collapsed). This catches the phrase regardless of exact timing as long as it's within a single utterance.

### Limitations of the MVP approach

- Every utterance is transcribed — not just wake-word candidates. Whisper on tiny/base is OK on CPU but adds latency.
- Cannot detect wake word mid-utterance (requires the phrase to be a complete utterance or at least the first part).
- Energy threshold must be tuned per environment / mic.

### Better alternatives (future iteration)

- **Porcupine** (picovoice): dedicated wake-word engine, very low latency, but requires an access key (free tier available).
- **Silero VAD** + custom keyword spotting: combine a good VAD with a small KWS model.
- **OpenWakeWord**: open-source wake-word models (e.g. "Hey Jarvis", "OK Nuance").

## Whisper STT

### Model size vs. quality vs. speed

| Model | VRAM/RAM | CPU speed | Quality |
|-------|----------|-----------|---------|
| tiny  | ~39 MB   | very fast | coarse |
| base  | ~142 MB  | fast      | decent |
| small | ~488 MB  | moderate  | good |
| medium| ~1.5 GB  | slow      | very good |
| large | ~2.9 GB  | very slow | best |

For a voice bot that transcribes every command utterance, "base" or "small" is the sweet spot on CPU. "tiny" is fast but may miss words. "large" is overkill unless the bot needs very high transcription accuracy.

### FP16 on CPU

Whisper runs in FP16 by default, which fails on some CPU setups. Force `fp16=False` when loading on CPU (the stt.py module does this).

### Language

- Set `language` explicitly if you know the bot will only hear one language (faster + more accurate).
- Set `language=None` for auto-detect if multilingual.

## Ollama LLM

### Local API

- Endpoint: `http://localhost:11434`
- Chat: `POST /api/chat` with `{"model": "...", "messages": [...], "stream": false}`
- Tags: `GET /api/tags` to list pulled models.
- Pull a model: `ollama pull <tag>` — on this system, Qwen 3.6 is already pulled (`qwen3.6:latest`).

### Streaming vs. non-streaming

- `stream=false` returns the full response in one HTTP response. Simpler for a voice bot.
- `stream=true` returns SSE lines — useful for long responses to start playback early, but adds complexity.

### Timeouts

- Ollama can take a while on large models. Set generous timeout (120 s recommended for the chat endpoint in llm.py).
- Handle `ConnectionError` (Ollama not running) and `Timeout` gracefully — the bot should say "can't reach LLM" rather than crash.

## File Reading Tool

### Supported formats (file_tool.py)

| Extension | Reader | Notes |
|-----------|--------|-------|
| `.txt, .md` | plain text | encoding fallback latin-1 |
| `.py, .js, .ts, .html, .css, ...` | plain text | all code files |
| `.json` | JSON pretty-print | falls back to raw on parse error |
| `.csv` | CSV reader | tab-separated output |
| `.xml` | plain text | |
| `.yaml, .yml` | YAML→JSON | requires PyYAML |
| `.toml` | TOML→JSON | requires tomli |
| `.docx` | python-docx | paragraphs + tables |
| `.xlsx` | openpyxl | all sheets, tab-separated |
| `.pptx` | python-pptx | slides + text frames + tables |
| `.pdf` | pymupdf (fitz) | text layer only |
| `.ini, .cfg` | plain text | |
| `.sh, .bat, .ps1` | plain text | |

### Size limit

- `FILES.max_read_bytes` (default 2 MB) — files larger than this return an error string rather than being read. Prevents huge files from blowing up LLM context.
- Very long files are truncated to the first `max_read_bytes` chars.

### Path extraction heuristic

`extract_paths(text)` looks for:
- Absolute Windows paths (`C:\...`)
- Absolute Unix paths (`/...`)
- Relative paths with a dot-extension (`file.json`, `index.html`)

Returns only paths that actually exist on disk.

## TTS playback

### flow

1. Synthesize text → WAV file (XTTS or pyttsx3)
2. Play WAV via `sounddevice` (blocking `sd.play` + `sd.wait`)
3. Next utterance queued after playback finishes

### pyttsx3 limitations

- `save_to_file` is backend-dependent — may not work on all systems.
- Fallback in tts.py: if save_to_file fails, produces a 0.5-second silent WAV so the bot doesn't crash.

### XTTS speed on CPU

- If XTTS on CPU is too slow for a conversational bot, switch `TTS.backend` to "pyttsx3" in config. No cloning, but instant synthesis.
- Mid-point: use `faster-whisper` for STT (faster) + XTTS only for the response (acceptable latency on GPU).

## System prompt

The LLM system prompt (config.LLM.system_prompt) defines the bot's personality and capabilities. For a Hermes-like file-reading bot, include:

- Name ("Hermes")
- Capabilities (read files, explain, summarize, answer questions)
- Tone (concise, friendly, conversational)
- How to handle file content (base answers on what you find)

Keep it concise — long system prompts eat context window.
