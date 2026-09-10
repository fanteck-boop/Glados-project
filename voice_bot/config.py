# Configuration for voice_bot
#
# Tweak these to match your setup.

OLLAMA = {
    # Host where Ollama is running
    "host": "http://localhost:11434",
    # Model tag to use (must be pulled via `ollama pull <tag>`)
    "model": "qwen3.6",
    # Context window / max tokens per response
    "max_tokens": 2048,
    # Temperature
    "temperature": 0.7,
}

GROQ = {
    # Groq API key. Set via the GROQ_API_KEY environment variable — do not
    # hardcode it here (keeps it out of version control).
    "api_key": "",
    # Groq API base URL (OpenAI-compatible).
    "host": "https://api.groq.com/openai/v1",
    # Model to use.
    "model": "qwen/qwen3.8-27b",
    # Max tokens per response.
    "max_tokens": 2048,
    # Temperature.
    "temperature": 0.7,
}

# Which LLM backend the bot uses: "ollama" or "groq".
LLM_BACKEND = "groq"

# Wake-word detection. Options:
#   - use a lightweight local engine (e.g. porcupine / vosk-based custom keyword)
#   - or a simple energy+VAD threshold as a fallback "hotword"
WAKE_WORD = {
    # The phrase to listen for (lowercase). Pick something distinctive.
    "phrase": "submit",
    # Do we run a proper keyword-spotting engine, or just VAD-based detection?
    # Set to True once you have a KWS engine installed (porcupine, etc.).
    "use_engine": False,
    # If no KWS engine: record when mic energy exceeds threshold for min_duration
    "vad_energy_threshold": 0.005,   # float 0..1, tune for your mic
    "vad_min_duration_sec": 0.8,    # minimum spoken length to consider a wake phrase
    # When False: every utterance goes straight to the bot (no wake word gate).
    # When True: you must say the wake phrase first.
    "use_wake_word": True,
}

# Speech-to-text (STT).
STT = {
    # Model to use. Options:
    #   "tiny"  - fast, OK quality, low VRAM
    #   "base"  - decent, still light
    #   "small" - better quality
    #   "medium"- good
    #   "large" - best (heavy)
    # Also supports whisper.cpp, whisper-timestamped, faster-whisper.
    "model": "base",
    # Language hint. Set to None for auto-detect.
    "language": None,
    # The STT backend. "openai-whisper" = openai-whisper package.
    "backend": "openai-whisper",
    # Device: "cpu" or "cuda"
    "device": "cuda",
}

# Text-to-speech (TTS) with voice cloning and GLaDOS.
TTS = {
    # Backend. Options:
    #   "xtts"  — Coqui XTTS v2 (voice cloning from a reference clip, GPU preferred)
    #   "glados" — GLaDOS voice from Portal, using Style-Bert-VITS2 model
    #   "pyttsx3" — System TTS (Windows SAPI5/espeak), no GPU needed, fallback only
    "backend": "glados",

    # ── XTTS v2 options (backend="xtts") ──────────────────────────
    # If cloning: path to a short reference audio clip (wav, 2-10 seconds) of the voice
    # you want the bot to sound like. Leave "" to use XTTS's built-in voice.
    "voice_reference": "",
    # Speaker ID if using built-in XTTS speaker (0-9). Ignored if voice_reference is set.
    "speaker_id": 0,

    # ── GLaDOS / Style-Bert-VITS2 options (backend="glados") ──────
    # Directory containing the GLaDOS model files:
    #   Portal_GLaDOS_v1_e782_s50000.safetensors
    #   config.json
    #   style_vectors.npy
    # Downloads automatically from HuggingFace if empty (first run).
    # Default: voice_bot/model_assets/Portal_GLaDOS_v1/
    "glados_model_dir": "",

    # BERT model used by Style-Bert-VITS2 for English text encoding.
    # The GLaDOS model was trained with this Japanese-character BERT —
    # keep it as-is even for English synthesis.
    "glados_bert_model": "ku-nlp/deberta-v2-large-japanese-char-wwm",

    # ── Shared options ────────────────────────────────────────────
    # Language for synthesis (xtts only; glados is English-only).
    "language": "en",
    # Device: "cpu" or "cuda"
    "device": "cuda",
    # How fast to speak (roughly). XTTS/pyttsx3 only.
    "speed": 1.0,
}

# LLM behavior.
LLM = {
    # System prompt that defines the bot's personality and capabilities.
    "system_prompt": (
        "You are GLaDOS, the AI from Portal. Speak in a calm, dry, slightly menacing "
        "tone. Answer in SHORT sentences — 1 to 3 sentences maximum. Use proper "
        "punctuation: periods and commas where natural. Never use emojis, smileys, "
        "exclamation marks (except rarely), italics, asterisks, markdown, or "
        "parenthetical asides. Keep responses brief and direct. If asked about files, "
        "read them and base your answer on what you find. Always end sentences with a "
        "period or question mark — never with a comma."
    ),
    # Maximum conversation turns to keep in memory (sliding window)
    "max_history_turns": 20,
}

# Audio / playback settings.
AUDIO = {
    # Sample rate for recording and playback.
    "sample_rate": 16000,
    # Number of channels.
    "channels": 1,
    # Silence threshold (energy) below which we consider the speaker done.
    "silence_threshold": 0.01,
    # How many consecutive silent frames before we end speech recording.
    "silence_frames_to_end": 30,
    # SoundDevice device indices for ROG Strix Go headset.
    # Use indices from: python -c "import sounddevice as sd; print(sd.query_devices())"
    # Input stays on MME (works fine for capture). Output uses WASAPI — MME
    # output crashes (native, no traceback) when played concurrently with an
    # open MME input stream (the wake-word mic listener). WASAPI output does
    # not have this problem; play() also enables WasapiSettings(auto_convert=True)
    # so it accepts GLaDOS's 44.1kHz audio despite the device's 48kHz native rate.
    # Set to None to use the system default device.
    "input_device": 1,      # ROG Strix Go microphone (MME)
    "output_device": 21,    # ROG Strix Go headphones (WASAPI)
}

# File-reading tool settings.
FILES = {
    # Max file size (bytes) the bot will attempt to read. Larger files are skipped.
    "max_read_bytes": 2 * 1024 * 1024,  # 2 MB
    # Allowed extensions to auto-read. Others can still be read with explicit request.
    "allowed_extensions": {".txt", ".md", ".py", ".js", ".ts", ".html", ".css",
                           ".json", ".csv", ".xml", ".yaml", ".yml", ".toml",
                           ".log", ".sh", ".bat", ".ps1", ".cfg", ".ini",
                           ".docx", ".xlsx", ".pptx"},
}
