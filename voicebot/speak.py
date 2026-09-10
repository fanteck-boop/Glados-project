#!/usr/bin/env python3
"""
speak.py — GLaDOS voice playback for ROG Strix Go headphones.

Usage:
    python speak.py "Testing. This is GLaDOS speaking."

Loads the GLaDOS Style-Bert-VITS2 model, synthesizes the given text
with natural pauses at punctuation, and plays it through the ROG
Strix Go headphones (audio device index 4).

Exits with code 0 after playback completes. No background listening,
no wake words, no queue — just text in, audio out.

Model resolution (in order):
  1. VOICEBOT_MODEL_DIR env var
  2. voicebot/model_assets/Portal_GLaDOS_v1/
  3. ../voice_bot/model_assets/Portal_GLaDOS_v1/  (old project layout fallback)

NOTE: GLaDOS/Style-Bert-VITS2 returns audio in int16 scale (approx ±32767)
as float32. Every output path (WAV write + playback) normalizes by peak
before use, so the final audio is full-volume and audible.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths & logging
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
TTS_CACHE_DIR = PROJECT_DIR / "cache"
TTS_CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("speak")


# ---------------------------------------------------------------------------
# GLaDOS model directory resolution
# ---------------------------------------------------------------------------

def resolve_model_dir() -> Path:
    """Find the GLaDOS model directory.

    Checks (in order):
      1. VOICEBOT_MODEL_DIR environment variable
      2. voicebot/model_assets/Portal_GLaDOS_v1/
      3. ../voice_bot/model_assets/Portal_GLaDOS_v1/
    Exits with a clear error if none contain the model file.
    """
    marker = "Portal_GLaDOS_v1_e782_s50000.safetensors"

    # 1. Env var
    env = os.environ.get("VOICEBOT_MODEL_DIR", "")
    if env:
        p = Path(env).resolve()
        if (p / marker).exists():
            log.info("Model dir from VOICEBOT_MODEL_DIR: %s", p)
            return p

    # 2. Local model_assets/
    local = (PROJECT_DIR / "model_assets" / "Portal_GLaDOS_v1").resolve()
    if (local / marker).exists():
        log.info("Model dir from local layout: %s", local)
        return local

    # 3. Fallback: old voice_bot project
    fallback = PROJECT_DIR.parent / "voice_bot" / "model_assets" / "Portal_GLaDOS_v1"
    if (fallback / marker).exists():
        log.info("Model dir from fallback layout: %s", fallback)
        return fallback

    # Nothing found — exit with the most likely candidate shown
    candidates = [local, fallback]
    msg = "GLaDOS model not found.\n"
    msg += "Searched:\n"
    for c in candidates:
        msg += f"  {c}\n"
    msg += (
        "Download from HuggingFace: WarriorMama777/GLaDOS_TTS\n"
        "Or set VOICEBOT_MODEL_DIR to the folder containing the model files."
    )
    log.error(msg)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Punctuation -> pause mapping (seconds)
# ---------------------------------------------------------------------------

PAUSE_MAP = {
    ",": 0.20,
    ";": 0.30,
    ":": 0.30,
    ".": 0.55,
    "?": 0.55,
    "!": 0.55,
}
PARAGRAPH_PAUSE = 0.90
MAX_CHUNK_WORDS = 16


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def split_text(text: str) -> list[tuple[str, float]]:
    """Normalize text and split into chunks with pause durations.

    Returns list of (chunk_text, pause_after_seconds).
    """
    # Ensure punctuation followed by space, collapse whitespace
    text = re.sub(r"([,;:.!?])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split on sentence endings first
    segments: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+", text):
        part = part.strip()
        if part:
            segments.append(part)

    chunks: list[tuple[str, float]] = []
    for seg in segments:
        words = seg.split()
        for i in range(0, len(words), MAX_CHUNK_WORDS):
            chunk_words = words[i:i + MAX_CHUNK_WORDS]
            chunk = " ".join(chunk_words)
            last_char = chunk[-1] if chunk else ""
            pause = (
                PARAGRAPH_PAUSE
                if last_char in ".?!"
                else PAUSE_MAP.get(last_char, 0.0)
            )
            chunks.append((chunk, pause))

    return chunks


# ---------------------------------------------------------------------------
# WAV I/O — normalize to full range
# ---------------------------------------------------------------------------

def write_wav(path: Path, sr: int, audio: np.ndarray) -> None:
    """Write float32 audio to a 16-bit WAV file.

    Normalizes the audio to full range first, because GLaDOS/Style-Bert-VITS2
    returns audio in int16 scale (approx ±32767) as float32, not [-1, 1].
    """
    import wave

    # Normalize: divide by peak so audio fills the full range
    peak = float(np.max(np.abs(audio)))
    if peak > 0.001:
        audio = audio / peak
    audio = np.clip(audio, -1.0, 1.0)
    int16 = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(int16.tobytes())


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read a mono/stereo WAV, return (sample_rate, float32 mono)."""
    import wave

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return sr, data


# ---------------------------------------------------------------------------
# GLaDOS TTS (Style-Bert-VITS2)
# ---------------------------------------------------------------------------

def load_glados(model_dir: Path):
    """Load GLaDOS model. Returns (model, language_enum)."""
    import torch
    import style_bert_vits2
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel as SBV2TTS

    log.info("[1/4] Loading GLaDOS model...")

    model_path = model_dir / "Portal_GLaDOS_v1_e782_s50000.safetensors"
    config_path = model_dir / "config.json"
    style_path = model_dir / "style_vectors.npy"

    for p, name in [(model_path, "model"), (config_path, "config"), (style_path, "style vectors")]:
        if not p.exists():
            log.error("Missing %s: %s", name, p)
            sys.exit(1)

    bert_models.load_model(Languages.EN, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.EN, "ku-nlp/deberta-v2-large-japanese-char-wwm")

    import style_bert_vits2.nlp.bert_models as bm
    loaded_models = bm.__loaded_models
    if Languages.EN in loaded_models:
        for param in loaded_models[Languages.EN].parameters():
            param.data = param.data.float()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SBV2TTS(
        model_path=str(model_path),
        config_path=str(config_path),
        style_vec_path=str(style_path),
        device=device,
    )

    log.info("GLaDOS model loaded on %s", device)
    return model, Languages.EN


# ---------------------------------------------------------------------------
# Concatenate chunks with silence gaps — normalize each chunk
# ---------------------------------------------------------------------------

def concatenate_chunks(
    chunks: list[tuple[str, float]],
    model,
    language,
) -> tuple[np.ndarray, int]:
    """Synthesize each chunk, concatenate with silence gaps.

    Returns (merged_float32_in_range_minus1_to_1, sample_rate).
    """
    # Synthesize probe to learn real sample rate
    probe_sr, probe_audio = model.infer(text="a", language=language)
    write_wav(TTS_CACHE_DIR / "_probe.wav", probe_sr, probe_audio)
    sr = probe_sr

    merged = np.array([], dtype=np.float32)

    for i, (chunk_text, pause) in enumerate(chunks):
        log.info("[3/4] Synthesizing: %s", chunk_text[:60])
        chunk_sr, audio = model.infer(text=chunk_text, language=language)

        # Normalize: GLaDOS returns int16-scale float32, not [-1,1]
        peak = float(np.max(np.abs(audio)))
        if peak > 0.001:
            audio = audio / peak

        if abs(chunk_sr - sr) > 1:
            # Resample without scipy: linear interpolation via numpy
            old_x = np.arange(len(audio))
            new_len = int(len(audio) * sr / chunk_sr)
            new_x = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(new_x, old_x, audio)

        merged = np.concatenate([merged, audio])

        if i < len(chunks) - 1 and pause > 0:
            gap = np.zeros(int(sr * pause), dtype=np.float32)
            merged = np.concatenate([merged, gap])

    # Final safety clip
    merged = np.clip(merged, -1.0, 1.0)
    return merged, sr


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play_audio(audio: np.ndarray, sr: int, device_index: int) -> None:
    import sounddevice as sd

    log.info("[2/4] Using audio device: %s", _device_name(device_index))
    log.info("[4/4] Playing audio...")

    sd.play(audio, samplerate=sr, device=device_index)
    sd.wait()

    log.info("Playback completed successfully.")


def _device_name(device_index: int) -> str:
    import sounddevice as sd

    try:
        return sd.query_devices(device_index)["name"]
    except Exception:
        return f"device {device_index}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Speak text in GLaDOS voice.")
    parser.add_argument("text", nargs="+", help="Text to speak (one or more words).")
    parser.add_argument(
        "--device",
        type=int,
        default=4,
        help="Audio device index for ROG Strix Go headphones (default: 4).",
    )
    args = parser.parse_args()

    text = " ".join(args.text).strip()
    if not text:
        log.error("No text provided.")
        sys.exit(1)

    log.info("Input: %s", text)

    try:
        model_dir = resolve_model_dir()
        model, language = load_glados(model_dir)

        chunks = split_text(text)
        if not chunks:
            log.error("No chunks to speak after splitting.")
            sys.exit(1)

        log.info("[2/4] Splitting into %d chunk(s) with pauses", len(chunks))

        merged, sr = concatenate_chunks(chunks, model, language)
        play_audio(merged, sr, device_index=args.device)

    except KeyboardInterrupt:
        log.info("Interrupted.")
        sys.exit(130)
    except Exception:
        log.exception("Playback failed")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
