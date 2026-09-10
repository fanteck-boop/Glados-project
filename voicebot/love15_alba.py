#!/usr/bin/env python3
"""
love15_alba.py — GLaDOS says "I love you Alba" 15 times in one audio file.

Loads the model once, synthesizes all 15 phrases, concatenates with
short gaps, plays through ROG Strix Go headphones.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
CACHE = PROJECT_DIR / "cache"
CACHE.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("love15_alba")

PAUSE_BETWEEN = 0.25  # seconds between each phrase
PHRASES = 15
PHRASE_TEXT = "I love you Alba"

PAUSE_MAP = {",": 0.20, ";": 0.30, ":": 0.30, ".": 0.55, "?": 0.55, "!": 0.55}


def split_text(text):
    text = re.sub(r"([,;:.!?])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    segments = []
    for part in re.split(r"(?<=[.!?])\s+", text):
        part = part.strip()
        if part:
            segments.append(part)
    chunks = []
    for seg in segments:
        words = seg.split()
        for i in range(0, len(words), 16):
            chunk = " ".join(words[i:i + 16])
            last_char = chunk[-1] if chunk else ""
            pause = PAUSE_MAP.get(last_char, 0.0)
            chunks.append((chunk, pause))
    return chunks


def write_wav(path, sr, audio):
    import wave
    audio = np.clip(audio, -1.0, 1.0)
    int16 = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(int16.tobytes())


def resolve_model_dir():
    marker = "Portal_GLaDOS_v1_e782_s50000.safetensors"
    fallback = PROJECT_DIR.parent / "voice_bot" / "model_assets" / "Portal_GLaDOS_v1"
    if (fallback / marker).exists():
        return fallback.resolve()
    local = (PROJECT_DIR / "model_assets" / "Portal_GLaDOS_v1").resolve()
    if (local / marker).exists():
        return local
    log.error("GLaDOS model not found in %s or %s", fallback, local)
    sys.exit(1)


def load_glados(model_dir):
    import torch
    import style_bert_vits2
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel as SBV2TTS

    log.info("[1/3] Loading GLaDOS model once...")
    bert_models.load_model(Languages.EN, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.EN, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    import style_bert_vits2.nlp.bert_models as bm
    loaded = bm.__loaded_models
    if Languages.EN in loaded:
        for p in loaded[Languages.EN].parameters():
            p.data = p.data.float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SBV2TTS(
        model_path=str(model_dir / "Portal_GLaDOS_v1_e782_s50000.safetensors"),
        config_path=str(model_dir / "config.json"),
        style_vec_path=str(model_dir / "style_vectors.npy"),
        device=device,
    )
    log.info("GLaDOS loaded on %s", device)
    return model, Languages.EN


def batch_speak():
    model_dir = resolve_model_dir()
    model, language = load_glados(model_dir)

    log.info("[2/3] Synthesizing %d x \"%s\"...", PHRASES, PHRASE_TEXT)

    # Probe for sample rate
    probe_sr, probe_audio = model.infer(text="a", language=language)
    sr = probe_sr

    merged = np.array([], dtype=np.float32)
    gap = np.zeros(int(sr * PAUSE_BETWEEN), dtype=np.float32)

    for i in range(1, PHRASES + 1):
        text = f"{PHRASE_TEXT} number {i} of {PHRASES}" if i > 1 else PHRASE_TEXT
        _, audio = model.infer(text=text, language=language)
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak
        merged = np.concatenate([merged, audio])
        if i < PHRASES:
            merged = np.concatenate([merged, gap])
        log.info("  synthesized %d/%d", i, PHRASES)

    merged = np.clip(merged, -1.0, 1.0)
    duration = len(merged) / sr
    log.info("[2/3] Total audio: %.2f seconds", duration)

    out_wav = CACHE / f"love15_alba_{PHRASES}.wav"
    write_wav(out_wav, sr, merged)
    log.info("Wrote %s", out_wav)

    # Play
    import sounddevice as sd
    log.info("[3/3] Playing through device 4 (Headphones ROG Strix Go)...")
    sd.play(merged, samplerate=sr, device=4)
    sd.wait()
    log.info("Playback completed. All %d phrases spoken.", PHRASES)


if __name__ == "__main__":
    try:
        batch_speak()
    except KeyboardInterrupt:
        log.info("Interrupted.")
        sys.exit(130)
    except Exception:
        log.exception("Failed")
        sys.exit(1)
