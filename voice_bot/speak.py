#!/usr/bin/env python3
"""Phase 1 — speak.py

Takes a line of text from the command line, synthesizes it in the GLaDOS voice,
and plays it through the ROG Strix Go headphones. One job, one shot, one exit.

Usage:
    python speak.py "Testing. This is GLaDOS speaking."
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — fixed for this machine
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent / "model_assets" / "Portal_GLaDOS_v1"

MODEL_PATH = MODEL_DIR / "Portal_GLaDOS_v1_e782_s50000.safetensors"
CONFIG_PATH = MODEL_DIR / "config.json"
STYLE_PATH = MODEL_DIR / "style_vectors.npy"

BERT_MODEL = "ku-nlp/deberta-v2-large-japanese-char-wwm"

# SoundDevice device index for ROG Strix Go headphones
OUTPUT_DEVICE = 4

# ---------------------------------------------------------------------------
# Step 0 — validate we have what we need
# ---------------------------------------------------------------------------


def _check_files() -> None:
    missing = [p for p in (MODEL_PATH, CONFIG_PATH, STYLE_PATH) if not p.exists()]
    if missing:
        print(f"ERROR: missing model files: {', '.join(missing)}")
        print(f"Expected at: {MODEL_DIR}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1 — load GLaDOS
# ---------------------------------------------------------------------------


def _load_glados() -> object:
    print("[1/4] Loading GLaDOS model...")

    sys.path.insert(0, str(Path(__file__).resolve().parent / "venv" / "Lib" / "site-packages"))

    import torch
    import style_bert_vits2
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel as SBV2TTS

    # Load BERT (English)
    bert_models.load_model(Languages.EN, BERT_MODEL)
    bert_models.load_tokenizer(Languages.EN, BERT_MODEL)

    # Convert BERT to float32 (torch 2.x compat)
    import style_bert_vits2.nlp.bert_models as bm
    loaded = bm.__loaded_models
    if Languages.EN in loaded:
        for param in loaded[Languages.EN].parameters():
            param.data = param.data.float()

    # Load GLaDOS TTS model
    model = SBV2TTS(
        model_path=str(MODEL_PATH),
        config_path=str(CONFIG_PATH),
        style_vec_path=str(STYLE_PATH),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    print("      GLaDOS loaded.")
    return model


# ---------------------------------------------------------------------------
# Step 2 — confirm audio device
# ---------------------------------------------------------------------------


def _pick_device() -> tuple[int, str]:
    import sounddevice as sd

    devs = sd.query_devices()
    name = devs[OUTPUT_DEVICE]["name"]
    print(f"[2/4] Using audio device: {name}")
    return OUTPUT_DEVICE, name


# ---------------------------------------------------------------------------
# Step 3 — synthesize
# ---------------------------------------------------------------------------


def _synthesize(model: object, text: str) -> tuple[bytes, int]:
    print(f"[3/4] Synthesizing: {text!r}")

    from style_bert_vits2.constants import Languages

    sr, audio = model.infer(text=text, language=Languages.EN)
    print(f"      Generated {len(audio)} samples at {sr} Hz ({len(audio)/sr:.2f}s)")
    return audio.astype("int16").tobytes(), sr


# ---------------------------------------------------------------------------
# Step 4 — play
# ---------------------------------------------------------------------------


def _play(wav_bytes: bytes, sr: int, device: int) -> None:
    print("[4/4] Playing audio...")

    import numpy as np
    import sounddevice as sd

    samples = np.frombuffer(wav_bytes, dtype="int16").astype("float32") / 32768.0
    sd.play(samples, samplerate=sr, device=device)
    sd.wait()
    print("      Playback completed successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: python speak.py \"text to speak\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:]).strip()

    _check_files()
    model = _load_glados()
    device, dev_name = _pick_device()
    wav_bytes, sr = _synthesize(model, text)
    _play(wav_bytes, sr, device)

    print("Done.")


if __name__ == "__main__":
    main()
