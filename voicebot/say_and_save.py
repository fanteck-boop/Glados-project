#!/usr/bin/env python3
"""say_and_save.py — GLaDOS says it AND saves to cache."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from speak import load_glados, resolve_model_dir, concatenate_chunks, split_text, write_wav, TTS_CACHE_DIR
import time

text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hello alba"
model_dir = resolve_model_dir()
model, lang = load_glados(model_dir)
chunks = split_text(text)
merged, sr = concatenate_chunks(chunks, model, lang)
ts = int(time.time())
out = TTS_CACHE_DIR / f"hello_alba_{ts}.wav"
write_wav(out, sr, merged)
print(f"Saved: {out}")
print(f"Duration: {len(merged)/sr:.2f}s")
