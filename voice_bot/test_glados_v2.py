"""Test GLaDOS synthesis with proper float32 handling."""
import torch
import sys
import os

# Force float32 globally
torch.set_default_dtype(torch.float32)

sys.path.insert(0, r"C:\Users\trist\Downloads\muster-roll-extracted\voice_bot\venv\Lib\site-packages")

from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
import numpy as np

# Access internal model dict and convert to float32
# First load the model
bert_models.load_model(Languages.EN, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.EN, "ku-nlp/deberta-v2-large-japanese-char-wwm")

# Access the private loaded models dict - cast to float32
import style_bert_vits2.nlp.bert_models as bm

# Access private dict via module internals
loaded_models = bm.__loaded_models  # type: ignore
print("Loaded models:", list(loaded_models.keys()))

if Languages.EN in loaded_models:
    en_bert = loaded_models[Languages.EN]
    print("EN BERT dtype before:", next(en_bert.parameters()).dtype)
    
    # Convert all params to float32
    for param in en_bert.parameters():
        param.data = param.data.float()
    
    print("EN BERT dtype after:", next(en_bert.parameters()).dtype)

# Load TTS model  
model_dir = r"C:\Users\trist\Downloads\muster-roll-extracted\voice_bot\model_assets\Portal_GLaDOS_v1"
model_path = os.path.join(model_dir, "Portal_GLaDOS_v1_e782_s50000.safetensors")
config_path = os.path.join(model_dir, "config.json")
style_path = os.path.join(model_dir, "style_vectors.npy")

tts_model = TTSModel(
    model_path=model_path,
    config_path=config_path,
    style_vec_path=style_path,
    device="cpu",
)
print("\nTTS model loaded")

# Test
test_text = "Hello. Would you like to begin the test?"
print(f"\nSynthesizing: {test_text}")

sr, audio = tts_model.infer(text=test_text, language=Languages.EN)
print(f"SUCCESS! SR={sr}, length={len(audio)} samples ({len(audio)/sr:.2f}s)")

# Save WAV
out_path = r"C:\Users\trist\Downloads\muster-roll-extracted\voice_bot\glados_test\glados_test_1.wav"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
import wave
with wave.open(out_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes(audio.astype(np.int16).tobytes())
print(f"Saved: {out_path}")

# Also test a second line
test2 = "Welcome, my new test subject. Your chances of survival are slim to none."
print(f"\nSynthesizing: {test2}")
sr2, audio2 = tts_model.infer(text=test2, language=Languages.EN)

out_path2 = out_path.replace("_1.wav", "_2.wav")
with wave.open(out_path2, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr2)
    wf.writeframes(audio2.astype(np.int16).tobytes())
print(f"Saved: {out_path2}")

print("\n=== GLaDOS TTS working! ===")
