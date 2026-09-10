"""Test GLaDOS Style-Bert-VITS2 synthesis."""
import sys
import os

sys.path.insert(0, r"C:\Users\trist\Downloads\muster-roll-extracted\voice_bot\venv\Lib\site-packages")

from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
import numpy as np

# Paths
model_dir = r"C:\Users\trist\Downloads\muster-roll-extracted\voice_bot\model_assets\Portal_GLaDOS_v1"
model_path = os.path.join(model_dir, "Portal_GLaDOS_v1_e782_s50000.safetensors")
config_path = os.path.join(model_dir, "config.json")
style_path = os.path.join(model_dir, "style_vectors.npy")

print("Model path:", model_path)
print("Exists:", os.path.isfile(model_path))

# Load BERT model (English)
print("\nLoading BERT model for English...")
bert_models.load_model(Languages.EN, "ku-nlp/deberta-v2-large-japanese-char-wwm")
bert_models.load_tokenizer(Languages.EN, "ku-nlp/deberta-v2-large-japanese-char-wwm")
print("BERT loaded")

# Load TTS model
print("\nLoading GLaDOS TTS model...")
device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
print("Device:", device)

tts_model = TTSModel(
    model_path=model_path,
    config_path=config_path,
    style_vec_path=style_path,
    device=device,
)
print("TTS model loaded")

# Test synthesis
test_texts = [
    "Hello. Would you like to begin the test?",
    "Welcome, my new test subject. Your chances of survival are slim to none.",
    "The Aperture Science Computer-Aided Enrichment Center is happy to welcome you today.",
]

output_dir = r"C:\Users\trist\Downloads\muster-roll-extracted\voice_bot\glados_test"
os.makedirs(output_dir, exist_ok=True)

for i, text in enumerate(test_texts):
    print(f"\n--- Synthesizing [{i+1}/{len(test_texts)}] ---")
    print(f"Text: {text}")
    
    try:
        sr, audio = tts_model.infer(text=text, language=Languages.EN)
        print(f"Sample rate: {sr}, audio length: {len(audio)} samples ({len(audio)/sr:.2f}s)")
        
        out_path = os.path.join(output_dir, f"glados_test_{i+1}.wav")
        # Write WAV
        import wave
        with wave.open(out_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.astype(np.int16).tobytes())
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n=== All tests done ===")
