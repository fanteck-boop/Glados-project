# Voice Clone Bot
#
# A Python bot that:
#   - Listens always for a wake word ("hey bot")
#   - Records your speech after wake word
#   - Transcribes it (Whisper STT)
#   - Sends it to a local LLM (Ollama Qwen 3.6) for a response
#   - Speaks the response back using a cloned/custom TTS voice (Coqui XTTS v2)
#
# Capabilities (Hermes-like):
#   - Read and analyze files you point it at
#   - Answer questions about file contents
#   - (Future) run safe commands, browse web, remember context
#
# Requirements:
#   - Ollama running locally with Qwen 3.6 pulled (ollama pull qwen3.6)
#   - A GPU with enough VRAM for XTTS v2 (or run on CPU if patient)
#   - A microphone

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Convenience imports for quick testing
from .tts import TTS
from .stt import STT
from .llm import OllamaClient
from .wake_word import WakeWordListener
from .file_tool import read_file, extract_paths
