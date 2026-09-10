# Speech-to-text (STT) wrapper around OpenAI Whisper.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

# Relative imports (package mode) or absolute (standalone mode)
try:
    from . import config as cfg
except ImportError:
    import config as cfg  # type: ignore[no-redef]

try:
    from .audio_utils import float32_to_int16, write_wav
except ImportError:
    from audio_utils import float32_to_int16, write_wav  # type: ignore[no-redef]

logger = logging.getLogger("voice_bot.stt")


class STT:
    """Whisper-based speech-to-text.

    Loads the chosen Whisper model once and reuses it for all transcriptions.
    """

    def __init__(self, config: cfg.Config):
        self._config = config
        self._model_name: str = config.STT["model"]
        self._device: str = config.STT["device"]
        self._language: Optional[str] = config.STT["language"]
        self._model = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._loaded:
            return
        logger.info("loading Whisper model %s on %s...", self._model_name, self._device)
        import whisper

        self._model = whisper.load_model(self._model_name, device=self._device)
        self._loaded = True
        logger.info("Whisper model loaded")

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    def transcribe(self, samples: np.ndarray) -> Optional[str]:
        """Transcribe float32 mono audio (any sample rate; Whisper resamples internally).

        Returns the transcribed text, or None if empty / failed.
        """
        if not self._loaded:
            raise RuntimeError("STT model not loaded; call load() first")
        if len(samples) == 0:
            return None

        audio = samples.astype(np.float32)

        # Whisper expects 16kHz; it handles resampling if needed, but we ensure
        # we pass something reasonable. If the input is already 16kHz, great.
        try:
            result = self._model.transcribe(
                audio,
                language=self._language,
                fp16=False,  # run in FP32 on CPU for stability
                suppress_tokens=None,
            )
        except Exception as e:
            logger.exception("whisper transcribe failed")
            return None

        text = (result.get("text") or "").strip()
        return text if text else None

    def transcribe_file(self, path: Path) -> Optional[str]:
        sr, samples = __import__("voice_bot.audio_utils", fromlist=["read_wav"]).read_wav(path)
        return self.transcribe(samples)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def segment_is_speech(self, samples: np.ndarray, threshold: Optional[float] = None) -> bool:
        """Crude speech/amplitude heuristic. Not a real VAD; for diagnostics."""
        if threshold is None:
            threshold = self._config.AUDIO["silence_threshold"]
        energy = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        return energy >= threshold
