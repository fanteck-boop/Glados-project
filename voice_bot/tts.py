# Text-to-speech with XTTS v2, pyttsx3 fallback, and GLaDOS (Style-Bert-VITS2).

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ------------------------------------------------------------------
# Relative imports (package mode) or absolute (standalone mode)
# ------------------------------------------------------------------
try:
    from . import config as cfg
    from .audio_utils import write_wav
except ImportError:
    import config as cfg
    from audio_utils import write_wav

logger = logging.getLogger("voice_bot.tts")


# ---------------------------------------------------------------------------
# Internal helper: convert SBV2's loaded BERT to float32 (torch 2.x compat)
# ---------------------------------------------------------------------------

def _convert_sbv2_bert_to_float32(language) -> None:
    """Cast the Style-Bert-VITS2 internal BERT model to float32.

    Style-Bert-VITS2 stores loaded BERT models in a module-private dict
    named __loaded_models in style_bert_vits2.nlp.bert_models.
    """
    import style_bert_vits2.nlp.bert_models as bm

    # The dict is module-level __loaded_models (NOT name-mangled because
    # it's not inside a class). Access via the raw name.
    loaded_models = bm.__loaded_models

    if language in loaded_models:
        en_bert = loaded_models[language]
        for param in en_bert.parameters():
            param.data = param.data.float()
        logger.info("Converted SBV2 EN BERT to float32")
        return

    logger.warning("No loaded BERT found for language %s; skipping float32 conversion", language)


# ---------------------------------------------------------------------------

class TTS:
    """Voice synthesis with three backend options:

    1. XTTS v2 (Coqui) — voice cloning from a reference clip, GPU preferred
    2. GLaDOS (Style-Bert-VITS2) — pre-trained GLaDOS voice model
    3. System TTS via pyttsx3 (Windows SAPI5 or espeak) — no GPU needed

    For GLaDOS mode:
      - Uses the Style-Bert-VITS2 library + Portal_GLaDOS_v1 model.
      - Model files: model_assets/Portal_GLaDOS_v1/{model.safetensors, config.json, style_vectors.npy}
      - Requires: pip install style-bert-vits2 torch (CPU OK)
      - BERT model: loaded from ku-nlp/deberta-v2-large-japanese-char-wwm (auto-downloaded)
    """

    def __init__(self, config: cfg.Config):
        self._config = config
        self._backend: str = config.TTS["backend"]
        self._voice_ref: str = config.TTS.get("voice_reference", "")
        self._speaker_id: int = config.TTS.get("speaker_id", 0)
        self._language: str = config.TTS.get("language", "en")
        self._device: str = config.TTS.get("device", "cpu")
        self._speed: float = config.TTS.get("speed", 1.0)

        # GLaDOS-specific paths (override from config if set)
        self._glados_model_dir: str = config.TTS.get("glados_model_dir", "")
        self._glados_bert_model: str = config.TTS.get("glados_bert_model", "ku-nlp/deberta-v2-large-japanese-char-wwm")

        self._xtts_model = None
        self._xtts_loaded = False
        self._pyttsx3_engine = None

        # GLaDOS / Style-Bert-VITS2
        self._sbv2_model = None
        self._sbv2_loaded = False
        self._sbv2_bert_loaded = False

        # temp dir for synthesized clips
        self._cache_dir = Path(__file__).resolve().parent / "tts_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        if self._xtts_loaded and self._sbv2_loaded:
            return
        if self._backend == "xtts":
            self._try_load_xtts()
        elif self._backend == "glados":
            self._try_load_glados()
        # pyttsx3 is loaded lazily on first fallback use

    def _try_load_xtts(self) -> None:
        try:
            from TTS.api import TTS as XTTS  # type: ignore
        except Exception as e:
            logger.warning("cannot import Coqui TTS: %s", e)
            return
        try:
            logger.info("loading XTTS v2 model on %s...", self._device)
            device = "cuda" if self._device == "cuda" else "cpu"
            self._xtts_model = XTTS(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
                gpu=(self._device == "cuda"),
            )
            self._xtts_loaded = True
            logger.info("XTTS v2 loaded")
        except Exception as e:
            logger.warning("failed to load XTTS v2: %s", e)
            self._xtts_model = None

    def _try_load_glados(self) -> None:
        """Load the Style-Bert-VITS2 GLaDOS model and BERT encoder."""
        try:
            import torch
            import style_bert_vits2
            from style_bert_vits2.nlp import bert_models
            from style_bert_vits2.constants import Languages
            from style_bert_vits2.tts_model import TTSModel as SBV2TTS
        except Exception as e:
            logger.warning("cannot import style-bert-vits2: %s", e)
            return

        try:
            logger.info("loading GLaDOS (Style-Bert-VITS2) model...")

            # Determine model directory
            model_dir = self._glados_model_dir or str(
                Path(__file__).resolve().parent / "model_assets" / "Portal_GLaDOS_v1"
            )

            model_path = Path(model_dir) / "Portal_GLaDOS_v1_e782_s50000.safetensors"
            config_path = Path(model_dir) / "config.json"
            style_path = Path(model_dir) / "style_vectors.npy"

            if not model_path.exists():
                raise FileNotFoundError(
                    f"GLaDOS model not found at {model_path}. "
                    "Download from HuggingFace: WarriorMama777/GLaDOS_TTS"
                )

            # Load BERT model (English) — required by SBV2
            bert_models.load_model(Languages.EN, self._glados_bert_model)
            bert_models.load_tokenizer(Languages.EN, self._glados_bert_model)
            self._sbv2_bert_loaded = True

            # Convert BERT to float32 via the module-level helper
            _convert_sbv2_bert_to_float32(Languages.EN)

            device = "cuda" if self._device == "cuda" else "cpu"

            self._sbv2_model = SBV2TTS(
                model_path=str(model_path),
                config_path=str(config_path),
                style_vec_path=str(style_path),
                device=device,
            )
            self._sbv2_loaded = True
            logger.info("GLaDOS (Style-Bert-VITS2) model loaded on %s", device)

            if device == "cuda":
                logger.info("warming up CUDA kernels...")
                t0 = time.time()
                self._synthesize_glados("Warming up.")
                logger.info("CUDA warmup done in %.1fs", time.time() - t0)

        except Exception as e:
            logger.warning("failed to load GLaDOS model: %s", e)
            self._sbv2_model = None
            self._sbv2_loaded = False

    # ------------------------------------------------------------------
    # Synthesize
    # ------------------------------------------------------------------

    def synthesize(self, text: str) -> Optional[Path]:
        """Synthesize speech for `text` and return path to a WAV file.

        Tries the configured backend first; on failure falls back through
        the chain: glados → xtts → pyttsx3. Returns None on total failure.
        """
        text = text.strip()
        if not text:
            return None

        logger.info("synthesizing TTS (%s): %r ...", self._backend, text[:80])

        # Try configured backend
        if self._backend == "glados":
            if self._sbv2_loaded and self._sbv2_model is not None:
                try:
                    return self._synthesize_glados(text)
                except Exception as e:
                    logger.warning("GLaDOS synthesis failed, falling back: %s", e)

        if self._backend == "xtts":
            if self._xtts_loaded and self._xtts_model is not None:
                try:
                    return self._synthesize_xtts(text)
                except Exception as e:
                    logger.warning("XTTS synthesis failed, falling back: %s", e)

        # Fallback chain
        if self._sbv2_loaded and self._sbv2_model is not None:
            try:
                return self._synthesize_glados(text)
            except Exception as e:
                logger.warning("GLaDOS fallback failed: %s", e)

        if self._xtts_loaded and self._xtts_model is not None:
            try:
                return self._synthesize_xtts(text)
            except Exception as e:
                logger.warning("XTTS fallback failed: %s", e)

        try:
            return self._synthesize_pyttsx3(text)
        except Exception as e:
            logger.warning("pyttsx3 synthesis failed: %s", e)

        return None

    def _synthesize_glados(self, text: str) -> Path:
        """Synthesize using the GLaDOS Style-Bert-VITS2 model."""
        import wave
        from style_bert_vits2.constants import Languages
        from style_bert_vits2.nlp import bert_models

        # Ensure BERT is still loaded and float32
        if not self._sbv2_bert_loaded:
            bert_models.load_model(Languages.EN, self._glados_bert_model)
            bert_models.load_tokenizer(Languages.EN, self._glados_bert_model)
            _convert_sbv2_bert_to_float32(Languages.EN)
            self._sbv2_bert_loaded = True

        t0 = time.time()
        sr, audio = self._sbv2_model.infer(text=text, language=Languages.EN)
        dt = time.time() - t0
        logger.info("GLaDOS synthesis done in %.1fs -> %d samples (%.2fs)", dt, len(audio), len(audio) / sr)

        audio = self._fade_edges(audio, sr)

        out_name = f"glados_{abs(hash(text)) % 1000000}.wav"
        out_path = self._cache_dir / out_name

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(audio.astype(np.int16).tobytes())

        return out_path

    @staticmethod
    def _fade_edges(audio: np.ndarray, sr: int, fade_ms: float = 15.0) -> np.ndarray:
        """Apply a short linear fade-in and fade-out to int16 audio.

        VITS-style models can start/end on an abrupt sample-boundary
        discontinuity, which plays back as a click/pop — since each chunk
        (sentence/clause, see bot.py's _say_with_pauses) is synthesized and
        played separately, that click lands right at the chunk boundary,
        heard as a scratch at the end of words/sentences. A short fade on
        both ends removes it; only the tail was covered before.
        """
        audio = audio.astype(np.float64)
        fade_len = min(len(audio) // 2, int(sr * fade_ms / 1000))
        if fade_len > 0:
            audio[:fade_len] *= np.linspace(0.0, 1.0, fade_len)
            audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
        return np.clip(audio, -32768, 32767).astype(np.int16)

    def _synthesize_xtts(self, text: str) -> Path:
        from TTS.api import TTS as XTTS  # type: ignore

        sr = 24000  # XTTS default output sr
        speaker_wav = self._voice_ref or None
        lang = self._language

        out_name = f"xtts_{abs(hash(text)) % 1000000}.wav"
        out_path = self._cache_dir / out_name

        t0 = time.time()
        self._xtts_model.tts_to_file(
            text=text,
            speaker_wav=speaker_wav,
            language=lang,
            file_path=str(out_path),
        )
        dt = time.time() - t0
        logger.info("XTTS done in %.1fs -> %s", dt, out_path)
        return out_path

    def _synthesize_pyttsx3(self, text: str) -> Path:
        try:
            import pyttsx3
        except Exception as e:
            raise RuntimeError(f"pyttsx3 not available: {e}")

        engine = pyttsx3.init()
        engine.setProperty("rate", int(200 * self._speed))

        voices = engine.getProperty("voices")
        if voices:
            chosen = None
            for v in voices:
                name = getattr(v, "name", "")
                langs = getattr(v, "languages", [])
                if "en" in str(langs).lower():
                    chosen = v
                    break
            if chosen is None:
                chosen = voices[0]
            engine.setProperty("voice", chosen.id)

        out_path = self._cache_dir / f"pyttsx3_{abs(hash(text)) % 1000000}.wav"

        try:
            engine.save_to_file(text, str(out_path))
            engine.runAndWait()
        except Exception:
            logger.warning("pyttsx3 save_to_file not supported; TTS will be silent")
            write_wav(out_path, 24000, np.zeros(int(24000 * 0.5)))

        return out_path

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def play(self, wav_path: Path, audio_cfg: dict, interrupted: Optional[threading.Event] = None) -> None:
        """Play a WAV file via sounddevice to the configured output device.

        Polls for `interrupted` on this same thread and calls sd.stop() from
        here rather than blocking on sd.wait() — a cross-thread sd.stop()
        does not reliably release a concurrent sd.wait() on this WASAPI
        setup (verified: it just hangs), so the wait has to be a same-thread
        poll loop instead for the interrupt phrase to actually work.
        """
        import soundfile as sf
        import sounddevice as sd

        data, sr = sf.read(str(wav_path), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        out_device = audio_cfg.get("output_device")
        # WASAPI output devices reject sample rates other than their native
        # rate unless auto_convert is enabled (needed since GLaDOS outputs
        # 44.1kHz but the headset's WASAPI endpoint is natively 48kHz).
        extra_settings = None
        try:
            hostapi = sd.query_devices(out_device)["hostapi"] if out_device is not None else None
            if hostapi is not None and "wasapi" in sd.query_hostapis(hostapi)["name"].lower():
                extra_settings = sd.WasapiSettings(auto_convert=True)
        except Exception:
            pass
        if out_device is not None:
            sd.play(data, samplerate=sr, device=out_device, extra_settings=extra_settings)
        else:
            sd.play(data, samplerate=sr)

        if interrupted is None:
            sd.wait()
        else:
            stream = sd.get_stream()
            while stream is not None and stream.active:
                if interrupted.is_set():
                    sd.stop()
                    break
                time.sleep(0.05)

        logger.debug("playback finished: %s (device=%s, sr=%d, dur=%.2fs)",
                     wav_path, out_device, sr, len(data) / sr)

    def stop(self) -> None:
        """Immediately halt any in-progress playback.

        Only reliable when called from the same thread that's polling in
        play() — see the note there. Kept for the case play() was called
        without an interrupted Event (e.g. external callers).
        """
        import sounddevice as sd

        sd.stop()

    # ------------------------------------------------------------------
    # Lazy engine getter
    # ------------------------------------------------------------------

    @property
    def is_xtts(self) -> bool:
        return self._xtts_loaded and self._xtts_model is not None

    @property
    def is_glados(self) -> bool:
        return self._sbv2_loaded and self._sbv2_model is not None
