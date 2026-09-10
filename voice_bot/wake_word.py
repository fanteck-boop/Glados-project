# Wake-word listener: always-listening mic stream with VAD + Whisper phrase check.

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Callable, Optional

import numpy as np
import sounddevice as sd

from . import config as cfg
from .audio_utils import compute_energy, audio_to_float32

logger = __import__("logging").getLogger("voice_bot.wake_word")


class WakeWordListener:
    """Continuously listens to the microphone and detects a wake phrase.

    Architecture (fully offline, no vendor key needed):
      - SoundDevice InputStream streams mic samples in a background audio thread.
      - A simple energy-based VAD decides when speech is happening.
      - Each completed utterance (speech segment ending in silence) is transcribed
        with a tiny Whisper model. If the transcription contains the wake phrase,
        the next utterance is treated as the command.
      - When a command is captured, it is transcribed and delivered to the bot.

    Limitations (MVP):
      - Requires a short pause between the wake phrase and the command.
      - Runs Whisper on every utterance (tiny model keeps this acceptable on CPU).
      - For faster/more reliable wake-word detection, replace with Porcupine or Silero VAD
        in a future iteration (config.WAKE_WORD.use_engine).
    """

    def __init__(self, config: cfg.Config, stt_model: "STT"):
        self.cfg = config
        self.sr: int = config.AUDIO["sample_rate"]
        self.phrase: str = config.WAKE_WORD["phrase"].lower()
        self.energy_threshold: float = config.WAKE_WORD["vad_energy_threshold"]
        self.min_duration_sec: float = config.WAKE_WORD["vad_min_duration_sec"]
        self.silence_threshold: float = config.AUDIO["silence_threshold"]
        self.silence_frames_to_end: int = config.AUDIO["silence_frames_to_end"]
        self.blocksize: int = 1024

        self._stt_model = stt_model  # injected STT wrapper
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wake")

        # State machine
        self._state: str = "idle"  # idle | expect_command
        self._utt_buffer: list[np.ndarray] = []
        self._utt_last_speech: float = 0.0
        self._cmd_buffer: list[np.ndarray] = []
        self._cmd_last_speech: float = 0.0

        # Continuous ring buffer of recent audio (for debugging / fallback)
        ring_max = int(self.cfg.AUDIO["sample_rate"] * 4.0)
        self._ring: deque[np.ndarray] = deque(maxlen=ring_max)
        self._ring_max_frames: int = ring_max

        # Control
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[str, Optional[str]], None]] = []

        # Suppress repeated wake detections within this window (seconds)
        self._wake_cooldown: float = 4.0
        self._last_wake_time: float = 0.0

        # TTS playing flag — set by bot when GLaDOS is speaking
        self._playing_tts = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_callback(self, cb: Callable[[str, Optional[str]], None]) -> None:
        """Register a callback for events."""
        with self._lock:
            self._callbacks.append(cb)

    def set_playing_tts(self, playing: bool) -> None:
        """Notify the listener whether TTS is currently playing audio."""
        if playing:
            self._playing_tts.set()
        else:
            self._playing_tts.clear()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="wake-main")
        self._thread.start()
        logger.info("WakeWordListener started (energy threshold=%.3f)", self.energy_threshold)

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning("error stopping stream: %s", e)
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("WakeWordListener stopped")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._stream = sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sr,
                blocksize=self.blocksize,
                dtype="int16",
                device=self.cfg.AUDIO.get("input_device"),
            )
            self._stream.start()
            while self._running:
                time.sleep(0.05)
                self._analyze()
        except Exception as e:
            logger.exception("wake listener loop error")
            self._notify("error", str(e))
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop()
                    self._stream.close()
                    self._stream = None
            except Exception:
                pass

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        try:
            if status:
                logger.debug("audio status: %s", status)
            samples = audio_to_float32(indata[:, 0])
            self._on_samples(samples)
        except Exception:
            logger.exception("audio callback crashed — restarting stream safety net")

    def _on_samples(self, samples: np.ndarray) -> None:
        # If GLaDOS is currently speaking, discard mic input to avoid feedback/capture
        if self._playing_tts.is_set():
            return

        energy = compute_energy(samples)
        now = time.time()

        # Maintain ring buffer
        self._ring.append(samples)
        while self._ring and sum(len(x) for x in self._ring) > self._ring_max_frames:
            self._ring.popleft()

        if energy > self.energy_threshold:
            self._utt_last_speech = now
            self._cmd_last_speech = now
            if self._state == "idle":
                self._utt_buffer.append(samples)
            elif self._state == "expect_command":
                self._cmd_buffer.append(samples)
        else:
            # Silence: decide whether to end an utterance
            if self._state == "idle" and self._utt_buffer:
                if now - self._utt_last_speech > 0.6:
                    self._finalize_utterance("idle")
            elif self._state == "expect_command" and self._cmd_buffer:
                if now - self._cmd_last_speech > 0.8:
                    self._finalize_utterance("expect_command")

    def _analyze(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Utterance finalization
    # ------------------------------------------------------------------

    def _finalize_utterance(self, source: str) -> None:
        buf: list[np.ndarray]
        if source == "idle":
            buf = self._utt_buffer
            self._utt_buffer = []
            if len(buf) == 0:
                return
            if sum(len(x) for x in buf) < int(self.min_duration_sec * self.sr):
                return
            self._transcribe_async(np.concatenate(buf, axis=0), source="idle")
        elif source == "expect_command":
            buf = self._cmd_buffer
            self._cmd_buffer = []
            self._state = "idle"
            if len(buf) == 0:
                return
            if sum(len(x) for x in buf) < int(0.3 * self.sr):
                return
            self._transcribe_async(np.concatenate(buf, axis=0), source="command")
        else:
            raise ValueError(f"unknown source: {source}")

    def _transcribe_async(self, samples: np.ndarray, source: str) -> None:
        future: Future[tuple[Optional[str], str]] = self._executor.submit(self._stt_model.transcribe, samples)
        future.add_done_callback(lambda f: self._on_transcription_done(f, source))

    def _on_transcription_done(self, future: Future, source: str) -> None:
        try:
            text: Optional[str] = future.result()
        except Exception as e:
            logger.exception("transcription failed")
            self._notify("error", f"transcription error: {e}")
            return

        if text is None:
            return

        text_lower = text.lower().strip()
        self._notify("utt_ready", text)

        if not self.cfg.WAKE_WORD.get("use_wake_word", True):
            if text.strip():
                logger.info("command captured (always-on): %r", text)
                self._notify("command", text)
            return

        if source == "idle":
            if self._contains_wake_phrase(text_lower):
                now = time.time()
                if now - self._last_wake_time < self._wake_cooldown:
                    return
                self._last_wake_time = now
                logger.info("wake phrase detected: %r", text)
                self._notify("wake_word", None)
                self._state = "expect_command"
                self._cmd_buffer = []
                self._cmd_last_speech = time.time()
        elif source == "command":
            logger.info("command captured: %r", text)
            self._notify("command", text)
            self._cmd_buffer = []
        else:
            raise ValueError(f"unknown source: {source}")

    # ------------------------------------------------------------------
    # Wake phrase matching
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        out = []
        for ch in text.lower():
            if ch.isalpha() or ch.isspace():
                out.append(ch)
            else:
                out.append(" ")
        return " ".join("".join(out).split())

    def _contains_wake_phrase(self, normalized_text: str) -> bool:
        norm_phrase = self._normalize(self.phrase)
        return norm_phrase in normalized_text

    # ------------------------------------------------------------------
    # Notification
    # ------------------------------------------------------------------

    def _notify(self, event: str, data: Any) -> None:
        with self._lock:
            cbs = list(self._callbacks)
        for cb in cbs:
            try:
                cb(event, data)
            except Exception as e:
                logger.exception("wake callback error: %s", e)
