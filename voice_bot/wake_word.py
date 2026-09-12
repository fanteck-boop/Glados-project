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

    def __init__(self, config: cfg.Config, stt_model: "STT", gpu_lock: Optional[threading.Lock] = None):
        self.cfg = config
        self.sr: int = config.AUDIO["sample_rate"]
        self.phrase: str = config.WAKE_WORD["phrase"].lower()
        self.sleep_phrase: str = config.WAKE_WORD.get("sleep_phrase", "").lower()
        self.interrupt_phrase: str = config.WAKE_WORD.get("interrupt_phrase", "").lower()
        self.energy_threshold: float = config.WAKE_WORD["vad_energy_threshold"]
        self.min_duration_sec: float = config.WAKE_WORD["vad_min_duration_sec"]
        self.silence_threshold: float = config.AUDIO["silence_threshold"]
        self.silence_frames_to_end: int = config.AUDIO["silence_frames_to_end"]
        self.blocksize: int = 1024

        self._stt_model = stt_model  # injected STT wrapper
        # Two workers: one can be busy handling the current response (which
        # blocks for the whole reply's playback) while the other transcribes
        # whatever's said in the meantime — needed so an interrupt phrase
        # spoken mid-speech is actually heard instead of queuing behind the
        # response that's currently playing. Safe because _gpu_lock (not
        # worker count) is what prevents STT and TTS from ever touching CUDA
        # at the same time — that concurrent-CUDA-access race was the actual
        # cause of the intermittent native crashes/hangs, not concurrency
        # itself.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="wake")
        self._gpu_lock: threading.Lock = gpu_lock or threading.Lock()

        # Every utterance is buffered the same way; what happens once it's
        # transcribed depends on whether the bot is active or asleep (see
        # _on_transcription_done). No separate "wait for the command after
        # the wake word" step — active mode treats each utterance as a
        # command directly.
        self._utt_buffer: list[np.ndarray] = []
        self._utt_last_speech: float = 0.0
        # Guards all of the state above — it's written from both the
        # real-time audio callback thread and the transcription worker thread.
        self._state_lock = threading.Lock()

        # Active = responding to utterances as commands. Asleep = only
        # listening for the wake phrase. Toggled by voice (sleep_phrase /
        # phrase) rather than needing the wake phrase before every command.
        self._active = threading.Event()
        if config.WAKE_WORD.get("start_active", True):
            self._active.set()

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

        # TTS playing flag — set by bot when GLaDOS is speaking. The mic
        # keeps listening throughout (needed to hear the interrupt phrase);
        # what a finished transcription is allowed to trigger is gated on
        # this flag in _on_transcription_done instead of dropping samples.
        self._playing_tts = threading.Event()

        # Diagnostics: loudest energy seen in the last ~2s, logged periodically
        # so it's possible to tell whether the mic is hearing anything at all.
        self._diag_max_energy: float = 0.0
        self._diag_last_log: float = 0.0

        # Most recent per-block energy reading — a plain float, updated from
        # the audio callback thread. Read-only single-value polling (e.g. a
        # GUI meter) doesn't need a lock for this; it's fine if a reader
        # occasionally sees a value from one block ago.
        self._last_energy: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_callback(self, cb: Callable[[str, Optional[str]], None]) -> None:
        """Register a callback for events."""
        with self._lock:
            self._callbacks.append(cb)

    def get_last_energy(self) -> float:
        """Most recent mic energy reading, for a live level meter."""
        return self._last_energy

    def is_active(self) -> bool:
        """True if currently responding to commands (not asleep)."""
        return self._active.is_set()

    def set_playing_tts(self, playing: bool) -> None:
        """Notify the listener whether TTS is currently playing audio.

        The mic keeps listening throughout so the interrupt phrase can be
        heard — _on_transcription_done gates a finished transcription to
        only react to that phrase while this flag is set, ignoring
        everything else (including her own speech bleeding into the mic).
        """
        if playing:
            self._playing_tts.set()
        else:
            # Drop anything that built up while muted before resuming.
            with self._state_lock:
                self._utt_buffer = []
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
        # Kept listening even while GLaDOS is speaking (gated in
        # _on_transcription_done instead) so the interrupt phrase can
        # actually be heard and cut her off mid-response.
        energy = compute_energy(samples)
        self._last_energy = energy

        # Periodic diagnostic: log the loudest energy seen every ~2s so we can
        # tell whether the mic is actually hearing anything while idle.
        now_diag = time.time()
        self._diag_max_energy = max(self._diag_max_energy, energy)
        if now_diag - self._diag_last_log > 2.0:
            logger.info("[diag] max energy in last ~2s: %.6f (threshold %.6f)",
                        self._diag_max_energy, self.energy_threshold)
            self._diag_max_energy = 0.0
            self._diag_last_log = now_diag
        now = time.time()

        # Maintain ring buffer
        self._ring.append(samples)
        while self._ring and sum(len(x) for x in self._ring) > self._ring_max_frames:
            self._ring.popleft()

        finalize = False
        with self._state_lock:
            if energy > self.energy_threshold:
                self._utt_last_speech = now
                self._utt_buffer.append(samples)
            elif self._utt_buffer and now - self._utt_last_speech > 0.6:
                finalize = True

        if finalize:
            self._finalize_utterance()

    def _analyze(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Utterance finalization
    # ------------------------------------------------------------------

    def _finalize_utterance(self) -> None:
        with self._state_lock:
            buf = self._utt_buffer
            self._utt_buffer = []

        if len(buf) == 0:
            return
        if sum(len(x) for x in buf) < int(self.min_duration_sec * self.sr):
            return
        self._transcribe_async(np.concatenate(buf, axis=0))

    def _transcribe_async(self, samples: np.ndarray) -> None:
        def _locked_transcribe() -> Optional[str]:
            with self._gpu_lock:
                return self._stt_model.transcribe(samples)

        future: Future[Optional[str]] = self._executor.submit(_locked_transcribe)
        future.add_done_callback(self._on_transcription_done)

    def _on_transcription_done(self, future: Future) -> None:
        try:
            text: Optional[str] = future.result()
        except Exception as e:
            logger.exception("transcription failed")
            self._notify("error", f"transcription error: {e}")
            return

        if text is None or not text.strip():
            return

        text_lower = text.lower().strip()
        self._notify("utt_ready", text)

        if self._playing_tts.is_set():
            # GLaDOS is currently speaking — the only thing that reacts is
            # the interrupt phrase, so her own speech (or background noise
            # picked up while she talks) can never misfire as a command.
            if self._contains_phrase(text_lower, self.interrupt_phrase):
                logger.info("interrupt phrase detected: %r", text)
                self._notify("interrupt", None)
            return

        if not self.cfg.WAKE_WORD.get("use_wake_word", True):
            logger.info("command captured (always-on): %r", text)
            self._notify("command", text)
            return

        if not self._active.is_set():
            # Asleep — only the wake phrase gets a reaction.
            if self._contains_phrase(text_lower, self.phrase):
                now = time.time()
                with self._state_lock:
                    if now - self._last_wake_time < self._wake_cooldown:
                        return
                    self._last_wake_time = now
                self._active.set()
                logger.info("wake phrase detected — resuming: %r", text)
                self._notify("wake_word", None)
            return

        # Active — the sleep phrase pauses instead of being treated as a command.
        if self.sleep_phrase and self._contains_phrase(text_lower, self.sleep_phrase):
            self._active.clear()
            logger.info("sleep phrase detected — pausing: %r", text)
            self._notify("sleep", None)
            return

        logger.info("command captured: %r", text)
        self._notify("command", text)

    # ------------------------------------------------------------------
    # Phrase matching
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        # Strips everything but letters — including spaces — so a phrase
        # matches regardless of how Whisper happens to word-split it (e.g.
        # "shutdown" vs "shut down" both normalize to "shutdown").
        return "".join(ch for ch in text.lower() if ch.isalpha())

    def _contains_phrase(self, text: str, phrase: str) -> bool:
        if not phrase:
            return False
        return self._normalize(phrase) in self._normalize(text)

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
