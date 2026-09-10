# ------------------------------------------------------------------
# Voice Bot — GLaDOS voice, Ollama Qwen, Whisper STT, wake-word loop
# ------------------------------------------------------------------

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as cfg
from .tts import TTS
from .stt import STT
from .llm import make_llm_client
from .wake_word import WakeWordListener
from .file_tool import read_file, extract_paths

log = logging.getLogger("voice_bot.bot")


# ------------------------------------------------------------------
# Bot
# ------------------------------------------------------------------


class Bot:
    """Wires STT → Ollama LLM → GLaDOS TTS with wake-word audio loop.

    Pipeline:
      1. WakeWordListener streams mic, detects wake phrase ("submit"),
         captures the following utterance as a command.
      2. Command audio is transcribed by Whisper STT.
      3. Text is sent to Ollama (Qwen 3.6) for a response.
         If the text mentions file paths, those files are read first
         and their content injected as context.
      4. The LLM response is spoken back through GLaDOS TTS with
         sentence/comma pauses, routed to the ROG Strix Go headphones.
    """

    SENTENCE_ENDS = ".!?"
    COMMA = ","

    def __init__(self, config: Optional[cfg.Config] = None) -> None:
        self.cfg = config or cfg
        self._stt = STT(self.cfg)
        self._tts = TTS(self.cfg)
        self._llm = make_llm_client(self.cfg)
        self._wake: Optional[WakeWordListener] = None
        self._running = False
        self._last_tts_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load STT, TTS, and verify Ollama connectivity."""
        self._stt.load()
        self._tts.load()
        if not self._llm.ping():
            log.warning("LLM backend not reachable at %s — LLM responses will fail",
                        self._llm._host)
        else:
            log.info("LLM backend reachable, model=%s", self._llm.model_name())

    def start(self) -> None:
        """Start the wake-word listener (blocks on a background thread)."""
        if self._running:
            return
        self._wake = WakeWordListener(self.cfg, self._stt)
        self._wake.add_callback(self._on_wake_event)
        self._wake.start()
        self._running = True
        log.info("Bot started — listening for '%s'", self.cfg.WAKE_WORD["phrase"])

    def stop(self) -> None:
        """Stop the wake-word listener."""
        if self._wake is not None:
            self._wake.stop()
            self._wake = None
        self._running = False
        log.info("Bot stopped")

    # ------------------------------------------------------------------
    # Wake-word callbacks
    # ------------------------------------------------------------------

    def _on_wake_event(self, event: str, data: Optional[str]) -> None:
        if event == "command":
            self._handle_command(data)
        elif event == "wake_word":
            log.info("Wake word detected")
        elif event == "utt_ready":
            pass  # intermediate utterance — ignore
        elif event == "error":
            log.error("Wake listener error: %s", data)

    # ------------------------------------------------------------------
    # Command handling: STT text → LLM → TTS
    # ------------------------------------------------------------------

    def _handle_command(self, text: str) -> None:
        """Transcribe → LLM → speak response."""
        if not text or not text.strip():
            return

        log.info("command: %r", text[:120])

        # ---- File context injection ----
        file_context: Optional[str] = None
        paths = extract_paths(text, Path.cwd())
        if paths:
            parts: list[str] = []
            for p in paths:
                content = read_file(p, self.cfg)
                if content.startswith("[ERROR:"):
                    parts.append(f"[ERROR reading {p}: {content}]")
                else:
                    parts.append(f"--- {p} ---\n{content}")
            if parts:
                file_context = "\n\n".join(parts)
                log.info("injected %d file(s) as context", len(paths))

        # ---- LLM response ----
        response = self._llm.chat_with_context(text, file_context)
        if not response:
            response = "Sorry, I couldn't think of a response."

        log.info("response: %r", response[:120])

        # ---- Speak ----
        if self.cfg.LLM.get("use_pauses", True):
            self._say_with_pauses(response)
        else:
            self._say(response)

    # ------------------------------------------------------------------
    # TTS with pauses at sentence/comma boundaries
    # ------------------------------------------------------------------

    def _say(self, text: str) -> None:
        """Speak text with the GLaDOS voice, routed to the ROG Strix Go headphones.

        The wake-word listener is told to ignore mic input while speaking so
        commands don't queue up mid-sentence.
        """
        if not text:
            return
        if self._wake is not None:
            self._wake.set_playing_tts(True)
        try:
            wav = self._tts.synthesize(text)
            if wav is not None:
                self._tts.play(wav, self.cfg.AUDIO)
            else:
                log.warning("TTS synthesis returned None for: %r", text[:60])
        except Exception:
            log.exception("TTS playback failed for: %r", text[:60])
        finally:
            if self._wake is not None:
                self._wake.set_playing_tts(False)
        self._last_tts_path = None

    def _say_with_pauses(self, text: str) -> None:
        """Split text into sentence/comma chunks, synthesize each, concatenate with silence.

        Pauses:
          - Sentence ending (. ! ?) → ~0.35 s silence (full breath)
          - Comma (,)              → ~0.15 s silence (half breath)
        """
        # Clean text: remove emoji and symbols that confuse TTS
        clean = re.sub(r"[^\w\s.\,!?\-\'\"\,\u2014]", "", text).strip()
        if not clean:
            return

        # ---- Split into chunks ----
        # First: split on sentence endings
        sentence_parts = re.split(rf"(?<=[{self.SENTENCE_ENDS}])\s+", clean)
        chunks: list[str] = []
        for part in sentence_parts:
            part = part.strip()
            if not part:
                continue
            # Within each sentence, split on commas (comma stays with preceding text)
            sub = re.split(rf"(?<=[{self.COMMA}])\s+", part)
            chunks.extend([s.strip() for s in sub if s.strip()])

        if not chunks:
            return

        # Pause durations in seconds
        SENTENCE_PAUSE = 0.35
        COMMA_PAUSE = 0.15

        # ---- Synthesize each chunk ----
        wav_paths: list[Path] = []
        for chunk in chunks:
            wav = self._tts.synthesize(chunk)
            if wav is None:
                continue
            wav_paths.append(wav)

        if not wav_paths:
            return

        # ---- Concatenate WAVs with silence gaps ----
        # Detect if a chunk ends with a sentence-ending punctuation
        def ends_with_sentence(ch: str) -> bool:
            return ch.strip().endswith(self.SENTENCE_ENDS)

        import soundfile as sf

        # Use the first chunk's real sample rate as the target — no extra
        # probe synthesis needed, this file was going to be read anyway.
        _, sr = sf.read(str(wav_paths[0]), dtype="float32", always_2d=False)
        sr = int(sr)

        merged = np.array([], dtype=np.float32)
        for i, wp in enumerate(wav_paths):
            data, file_sr = sf.read(str(wp), dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]
            # Resample if sample rate differs
            if abs(file_sr - sr) > 1:
                from scipy.signal import resample

                new_len = int(len(data) * sr / file_sr)
                data = resample(data, new_len)
            merged = np.concatenate([merged, data])

            # Add silence gap after this chunk (except last)
            if i < len(wav_paths) - 1:
                # Determine pause length: sentence end = long, comma = short
                chunk_text = chunks[i]
                pause = SENTENCE_PAUSE if ends_with_sentence(chunk_text) else COMMA_PAUSE
                gap = np.zeros(int(sr * pause), dtype=np.float32)
                merged = np.concatenate([merged, gap])

        # ---- Write merged WAV ----
        import wave

        out_path = self._tts._cache_dir / f"paused_{abs(hash(text)) % 1000000}.wav"
        merged_i16 = np.clip(merged * 32767.0, -32768, 32767).astype(np.int16)
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(merged_i16.tobytes())

        # ---- Play the full concatenated audio ----
        self._tts.play(out_path, self.cfg.AUDIO)
        self._last_tts_path = out_path

    # ------------------------------------------------------------------
    # Text-to-speech queue (file-based IPC for external prompts)
    # ------------------------------------------------------------------

    _QUEUE_PATH = Path(__file__).resolve().parent / "tts_queue.txt"

    def speak(self, text: str) -> None:
        """Enqueue text to be spoken in GLaDOS voice.

        Writes `text` to the queue file. The main loop picks it up and
        speaks it via _say_with_pauses. Safe to call from another process.
        """
        if not text:
            return
        try:
            self._QUEUE_PATH.write_text(text.strip(), encoding="utf-8")
        except Exception:
            log.exception("failed to write TTS queue")

    def _drain_queue(self) -> Optional[str]:
        """Read and clear the TTS queue file. Returns the text, or None."""
        try:
            if not self._QUEUE_PATH.exists():
                return None
            text = self._QUEUE_PATH.read_text(encoding="utf-8").strip()
            if not text:
                return None
            self._QUEUE_PATH.write_text("", encoding="utf-8")
            return text
        except Exception:
            log.exception("failed to read/clear TTS queue")
            return None


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------


def main() -> None:
    from .audio_utils import setup_logging

    setup_logging("INFO")
    bot = Bot()
    bot.load()
    try:
        bot.start()
        log.info("Press Ctrl+C to stop")
        while True:
            import time

            time.sleep(0.5)
            # Drain any externally queued TTS text
            queued = bot._drain_queue()
            if queued:
                log.info("speaking queued text: %r", queued[:80])
                bot._say_with_pauses(queued)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        bot.stop()


if __name__ == "__main__":
    main()
