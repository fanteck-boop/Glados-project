# ------------------------------------------------------------------
# Voice Bot — GLaDOS voice, Ollama Qwen, Whisper STT, wake-word loop
# ------------------------------------------------------------------

from __future__ import annotations

import importlib.util
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, Optional

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
    """Wires STT → LLM → GLaDOS TTS with an always-listening audio loop.

    Pipeline:
      1. WakeWordListener streams the mic and transcribes each utterance.
         While active, every utterance is treated as a command directly —
         no need to say the wake phrase first. Saying the sleep phrase
         (config.WAKE_WORD["sleep_phrase"]) pauses her; saying the wake
         phrase (config.WAKE_WORD["phrase"]) resumes her.
      2. Command audio is transcribed by Whisper STT.
      3. Text is sent to the configured LLM backend for a response.
         If the text mentions file paths, those files are read first
         and their content injected as context.
      4. The LLM response is spoken back through GLaDOS TTS with
         sentence/comma pauses, routed to the ROG Strix Go headphones.

    A subclass can customize command handling without editing this file —
    see _maybe_handle_meta_command, _maybe_intercept,
    _maybe_transform_command_text, and _after_response below, and
    create_bot() at the bottom of this module.
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
        # Held around every Whisper transcribe() and GLaDOS synthesize() call
        # (here and in WakeWordListener) so STT and TTS never touch the CUDA
        # context concurrently — that race was causing intermittent native
        # crashes/hangs after a few exchanges.
        self._gpu_lock = threading.Lock()
        # Set by the interrupt phrase to cut a response short between chunks.
        self._interrupted = threading.Event()
        # Held for the entire duration of _say/_say_with_pauses so two speech
        # requests from different sources (a voice command, the wake/sleep
        # greetings, queued external text) can never play over each other —
        # playback itself isn't under _gpu_lock (that would block the
        # interrupt phrase from being heard while she talks), so without this
        # nothing else stops two of them overlapping.
        self._speaking_lock = threading.Lock()
        # Optional external listeners (e.g. the GUI) notified of high-level
        # events: ("state", "Listening"|"Asleep"|"Speaking"), ("heard", text),
        # ("response", text). Separate from WakeWordListener's own callback
        # list since those are lower-level (wake_word/sleep/command/error).
        self._gui_callbacks: list[Callable[[str, Optional[str]], None]] = []

    def add_gui_callback(self, cb: Callable[[str, Optional[str]], None]) -> None:
        """Register a callback for high-level UI events. See _gui_callbacks."""
        self._gui_callbacks.append(cb)

    def _notify_gui(self, event: str, data: Optional[str] = None) -> None:
        for cb in list(self._gui_callbacks):
            try:
                cb(event, data)
            except Exception:
                log.exception("gui callback error")

    def _revert_gui_state_after_speech(self) -> None:
        if self._wake is not None:
            self._notify_gui("state", "Listening" if self._wake.is_active() else "Asleep")

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
        self._wake = WakeWordListener(self.cfg, self._stt, self._gpu_lock)
        self._wake.add_callback(self._on_wake_event)
        self._wake.start()
        self._running = True
        if self.cfg.WAKE_WORD.get("start_active", True):
            log.info("Bot started — actively listening (say '%s' to pause, '%s' to resume)",
                      self.cfg.WAKE_WORD.get("sleep_phrase", ""), self.cfg.WAKE_WORD["phrase"])
        else:
            log.info("Bot started — asleep, say '%s' to wake her", self.cfg.WAKE_WORD["phrase"])

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
            self._notify_gui("heard", data)
            self._handle_command(data)
        elif event == "wake_word":
            log.info("Wake word detected")
            self._notify_gui("state", "Listening")
            self._say("Hello, master.")
        elif event == "sleep":
            log.info("Sleep phrase detected")
            self._notify_gui("state", "Asleep")
            self._say("Goodnight.")
        elif event == "interrupt":
            log.info("Interrupt phrase detected — stopping playback")
            self._interrupted.set()
            self._tts.stop()
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

        stripped = text.strip()
        if self._maybe_handle_meta_command(stripped):
            return

        log.info("command: %r", text[:120])

        if self._maybe_intercept(text):
            return

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

        command_text = self._maybe_transform_command_text(text)

        # ---- LLM response ----
        response = self._llm.chat_with_context(command_text, file_context)
        if command_text != text:
            # Scrub any one-off transform from history so it affects this
            # reply only, not every later turn until it ages out.
            clean_msg = text
            if file_context:
                clean_msg = f"[CONTEXT FROM FILES]\n{file_context}\n\n[END CONTEXT]\n\n{text}"
            self._llm.replace_last_user_message(clean_msg)
        if not response:
            response = "Sorry, I couldn't think of a response."

        log.info("response: %r", response[:120])
        self._notify_gui("response", response)

        # ---- Speak ----
        if self.cfg.LLM.get("use_pauses", True):
            self._say_with_pauses(response)
        else:
            self._say(response)

        self._after_response(command_text, text, response)

    # ------------------------------------------------------------------
    # Extension hooks — no-ops here. A subclass (see create_bot()) can
    # override these to customize command handling without touching this
    # file. Kept deliberately generic: this file has no idea what, if
    # anything, uses them.
    # ------------------------------------------------------------------

    def _maybe_handle_meta_command(self, text: str) -> bool:
        """Handle a non-conversational command (e.g. a slash command).
        Return True to skip the normal LLM/TTS flow entirely."""
        return False

    def _maybe_intercept(self, text: str) -> bool:
        """Short-circuit command handling before the LLM is involved.
        Return True to stop processing this utterance."""
        return False

    def _maybe_transform_command_text(self, text: str) -> str:
        """Return the text to actually send to the LLM for this turn.
        Default: unchanged. The original `text` is still what's kept in
        conversation history — see the scrub logic in _handle_command."""
        return text

    def _after_response(self, command_text: str, text: str, response: str) -> None:
        """Called after a response has been spoken. Default: no-op."""
        pass

    # ------------------------------------------------------------------
    # TTS with pauses at sentence/comma boundaries
    # ------------------------------------------------------------------

    def _say(self, text: str) -> None:
        """Speak text with the GLaDOS voice, routed to the ROG Strix Go headphones.

        The wake-word listener keeps listening for the interrupt phrase while
        this plays — see set_playing_tts and WakeWordListener's gating.
        """
        if not text:
            return
        with self._speaking_lock:
            if self._wake is not None:
                self._wake.set_playing_tts(True)
            self._notify_gui("state", "Speaking")
            self._interrupted.clear()
            try:
                with self._gpu_lock:
                    wav = self._tts.synthesize(text)
                if wav is None:
                    log.warning("TTS synthesis returned None for: %r", text[:60])
                elif not self._interrupted.is_set():
                    self._tts.play(wav, self.cfg.AUDIO, self._interrupted)
            except Exception:
                log.exception("TTS playback failed for: %r", text[:60])
            finally:
                if self._wake is not None:
                    self._wake.set_playing_tts(False)
                self._revert_gui_state_after_speech()
            self._last_tts_path = None

    def _say_with_pauses(self, text: str) -> None:
        """Split text into sentence/comma chunks and speak them one at a time.

        Each chunk is synthesized up front (under _gpu_lock, since that's the
        GPU-bound part) but played individually, with a real pause between
        chunks — not concatenated into one file. That gap between plays is
        what lets the interrupt phrase actually cut a response short instead
        of only being noticed after the whole thing finished.

        Pauses:
          - Sentence ending (. ! ?) → ~0.35 s silence (full breath)
          - Comma (,)              → ~0.15 s silence (half breath)
        """
        # Clean text: remove emoji and symbols that confuse TTS
        clean = re.sub(r"[^\w\s.\,!?\-\'\"\,—]", "", text).strip()
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

        def ends_with_sentence(ch: str) -> bool:
            return ch.strip().endswith(self.SENTENCE_ENDS)

        with self._speaking_lock:
            if self._wake is not None:
                self._wake.set_playing_tts(True)
            self._notify_gui("state", "Speaking")
            self._interrupted.clear()
            try:
                with self._gpu_lock:
                    wav_paths: list[Path] = []
                    for chunk in chunks:
                        wav = self._tts.synthesize(chunk)
                        if wav is not None:
                            wav_paths.append(wav)

                if not wav_paths:
                    return

                for i, wp in enumerate(wav_paths):
                    if self._interrupted.is_set():
                        log.info("playback interrupted")
                        break
                    try:
                        self._tts.play(wp, self.cfg.AUDIO, self._interrupted)
                    except Exception:
                        log.exception("TTS playback failed for chunk: %r", chunks[i][:60])
                        break
                    self._last_tts_path = wp
                    if self._interrupted.is_set():
                        log.info("playback interrupted")
                        break
                    if i < len(wav_paths) - 1:
                        pause = SENTENCE_PAUSE if ends_with_sentence(chunks[i]) else COMMA_PAUSE
                        time.sleep(pause)
            finally:
                if self._wake is not None:
                    self._wake.set_playing_tts(False)
                self._revert_gui_state_after_speech()

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
# Bot factory — loads a local customization module if present
# ------------------------------------------------------------------

def create_bot(config: Optional[cfg.Config] = None) -> Bot:
    """Instantiate the bot.

    If a sibling folder next to the project root contains a local
    customization module, use the Bot subclass it defines instead of the
    plain class here — lets a particular machine's own setup extend
    behavior (see the Bot docstring for the extension points) without any
    of that code living in this repo. A clone without that folder just
    gets the plain Bot.
    """
    ext_path = Path(__file__).resolve().parents[2] / "Glados_private" / "extension.py"
    if ext_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("glados_local_extension", ext_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.CustomBot(config)
        except Exception:
            log.exception("failed to load local customization module, using plain Bot")
    return Bot(config)


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------


def main() -> None:
    from .audio_utils import setup_logging

    setup_logging("INFO")
    bot = create_bot()
    bot.load()
    try:
        bot.start()
        log.info("Press Ctrl+C to stop")
        while True:
            time.sleep(0.5)
            # Drain any externally queued TTS text
            queued = bot._drain_queue()
            if queued:
                log.info("speaking queued text: %r", queued[:80])
                try:
                    bot._say_with_pauses(queued)
                except Exception:
                    log.exception("failed to speak queued text")
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        bot.stop()


if __name__ == "__main__":
    main()
