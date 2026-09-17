# GLaDOS control panel — a small desktop app wrapping Bot, so the voice
# bot can be started/stopped/watched without a terminal.
#
# Usage:
#   .\venv\Scripts\pythonw.exe -m voice_bot.gui
#
# Tkinter (and customtkinter) widgets may only be touched from the main
# thread. Bot's events fire from background threads (the wake-word audio
# callback thread, its transcription executor), so everything crossing that
# boundary goes through a queue.Queue drained on a periodic `after()` tick.

from __future__ import annotations

import os
import sys

# pythonw.exe has no console, so sys.stdout/sys.stderr are None. Libraries
# that write to them directly (loguru inside the TTS model, tqdm progress
# bars) raise AttributeError instead of no-op'ing, silently breaking TTS
# synthesis mid-call — patch them to a null sink before anything else
# imports and captures the None reference.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import queue
import threading
import tkinter as tk
from typing import Optional

import customtkinter as ctk

from . import config as cfg
from .audio_utils import setup_logging
from .bot import Bot, create_bot

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APERTURE_ORANGE = "#f2a900"
APERTURE_ORANGE_HOVER = "#c98a00"
APERTURE_BG = "#111113"
APERTURE_CARD = "#1b1b1e"
APERTURE_CARD_BORDER = "#2c2c30"
TEXT_DIM = "#8a8a8f"
GREEN = "#4caf50"
RED = "#e05555"
STATE_COLORS = {
    "Listening": GREEN,
    "Speaking": APERTURE_ORANGE,
    "Asleep": TEXT_DIM,
}


class GladosApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("GLaDOS Control Panel")
        self.geometry("520x760")
        self.minsize(520, 760)
        self.configure(fg_color=APERTURE_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.bot: Optional[Bot] = None
        self._events: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()

        self._build_ui()
        self._poll_events()

    # ------------------------------------------------------------------
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = ctk.CTkFrame(self, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=20, pady=20)

        # ---- Header ----
        ctk.CTkLabel(
            root, text="APERTURE SCIENCE", font=("Segoe UI", 12, "bold"),
            text_color=APERTURE_ORANGE,
        ).pack(anchor="w")
        ctk.CTkLabel(
            root, text="GLaDOS Control Panel", font=("Segoe UI", 24, "bold"),
        ).pack(anchor="w", pady=(2, 4))

        model_name = (
            cfg.GROQ["model"] if cfg.LLM_BACKEND == "groq" else cfg.OLLAMA["model"]
        )
        ctk.CTkLabel(
            root, text=f"{cfg.LLM_BACKEND} / {model_name}",
            font=("Segoe UI", 11), text_color=TEXT_DIM,
        ).pack(anchor="w")

        ctk.CTkFrame(root, height=1, fg_color=APERTURE_CARD_BORDER).pack(
            fill="x", pady=(14, 16))

        # ---- Status card ----
        status_card = ctk.CTkFrame(
            root, fg_color=APERTURE_CARD, corner_radius=14,
            border_width=1, border_color=APERTURE_CARD_BORDER,
        )
        status_card.pack(fill="x", pady=(0, 14))

        status_row = ctk.CTkFrame(status_card, fg_color="transparent")
        status_row.pack(fill="x", padx=18, pady=(16, 4))

        self.status_var = tk.StringVar(value="Stopped")
        self.status_dot = ctk.CTkLabel(
            status_row, text="●", font=("Segoe UI", 14), text_color=TEXT_DIM, width=16,
        )
        self.status_dot.pack(side="left")
        self.status_label = ctk.CTkLabel(
            status_row, textvariable=self.status_var, font=("Segoe UI", 17, "bold"),
            text_color=TEXT_DIM, anchor="w",
        )
        self.status_label.pack(side="left", padx=(4, 0))

        self.toggle_btn = ctk.CTkButton(
            status_card, text="▶  Start", command=self._on_toggle, height=46,
            corner_radius=10, font=("Segoe UI", 14, "bold"), fg_color=APERTURE_ORANGE,
            hover_color=APERTURE_ORANGE_HOVER, text_color="#1a1a1a",
        )
        self.toggle_btn.pack(fill="x", padx=18, pady=(8, 10))

        ctk.CTkLabel(status_card, text="MIC LEVEL", font=("Segoe UI", 10, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", padx=18)
        self.mic_bar = ctk.CTkProgressBar(
            status_card, progress_color=APERTURE_ORANGE, corner_radius=6, height=8,
        )
        self.mic_bar.set(0)
        self.mic_bar.pack(fill="x", padx=18, pady=(6, 18))

        # ---- Log card ----
        ctk.CTkLabel(root, text="LOG", font=("Segoe UI", 10, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", pady=(0, 6))
        self.log_box = ctk.CTkTextbox(
            root, fg_color=APERTURE_CARD, corner_radius=14,
            border_width=1, border_color=APERTURE_CARD_BORDER,
            font=("Consolas", 11), state="disabled", wrap="word",
        )
        self.log_box.pack(fill="both", expand=True, pady=(0, 14))

        # ---- Input row ----
        input_frame = ctk.CTkFrame(root, fg_color="transparent")
        input_frame.pack(fill="x")

        self.input_var = tk.StringVar()
        self.input_entry = ctk.CTkEntry(
            input_frame, textvariable=self.input_var, height=40, corner_radius=10,
            fg_color=APERTURE_CARD, border_color=APERTURE_CARD_BORDER,
            placeholder_text="Type a message...",
        )
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.input_entry.bind("<Return>", self._on_send_text)

        ctk.CTkButton(
            input_frame, text="Send", width=76, height=40, corner_radius=10,
            command=self._on_send_text, fg_color=APERTURE_ORANGE,
            hover_color=APERTURE_ORANGE_HOVER, text_color="#1a1a1a",
        ).pack(side="left")

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        if self.bot is None:
            self._start_bot()
        else:
            self._stop_bot()

    def _set_status(self, text: str, color: str) -> None:
        self.status_var.set(text)
        self.status_label.configure(text_color=color)
        self.status_dot.configure(text_color=color)

    def _start_bot(self) -> None:
        self.toggle_btn.configure(state="disabled", text="Loading...")
        self._set_status("Loading models...", TEXT_DIM)

        def worker() -> None:
            try:
                bot = create_bot()
                bot.add_gui_callback(lambda ev, data: self._events.put((ev, data)))
                bot.load()
                bot.start()
                self.bot = bot
                self._events.put(("ready", None))
            except Exception as e:
                self._events.put(("load_error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_bot(self) -> None:
        if self.bot is not None:
            self.bot.stop()
            self.bot = None
        self.toggle_btn.configure(state="normal", text="▶  Start")
        self._set_status("Stopped", TEXT_DIM)
        self.mic_bar.set(0)

    def _on_close(self) -> None:
        self._stop_bot()
        self.destroy()

    def _on_send_text(self, event=None) -> None:
        """Send typed text through the same path as a voice command.

        Note: unlike voice input, this doesn't check for the sleep/wake/
        interrupt phrases first (that matching lives in WakeWordListener) —
        typed "stop" or "goodnight" is just sent to the LLM as text.
        """
        if self.bot is None:
            return
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")

        def worker() -> None:
            self.bot._notify_gui("heard", text)
            self.bot._handle_command(text)

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _poll_events(self) -> None:
        try:
            while True:
                event, data = self._events.get_nowait()
                self._handle_event(event, data)
        except queue.Empty:
            pass

        if self.bot is not None and self.bot._wake is not None:
            energy = self.bot._wake.get_last_energy()
            threshold = cfg.WAKE_WORD["vad_energy_threshold"]
            # Scale so the bar reaches full at ~10x the trigger threshold —
            # keeps normal speech visibly filling the meter.
            self.mic_bar.set(min(1.0, energy / (threshold * 10)))

        self.after(100, self._poll_events)

    def _handle_event(self, event: str, data: Optional[str]) -> None:
        if event == "ready":
            self.toggle_btn.configure(state="normal", text="■  Stop")
            active = cfg.WAKE_WORD.get("start_active", True)
            self._set_status("Listening" if active else "Asleep",
                              GREEN if active else TEXT_DIM)
            self._append_log("[Bot started]")
        elif event == "load_error":
            self.toggle_btn.configure(state="normal", text="▶  Start")
            self._set_status("Failed to start", RED)
            self._append_log(f"[ERROR] {data}")
        elif event == "state":
            self._set_status(data or "", STATE_COLORS.get(data or "", TEXT_DIM))
        elif event == "heard":
            self._append_log(f"You: {data}")
        elif event == "response":
            self._append_log(f"GLaDOS: {data}")

    def _append_log(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")


def main() -> None:
    setup_logging("INFO")
    app = GladosApp()
    app.mainloop()


if __name__ == "__main__":
    main()
