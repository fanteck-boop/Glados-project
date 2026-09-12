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

import queue
import threading
import tkinter as tk
from typing import Optional

import customtkinter as ctk

from . import config as cfg
from .audio_utils import setup_logging
from .bot import Bot

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APERTURE_ORANGE = "#f2a900"
APERTURE_BG = "#1a1a1a"
APERTURE_PANEL = "#242424"


class GladosApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("GLaDOS Control Panel")
        self.geometry("480x560")
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
        ctk.CTkLabel(
            self, text="APERTURE SCIENCE", font=("Segoe UI", 14, "bold"),
            text_color=APERTURE_ORANGE,
        ).pack(pady=(18, 0))
        ctk.CTkLabel(
            self, text="GLaDOS Control Panel", font=("Segoe UI", 20, "bold"),
        ).pack(pady=(0, 12))

        model_name = (
            cfg.GROQ["model"] if cfg.LLM_BACKEND == "groq" else cfg.OLLAMA["model"]
        )
        ctk.CTkLabel(
            self, text=f"{cfg.LLM_BACKEND} / {model_name}",
            font=("Segoe UI", 11), text_color="#999999",
        ).pack(pady=(0, 16))

        self.status_var = tk.StringVar(value="Stopped")
        self.status_label = ctk.CTkLabel(
            self, textvariable=self.status_var, font=("Segoe UI", 16, "bold"),
            text_color="#888888",
        )
        self.status_label.pack(pady=(0, 16))

        self.toggle_btn = ctk.CTkButton(
            self, text="Start", command=self._on_toggle, width=220, height=44,
            font=("Segoe UI", 14, "bold"), fg_color=APERTURE_ORANGE,
            hover_color="#c98a00", text_color="#1a1a1a",
        )
        self.toggle_btn.pack(pady=(0, 24))

        ctk.CTkLabel(self, text="MIC LEVEL", font=("Segoe UI", 10, "bold"),
                     text_color="#999999").pack()
        self.mic_bar = ctk.CTkProgressBar(self, width=400, progress_color=APERTURE_ORANGE)
        self.mic_bar.set(0)
        self.mic_bar.pack(pady=(4, 20))

        ctk.CTkLabel(self, text="LOG", font=("Segoe UI", 10, "bold"),
                     text_color="#999999").pack()
        self.log_box = ctk.CTkTextbox(
            self, width=440, height=260, fg_color=APERTURE_PANEL,
            font=("Consolas", 11), state="disabled", wrap="word",
        )
        self.log_box.pack(pady=(4, 16), padx=16)

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def _on_toggle(self) -> None:
        if self.bot is None:
            self._start_bot()
        else:
            self._stop_bot()

    def _start_bot(self) -> None:
        self.toggle_btn.configure(state="disabled", text="Loading...")
        self.status_var.set("Loading models...")
        self.status_label.configure(text_color="#888888")

        def worker() -> None:
            try:
                bot = Bot()
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
        self.toggle_btn.configure(state="normal", text="Start")
        self.status_var.set("Stopped")
        self.status_label.configure(text_color="#888888")
        self.mic_bar.set(0)

    def _on_close(self) -> None:
        self._stop_bot()
        self.destroy()

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
            self.toggle_btn.configure(state="normal", text="Stop")
            active = cfg.WAKE_WORD.get("start_active", True)
            self.status_var.set("Listening" if active else "Asleep")
            self.status_label.configure(text_color="#4caf50" if active else "#888888")
            self._append_log("[Bot started]")
        elif event == "load_error":
            self.toggle_btn.configure(state="normal", text="Start")
            self.status_var.set("Failed to start")
            self.status_label.configure(text_color="#e05555")
            self._append_log(f"[ERROR] {data}")
        elif event == "state":
            self.status_var.set(data or "")
            color = {
                "Listening": "#4caf50",
                "Speaking": APERTURE_ORANGE,
                "Asleep": "#888888",
            }.get(data or "", "#888888")
            self.status_label.configure(text_color=color)
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
