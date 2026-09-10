# ------------------------------------------------------------------
# glados_voice_server.py — standalone GLaDOS TTS HTTP server
#
# Listens on 127.0.0.1:8765, accepts POST /speak with JSON:
#   {"text": "something to say"}
# Synthesizes with GLaDOS (Style-Bert-VITS2), plays to the ROG
# Strix Go headphones, logs every step.
#
# Usage (from the project folder):
#   .\venv\Scripts\python.exe glados_voice_server.py
# ------------------------------------------------------------------
from __future__ import annotations

import logging
import queue
import threading

import sounddevice as sd
from flask import Flask, jsonify, request

import config as cfg
from tts import TTS

HOST = "127.0.0.1"
PORT = 8765
HEADSET_NAME = "Headphones (ROG Strix Go)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("glados_voice")

app = Flask(__name__)
speech_queue: queue.Queue[str] = queue.Queue()


def find_output_device(name: str) -> int:
    """Return the sounddevice index of the device whose name contains `name`."""
    target = name.casefold()
    for index, device in enumerate(sd.query_devices()):
        if device["max_output_channels"] > 0 and target in device["name"].casefold():
            return index
    raise RuntimeError(f"Could not find output device: {name!r}")


def speak_loop(tts: TTS, audio_cfg: dict) -> None:
    """Background thread: pull text from queue, synthesize, play."""
    while True:
        text = speech_queue.get()
        try:
            log.info("Speaking: %s", text[:160])
            wav_path = tts.synthesize(text)
            if wav_path is None:
                log.error("Synthesis returned None for: %s", text[:80])
            else:
                tts.play(wav_path, audio_cfg)
                log.info("Playback completed: %s", wav_path)
        except Exception:
            log.exception("Speech request failed.")
        finally:
            speech_queue.task_done()


@app.post("/speak")
def speak() -> None:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if not text:
        return jsonify({"ok": False, "error": "Missing 'text'."}), 400

    if len(text) > 1_500:
        return jsonify({"ok": False, "error": "Text exceeds 1,500 characters."}), 400

    speech_queue.put(text)

    return jsonify(
        {
            "ok": True,
            "queued": True,
            "queue_size": speech_queue.qsize(),
        }
    )


if __name__ == "__main__":
    device_index = find_output_device(HEADSET_NAME)

    audio_cfg = dict(cfg.AUDIO)
    audio_cfg["output_device"] = device_index

    log.info(
        "Using output device %d: %s",
        device_index,
        sd.query_devices(device_index)["name"],
    )

    tts = TTS(cfg)
    tts.load()

    worker = threading.Thread(
        target=speak_loop,
        args=(tts, audio_cfg),
        daemon=True,
        name="glados-speech-worker",
    )
    worker.start()

    log.info("GLaDOS voice server ready at http://%s:%d", HOST, PORT)
    app.run(host=HOST, port=PORT, threaded=True)