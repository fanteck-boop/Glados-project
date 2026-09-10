#!/usr/bin/env python3
"""
send_to_alba.py — GLaDOS speaks, saves MP3, sends to Alba on WhatsApp.

Usage:
    python send_to_alba.py "How is work and when will we go to gym"
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
VOICEBOT_DIR = PROJECT_DIR.parent / "voice_bot"
CACHE_DIR = PROJECT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

SPEAK_PY = PROJECT_DIR / "speak.py"
PYTHON = VOICEBOT_DIR / "venv" / "Scripts" / "python.exe"
FFMPEG = "ffmpeg"
WHATSAPP_CONTACT = "Alba"


def run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"[cmd] {cmd_str}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:400]
        print(f"[ERROR] exit {result.returncode}: {err}")
        raise RuntimeError(f"Command failed: {err}")
    return result


# ---------- Step 1: generate WAV via GLaDOS ----------
def generate_wav(text: str) -> Path:
    print(f"\n{'='*60}")
    print(f"[1/4] GLaDOS synthesis")
    print(f"       text: {text}")
    print(f"{'='*60}")
    run([str(PYTHON), str(SPEAK_PY), text], timeout=300)
    wavs = sorted(CACHE_DIR.glob("_probe*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        raise RuntimeError("No _probe WAV found in cache")
    wav = wavs[0]
    print(f"[1/4] OK  WAV: {wav.name}")
    return wav


# ---------- Step 2: save as MP3 ----------
def save_mp3(wav_path: Path) -> Path:
    print(f"\n[2/4] Saving MP3 to cache ...")
    mp3 = CACHE_DIR / f"{wav_path.stem}.mp3"
    run([
        FFMPEG, "-y",
        "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-q:a", "2",
        str(mp3),
    ], timeout=60)
    kb = mp3.stat().st_size / 1024
    print(f"[2/4] OK  MP3: {mp3.name}  ({kb:.1f} KB)")
    return mp3


# ---------- Step 3: convert to Opus for WhatsApp ----------
def convert_opus(wav_path: Path) -> Path:
    print(f"\n[3/4] Converting to Opus for WhatsApp ...")
    ts = int(time.time())
    ogg = CACHE_DIR / f"{wav_path.stem}_{ts}.ogg"
    run([
        FFMPEG, "-y",
        "-i", str(wav_path),
        "-c:a", "libopus",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "24k",
        str(ogg),
    ], timeout=60)
    kb = ogg.stat().st_size / 1024
    print(f"[3/4] OK  Opus: {ogg.name}  ({kb:.1f} KB)")
    return ogg


# ---------- Step 4: send via WhatsApp Desktop ----------
def send_whatsapp(opus_path: Path) -> bool:
    print(f"\n[4/4] Sending to {WHATSAPP_CONTACT} on WhatsApp Desktop ...")

    import pyautogui
    import pygetwindow as gw

    pyautogui.FAILSAFE = True

    wins = gw.getWindowsWithTitle("WhatsApp")
    if not wins:
        wins = [w for w in gw.getAllWindows() if w.visible and "whatsapp" in w.title.lower()]
    if not wins:
        print("[ERROR] WhatsApp Desktop window not found.")
        return False

    wa = wins[0]
    wa.restore()
    wa.activate()
    time.sleep(0.4)
    wa.moveTo(100, 60)
    wa.resizeTo(760, 960)
    time.sleep(0.3)

    # Search contact
    cx = wa.left + wa.width // 2
    cy = wa.top + int(wa.height * 0.06)
    pyautogui.click(cx, cy, duration=0.2)
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "k")
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "a")
    pyautogui.press("delete")
    time.sleep(0.15)
    pyautogui.write(WHATSAPP_CONTACT, interval=0.04)
    time.sleep(1.5)
    pyautogui.press("enter")
    time.sleep(2.0)

    # Attachment button
    ax = wa.left + wa.width - 48
    ay = wa.top + int(wa.height * 0.10)
    pyautogui.click(ax, ay, duration=0.2)
    time.sleep(0.6)

    # Document
    dx = wa.left + wa.width // 2
    dy = wa.top + int(wa.height * 0.22)
    pyautogui.click(dx, dy, duration=0.2)
    time.sleep(0.8)

    # File path
    pyautogui.write(str(opus_path), interval=0.012)
    time.sleep(0.4)
    pyautogui.press("enter")
    time.sleep(4.0)

    # Send button
    sx = wa.left + wa.width - 40
    sy = wa.top + wa.height - 48
    pyautogui.click(sx, sy, duration=0.2)
    time.sleep(1.5)

    pyautogui.screenshot(str(CACHE_DIR / f"wa_sent_{int(time.time())}.png"))
    print("[4/4] sent.")
    return True


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(sys.argv[0]).name} \"text to send\"")
        print(f"Example: python {Path(sys.argv[0]).name} \"How is work and when will we go to gym\"")
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    try:
        wav = generate_wav(text)
        mp3 = save_mp3(wav)
        opus = convert_opus(wav)
        ok = send_whatsapp(opus)

        print(f"\n{'='*60}")
        print(f"✓ MP3 saved : {mp3}")
        print(f"{'✓' if ok else '✗'} WhatsApp  : {WHATSAPP_CONTACT}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
