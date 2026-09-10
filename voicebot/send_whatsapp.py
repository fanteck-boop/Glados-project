#!/usr/bin/env python3
"""
send_whatsapp.py — GLaDOS audio → WhatsApp Desktop (local app).

Usage:
    python send_whatsapp.py "text to speak"

Steps:
1. Run speak.py to generate GLaDOS WAV
2. Convert WAV → Opus/OGG (16kHz mono, WhatsApp-friendly)
3. Automate WhatsApp Desktop: focus it, open Alba's chat,
   attach the audio file, send.

Note: WhatsApp Desktop voice messages are recorded from the mic.
What we send is an audio file attachment (playable inline).
If WhatsApp treats Opus as a voice message, great — otherwise
it arrives as a regular audio file. Either way GLaDOS speaks.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
VOICEBOT_DIR = PROJECT_DIR.parent / "voice_bot"
CACHE_DIR = PROJECT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

SPEAK_PY = PROJECT_DIR / "speak.py"
PYTHON = VOICEBOT_DIR / "venv" / "Scripts" / "python.exe"
FFMPEG = "ffmpeg"

# WhatsApp contact name — change to whoever you want to send to
WHATSAPP_CONTACT = "Alba"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"[cmd] {cmd_str}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "")[:300]
        print(f"[ERROR] exit {result.returncode}: {err}")
        raise RuntimeError(f"Command failed: {err}")
    return result


# ---------------------------------------------------------------------------
# Step 1 — generate GLaDOS WAV
# ---------------------------------------------------------------------------

def generate_audio(text: str) -> Path:
    print(f"\n{'='*60}")
    print(f"[1/4] Generating GLaDOS audio")
    print(f"       text: {text}")
    print(f"{'='*60}")
    run([str(PYTHON), str(SPEAK_PY), text], timeout=300)

    wavs = sorted(CACHE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not wavs:
        raise RuntimeError("No WAV file found in cache after speak.py")
    path = wavs[0]
    print(f"[1/4] OK  WAV: {path.name}")
    return path


# ---------------------------------------------------------------------------
# Step 2 — convert WAV → Opus/OGG
# ---------------------------------------------------------------------------

def convert_to_opus(wav_path: Path) -> Path:
    print(f"\n[2/4] Converting to Opus (16 kHz mono, WhatsApp-friendly)...")
    out = CACHE_DIR / f"{wav_path.stem}_{int(time.time())}.ogg"
    run([
        FFMPEG, "-y",
        "-i", str(wav_path),
        "-c:a", "libopus",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", "24k",
        str(out),
    ], timeout=60)
    kb = out.stat().st_size / 1024
    print(f"[2/4] OK  Opus: {out.name}  ({kb:.1f} KB)")
    return out


# ---------------------------------------------------------------------------
# Step 3 — automate WhatsApp Desktop
# ---------------------------------------------------------------------------

def send_via_whatsapp(opus_path: Path) -> bool:
    print(f"\n[3/4] Automating WhatsApp Desktop ...")
    print(f"       contact : {WHATSAPP_CONTACT}")
    print(f"       file    : {opus_path.name}")

    import pyautogui
    import pygetwindow as gw

    pyautogui.FAILSAFE = True  # slam mouse to top-left corner to abort

    # ---- locate WhatsApp window ----
    wins = gw.getWindowsWithTitle("WhatsApp")
    if not wins:
        all_wins = gw.getAllWindows()
        wins = [w for w in all_wins if w.visible and "whatsapp" in w.title.lower()]
    if not wins:
        print("[ERROR] WhatsApp Desktop window not found.")
        print("        Make sure it is running and not minimised.")
        return False

    wa = wins[0]
    wa.restore()
    time.sleep(0.25)
    wa.activate()
    time.sleep(0.5)

    wx, wy, ww, wh = wa.left, wa.top, wa.width, wa.height
    print(f"       window : ({wx}, {wy})  {ww} x {wh}")

    # ---- 3a — search for the contact ----
    print(f"[3/4a] Searching for '{WHATSAPP_CONTACT}' ...")
    pyautogui.click(wx + ww // 2, wy + int(wh * 0.07), duration=0.2)
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "k")
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "a")
    pyautogui.press("delete")
    time.sleep(0.15)
    pyautogui.write(WHATSAPP_CONTACT, interval=0.04)
    time.sleep(1.0)
    pyautogui.press("enter")
    time.sleep(1.5)

    _screenshot("wa_chat")

    # ---- 3b — click attachment (paperclip / plus) ----
    print("[3/4b] Clicking attachment button ...")
    # paperclip lives in the top toolbar above the message input
    px = wx + ww - 45
    py = wy + int(wh * 0.10)
    pyautogui.click(px, py, duration=0.25)
    time.sleep(0.7)

    _screenshot("wa_attach_menu")

    # ---- 3c — choose "Document" in the menu ----
    print("[3/4c] Choosing Document ...")
    # Document is usually the 2nd/3rd item; pick centre-ish below the menu header
    dx = wx + ww // 2
    dy = wy + int(wh * 0.22)
    pyautogui.click(dx, dy, duration=0.25)
    time.sleep(1.0)

    # ---- 3d — file picker: type full path, press Enter ----
    print(f"[3/4d] Picking file: {opus_path.name}")
    pyautogui.write(str(opus_path), interval=0.015)
    time.sleep(0.4)
    pyautogui.press("enter")
    time.sleep(3.0)  # let WhatsApp load the file

    _screenshot("wa_file_picked")

    # ---- 3e — send ----
    print("[3/4e] Sending ...")
    sx = wx + ww - 40
    sy = wy + wh - 45
    pyautogui.click(sx, sy, duration=0.25)
    time.sleep(1.5)

    print("[3/4] done.")
    return True


def _screenshot(tag: str) -> None:
    try:
        import pyautogui
        p = CACHE_DIR / f"{tag}_{int(time.time())}.png"
        pyautogui.screenshot(str(p))
        print(f"       screenshot: {p.name}")
    except Exception as e:
        print(f"       (screenshot skipped: {e})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(sys.argv[0]).name} \"text to speak\"")
        print(f"Example: python {Path(sys.argv[0]).name} \"I love you Alba\"")
        print()
        print(f"Config:")
        print(f"  contact      : {WHATSAPP_CONTACT}")
        print(f"  speak.py     : {SPEAK_PY}")
        print(f"  cache dir    : {CACHE_DIR}")
        print(f"  ffmpeg       : {FFMPEG}")
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    try:
        wav = generate_audio(text)
        opus = convert_to_opus(wav)
        ok = send_via_whatsapp(opus)

        print(f"\n{'='*60}")
        if ok:
            print(f"✓ Sent voice message to {WHATSAPP_CONTACT}")
            print(f"  file : {opus}")
        else:
            print(f"✗ Automation failed — send manually")
            print(f"  file : {opus}")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        print(f"\n[TIP] Send {opus_path_if_exists()} manually to {WHATSAPP_CONTACT}")
        print(f"{'='*60}")
        sys.exit(1)


def opus_path_if_exists() -> str:
    oggs = list(CACHE_DIR.glob("*.ogg"))
    if not oggs:
        return "<no opus file found>"
    return str(max(oggs, key=lambda p: p.stat().st_mtime))


if __name__ == "__main__":
    main()
