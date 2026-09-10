# Core utilities: audio helpers, whisper wrappers, ollama client,
# xtts wrapper, file reader, and the main bot loop.

from __future__ import annotations

import asyncio
import json
import logging
import queue
import re
import sys
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Relative imports (package mode) or absolute (standalone mode)
try:
    from . import config as cfg
except ImportError:
    import config as cfg  # type: ignore[no-redef]

logger = logging.getLogger("voice_bot")

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Little helpers
# ---------------------------------------------------------------------------

def slugify(s: str) -> str:
    return re.sub(r"[^\w\s-]", "", s).strip().lower().replace(" ", "-")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Audio helpers (numpy-based)
# ---------------------------------------------------------------------------

def audio_to_float32(data: Any) -> np.ndarray:
    """Convert bytes or int16 array to float32 in range [-1, 1]."""
    if isinstance(data, np.ndarray):
        arr = data
    elif isinstance(data, (bytes, bytearray)):
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    elif isinstance(data, list):
        arr = np.array(data, dtype=np.float32)
    else:
        arr = np.array(data, dtype=np.float32)

    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    elif arr.dtype.kind in ("u", "i"):
        arr = arr.astype(np.float32)
        info = np.iinfo(arr.dtype)
        if info.min < 0:
            arr /= float(info.max - info.min)
        else:
            arr /= float(info.max)
    return arr.astype(np.float32)


def float32_to_int16(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16)


def compute_energy(samples: np.ndarray) -> float:
    """RMS energy in [0, 1] for float32 audio in [-1, 1]."""
    if len(samples) == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples**2)))


def is_silence(samples: np.ndarray, threshold: float) -> bool:
    return compute_energy(samples) < threshold


# ---------------------------------------------------------------------------
# WAV file I/O
# ---------------------------------------------------------------------------

def read_wav(path: Path) -> tuple[int, np.ndarray]:
    """Read a mono/stereo WAV file, return (sample_rate, float32 mono)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)
    return sr, data


def write_wav(path: Path, sr: int, samples: np.ndarray) -> None:
    """Write float32 audio to a mono WAV file."""
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    int16 = (samples * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(int16.tobytes())
