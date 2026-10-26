"""Lectura eficiente del WAV normalizado (16 kHz mono PCM16) por ventanas.

Nunca carga el episodio entero: lee solo las muestras de la ventana pedida.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .schemas import Segment


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def read_window(path: Path, start_s: float, end_s: float) -> tuple[np.ndarray, int]:
    """Devuelve ``(muestras float32 en [-1, 1], sample_rate)`` de ``[start_s, end_s)``."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path} no es PCM16 mono; pasa antes por ingest")
        sr = w.getframerate()
        i0 = max(0, round(start_s * sr))
        i1 = min(w.getnframes(), round(end_s * sr))
        w.setpos(min(i0, w.getnframes()))
        raw = w.readframes(max(0, i1 - i0))
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, sr


def mute_outside(
    samples: np.ndarray, sr: int, offset_s: float, speech: list[Segment], pad_s: float = 0.2
) -> np.ndarray:
    """Pone a cero lo que no es voz según el VAD (con ``pad_s`` de holgura).

    Evita que el ASR alucine texto en silencios, música o ruido, sin mover los timestamps.
    """
    keep = np.zeros(len(samples), dtype=bool)
    for s in speech:
        a = int((s.start - pad_s - offset_s) * sr)
        b = int((s.end + pad_s - offset_s) * sr)
        if b <= 0 or a >= len(samples):
            continue
        keep[max(0, a) : min(len(samples), b)] = True
    out = samples.copy()
    out[~keep] = 0.0
    return out
