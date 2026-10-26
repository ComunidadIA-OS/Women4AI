"""Genera tests/fixtures/clip_120s.wav: audio sintético (sin voces reales) de 120 s.

Tres "hablantes" simulados con tonos de distinta frecuencia fundamental y armónicos,
pausas, y algunos solapes, para depurar el pipeline sin GPU ni audio con derechos.
Determinista (semilla fija). Solo stdlib.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import wave
from pathlib import Path

SR = 16000

# (inicio s, fin s, f0 Hz): incluye solapes (p. ej. 30.0–31.2) y huecos de silencio.
SCRIPT = [
    (0.5, 14.0, 120), (14.6, 29.8, 210), (29.2, 31.2, 120), (31.5, 45.0, 120),
    (45.3, 46.0, 210), (46.5, 60.0, 180), (60.4, 75.0, 210), (74.2, 76.0, 120),
    (76.3, 90.0, 120), (90.5, 104.0, 180), (104.5, 119.5, 210),
]  # fmt: skip


def synth(duration_s: float = 120.0, seed: int = 13) -> list[int]:
    rng = random.Random(seed)
    n = int(duration_s * SR)
    buf = [0.0] * n
    for start, end, f0 in SCRIPT:
        i0, i1 = int(start * SR), min(int(end * SR), n)
        phase = 0.0
        for i in range(i0, i1):
            t = (i - i0) / SR
            f = f0 * (1 + 0.08 * math.sin(2 * math.pi * 3.0 * t))  # entonación
            phase += 2 * math.pi * f / SR
            syll = 0.5 * (1 + math.sin(2 * math.pi * 4.0 * t))  # ~4 sílabas/s
            env = min(1.0, (i - i0) / 800, (i1 - i) / 800)
            s = sum(math.sin(k * phase) / k for k in (1, 2, 3, 4))
            buf[i] += 0.18 * env * syll * s
    return [max(-32767, min(32767, int(32767 * (x + rng.gauss(0, 0.003))))) for x in buf]


def write_wav(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, nargs="?", default=Path("tests/fixtures/clip_120s.wav"))
    p.add_argument("--seconds", type=float, default=120.0)
    a = p.parse_args()
    write_wav(a.out, synth(a.seconds))
    print(f"Fixture escrito en {a.out}")


if __name__ == "__main__":
    main()
