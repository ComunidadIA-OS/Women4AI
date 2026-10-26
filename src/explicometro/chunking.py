"""Troceado de episodios largos y fusión sin duplicados (funciones puras).

Cada trozo tiene un **núcleo** ``[core_start, core_end)`` que no se solapa con el de los
demás y una **ventana** = núcleo ± margen que es lo que ve el modelo. De cada trozo solo se
conserva lo que cae en su núcleo (por el punto medio), así cada palabra o segmento entra una
sola vez aunque el modelo la haya visto en dos ventanas.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from dataclasses import dataclass

from .schemas import Segment, Word


@dataclass(frozen=True)
class Chunk:
    index: int
    core_start: float
    core_end: float
    start: float  # ventana (con margen)
    end: float

    def owns(self, t: float) -> bool:
        return self.core_start <= t < self.core_end


def _silence_midpoints(speech: list[Segment]) -> list[float]:
    ordered = sorted(speech, key=lambda s: s.start)
    return [(a.end + b.start) / 2 for a, b in itertools.pairwise(ordered) if b.start > a.end]


def plan_chunks(
    duration: float, chunk_s: float = 600.0, margin_s: float = 5.0, speech: list[Segment] | None = None
) -> list[Chunk]:
    """Trozos de ~``chunk_s`` con ``margin_s`` a cada lado.

    Si hay VAD, cada frontera se desplaza al centro del silencio más cercano dentro de
    ±``margin_s`` para no cortar palabras. Un episodio corto es un único trozo.
    """
    if duration <= chunk_s + margin_s:
        return [Chunk(0, 0.0, duration, 0.0, duration)]
    gaps = _silence_midpoints(speech or [])
    bounds = [0.0]
    t = chunk_s
    while t < duration - margin_s:
        near = [g for g in gaps if abs(g - t) <= margin_s and g > bounds[-1]]
        bounds.append(min(near, key=lambda g: abs(g - t)) if near else t)
        t = bounds[-1] + chunk_s
    bounds.append(duration)
    return [
        Chunk(i, a, b, max(0.0, a - margin_s), min(duration, b + margin_s))
        for i, (a, b) in enumerate(itertools.pairwise(bounds))
    ]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w']", "", text)


def merge_words(per_chunk: list[tuple[Chunk, list[Word]]], dedup_overlap: float = 0.5) -> list[Word]:
    """Une palabras de varios trozos (timestamps ya absolutos).

    1. Cada trozo aporta solo las palabras cuyo punto medio cae en su núcleo.
    2. Red de seguridad en la frontera: si dos palabras consecutivas tienen el mismo texto
       normalizado y se solapan más de ``dedup_overlap`` (de la más corta), se queda una.
    """
    kept = [w for chunk, words in per_chunk for w in words if chunk.owns((w.start + w.end) / 2)]
    kept.sort(key=lambda w: (w.start, w.end))
    out: list[Word] = []
    for w in kept:
        if out:
            p = out[-1]
            inter = min(p.end, w.end) - max(p.start, w.start)
            shortest = max(min(p.duration, w.duration), 1e-6)
            if _norm(p.text) == _norm(w.text) and inter / shortest > dedup_overlap:
                if (w.prob or 0) > (p.prob or 0):
                    out[-1] = w
                continue
        out.append(w)
    return out


def merge_segments(per_chunk: list[tuple[Chunk, list[Segment]]], join_gap_s: float = 0.0) -> list[Segment]:
    """Une segmentos de voz (sin speaker) recortándolos al núcleo de su trozo.

    Los segmentos que se tocan (hueco <= ``join_gap_s``) se fusionan, así un tramo de voz
    que cruza una frontera vuelve a ser uno solo.
    """
    clipped: list[Segment] = []
    for chunk, segs in per_chunk:
        for s in segs:
            a, b = max(s.start, chunk.core_start), min(s.end, chunk.core_end)
            if b > a:
                clipped.append(s.model_copy(update={"start": a, "end": b}))
    clipped.sort(key=lambda s: s.start)
    out: list[Segment] = []
    for s in clipped:
        if out and s.start - out[-1].end <= join_gap_s:
            prev = out[-1]
            out[-1] = prev.model_copy(update={"end": max(prev.end, s.end)})
        else:
            out.append(s)
    return out


def offset_words(words: list[Word], offset: float) -> list[Word]:
    return [w.model_copy(update={"start": w.start + offset, "end": w.end + offset}) for w in words]
