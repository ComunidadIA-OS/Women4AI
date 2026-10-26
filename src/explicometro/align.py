"""Alineación palabra->hablante por máximo solape temporal (determinista, sin modelos)."""

from __future__ import annotations

from .backends import resolve
from .config import StageCfg
from .schemas import Segment, Word


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speakers(words: list[Word], diar: list[Segment]) -> list[Word]:
    """Asigna a cada palabra el hablante con más solape; si no solapa, el más cercano."""
    if not diar:
        return list(words)
    out = []
    for w in words:
        best = max(diar, key=lambda s: _overlap(w.start, w.end, s.start, s.end))
        if _overlap(w.start, w.end, best.start, best.end) == 0.0:
            mid = (w.start + w.end) / 2
            best = min(diar, key=lambda s: min(abs(mid - s.start), abs(mid - s.end)))
        out.append(w.model_copy(update={"speaker": best.speaker}))
    return out


def _overlap_backend(words: list[Word], diar: list[Segment], cfg: StageCfg) -> list[Word]:
    return assign_speakers(words, diar)


BACKENDS = {"overlap": _overlap_backend, "stub": _overlap_backend}


def run(words: list[Word], diar: list[Segment], cfg: StageCfg) -> list[Word]:
    return resolve(BACKENDS, cfg.backend, "align")(words, diar, cfg)
