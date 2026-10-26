"""Alineación palabra → hablante (determinista, sin modelos).

Regla (por palabra ``w``):

1. Candidatos = segmentos de diarización que se solapan con ``w``; por hablante se toma su
   segmento de mayor solape.
2. Un solo hablante candidato → ese hablante (mayor solape temporal).
3. Varios (zona de solape): el segmento **nuevo** es el que empezó más tarde. Si ``w`` cae
   entera dentro del nuevo → hablante nuevo; si no → el hablante cuyo segmento empezó antes.
   Así, las palabras que A sigue diciendo mientras B arranca se quedan en A, y lo que B
   dice ya dentro de su intervención va a B (necesario para medir interrupciones).
4. Sin solape: el segmento más cercano si está a <= ``max_gap_s``; si no, sin hablante.
"""

from __future__ import annotations

from pydantic import Field

from .backends import resolve
from .config import Params, StageCfg
from .schemas import Segment, Word


class AlignParams(Params):
    max_gap_s: float = Field(default=0.5, ge=0, description="Distancia máxima para palabras sin solape")
    eps: float = Field(default=0.02, ge=0, description="Tolerancia (s) para 'cae entera dentro'")


def _overlap(w: Word, s: Segment) -> float:
    return max(0.0, min(w.end, s.end) - max(w.start, s.start))


def _distance(w: Word, s: Segment) -> float:
    if w.end < s.start:
        return s.start - w.end
    if w.start > s.end:
        return w.start - s.end
    return 0.0


def speaker_for(w: Word, diar: list[Segment], max_gap_s: float = 0.5, eps: float = 0.02) -> str | None:
    best_by_spk: dict[str, Segment] = {}
    for s in diar:
        if s.speaker is None or _overlap(w, s) <= 0:
            continue
        cur = best_by_spk.get(s.speaker)
        if cur is None or _overlap(w, s) > _overlap(w, cur):
            best_by_spk[s.speaker] = s
    if len(best_by_spk) == 1:
        return next(iter(best_by_spk))
    if len(best_by_spk) > 1:
        cands = sorted(best_by_spk.values(), key=lambda s: (s.start, -s.end))
        newest, oldest = cands[-1], cands[0]
        if newest.start - eps <= w.start and w.end <= newest.end + eps:
            return newest.speaker
        return oldest.speaker
    near = [s for s in diar if s.speaker is not None and _distance(w, s) <= max_gap_s]
    if not near:
        return None
    return min(near, key=lambda s: _distance(w, s)).speaker


def _bucket_index(diar: list[Segment], size: float = 1.0) -> dict[int, list[Segment]]:
    """Índice por cubetas de ``size`` s para no recorrer toda la diarización por palabra."""
    idx: dict[int, list[Segment]] = {}
    for s in diar:
        for b in range(int(s.start // size), int(s.end // size) + 1):
            idx.setdefault(b, []).append(s)
    return idx


def assign_speakers(
    words: list[Word], diar: list[Segment], max_gap_s: float = 0.5, eps: float = 0.02
) -> list[Word]:
    """Devuelve las palabras con ``speaker`` asignado (``None`` si no hay hablante cerca)."""
    idx = _bucket_index(diar)
    out = []
    for w in words:
        lo, hi = int((w.start - max_gap_s) // 1.0), int((w.end + max_gap_s) // 1.0)
        seen: dict[int, Segment] = {}
        for b in range(lo, hi + 1):
            for s in idx.get(b, ()):
                seen[id(s)] = s
        cands = sorted(seen.values(), key=lambda s: s.start)
        out.append(w.model_copy(update={"speaker": speaker_for(w, cands, max_gap_s, eps)}))
    return out


def _overlap_backend(words: list[Word], diar: list[Segment], p: AlignParams) -> list[Word]:
    return assign_speakers(words, diar, p.max_gap_s, p.eps)


BACKENDS = {"overlap": _overlap_backend, "stub": _overlap_backend}


def run(words: list[Word], diar: list[Segment], cfg: StageCfg) -> list[Word]:
    return resolve(BACKENDS, cfg.backend, "align")(words, diar, cfg.parsed(AlignParams))
