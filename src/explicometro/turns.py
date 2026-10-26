"""Turnos a partir de la diarización + palabras alineadas (funciones puras).

1. Por hablante, se unen sus segmentos de diarización separados por < ``gap_s`` (0,3 s).
   Los turnos de hablantes distintos pueden solaparse: los solapes se conservan para C1.
2. Cada palabra alineada se asigna al turno de su hablante que contiene su punto medio (o
   al más cercano a <= ``max_word_dist_s``).
3. ``is_backchannel``: duración < ``backchannel_max_s`` (1 s) o texto normalizado (minúsculas,
   sin tildes ni puntuación) igual a una entrada del léxico.

Los tiempos del turno son los de la diarización, no los del ASR.
"""

from __future__ import annotations

import bisect
import logging
import re
import unicodedata

from pydantic import Field

from .backends import resolve
from .config import Params, StageCfg
from .schemas import Segment, Turn, Word

log = logging.getLogger(__name__)

DEFAULT_LEXICON = ["sí", "ya", "claro", "vale", "ajá", "exacto", "clar", "sí sí"]


class TurnsParams(Params):
    gap_s: float = Field(default=0.3, ge=0, description="Une segmentos del mismo hablante con hueco < esto")
    backchannel_max_s: float = Field(default=1.0, ge=0)
    backchannel_lexicon: list[str] = Field(default_factory=lambda: list(DEFAULT_LEXICON))
    max_word_dist_s: float = Field(default=1.0, ge=0)


def normalize_text(text: str) -> str:
    """Minúsculas, sin tildes ni puntuación, espacios simples: 'Sí, sí.' → 'si si'."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def is_backchannel(duration: float, text: str, lexicon: set[str], max_s: float) -> bool:
    return duration < max_s or (bool(text) and normalize_text(text) in lexicon)


def merge_segments(diar: list[Segment], gap_s: float) -> list[Segment]:
    """Une, por hablante, segmentos consecutivos con hueco < ``gap_s`` (o solapados)."""
    out: list[Segment] = []
    for spk in sorted({s.speaker for s in diar if s.speaker}):
        cur: Segment | None = None
        for s in sorted((s for s in diar if s.speaker == spk), key=lambda s: s.start):
            if cur is not None and round(s.start - cur.end, 6) < gap_s:  # redondeo: 4.3-4.0 = 0.2999…
                cur = cur.model_copy(update={"end": max(cur.end, s.end)})
            else:
                if cur is not None:
                    out.append(cur)
                cur = s
        if cur is not None:
            out.append(cur)
    return sorted(out, key=lambda s: (s.start, s.end))


def _owner(
    w: Word, by_spk: dict[str, tuple[list[float], list[int]]], spans: list[Segment], max_dist: float
) -> int | None:
    """Índice del turno de ``w.speaker`` que contiene su punto medio o el más cercano."""
    if w.speaker not in by_spk:
        return None
    starts, idx = by_spk[w.speaker]
    mid = (w.start + w.end) / 2
    j = bisect.bisect_right(starts, mid)
    best, best_d = None, max_dist
    for k in (j - 1, j):  # el que empieza justo antes (puede contenerlo) y el siguiente
        if 0 <= k < len(idx):
            s = spans[idx[k]]
            d = 0.0 if s.start <= mid <= s.end else min(abs(mid - s.start), abs(mid - s.end))
            if d <= best_d:
                best, best_d = idx[k], d
    return best


def build_turns(diar: list[Segment], words: list[Word], p: TurnsParams | None = None) -> list[Turn]:
    p = p or TurnsParams()
    spans = merge_segments(diar, p.gap_s)
    texts: list[list[Word]] = [[] for _ in spans]
    by_spk: dict[str, tuple[list[float], list[int]]] = {}
    for i, sp in enumerate(spans):  # spans ya ordenados por inicio
        starts, idx = by_spk.setdefault(sp.speaker, ([], []))  # type: ignore[arg-type]
        starts.append(sp.start)
        idx.append(i)
    dropped = 0
    for w in sorted(words, key=lambda w: w.start):
        k = _owner(w, by_spk, spans, p.max_word_dist_s) if w.speaker else None
        if k is None:
            dropped += 1
        else:
            texts[k].append(w)
    if dropped:
        log.info("turns: %d palabras sin turno de su hablante (descartadas)", dropped)
    lexicon = {normalize_text(x) for x in p.backchannel_lexicon}
    turns = []
    for i, (s, ws) in enumerate(zip(spans, texts, strict=True)):
        text = " ".join(w.text for w in ws)
        turns.append(
            Turn(
                id=i,
                speaker=s.speaker,  # type: ignore[arg-type]
                start=s.start,
                end=s.end,
                text=text,
                n_words=len(ws),
                is_backchannel=is_backchannel(s.duration, text, lexicon, p.backchannel_max_s),
            )
        )
    return turns


def _segments_backend(diar: list[Segment], words: list[Word], p: TurnsParams) -> list[Turn]:
    return build_turns(diar, words, p)


BACKENDS = {"segments": _segments_backend, "stub": _segments_backend}


def run(diar: list[Segment], words: list[Word], cfg: StageCfg) -> list[Turn]:
    return resolve(BACKENDS, cfg.backend, "turns")(diar, words, cfg.parsed(TurnsParams))
