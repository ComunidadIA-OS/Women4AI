"""Evaluación de eventos predichos frente a anotaciones (funciones puras).

Un evento predicho casa con uno de referencia si es del mismo tipo y sus intervalos
están a <= ``tol_s`` (inicio) o se solapan. Emparejamiento voraz 1:1 por orden temporal.
Los IDs de hablante no se comparan (dependen de la diarización).
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import numpy as np
from pydantic import BaseModel

from .schemas import Event, EventType, Segment, Word


class PRF(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else None


def _match(a: Event, b: Event, tol_s: float) -> bool:
    overlap = min(a.t_end, b.t_end) - max(a.t_start, b.t_start)
    return a.type == b.type and (abs(a.t_start - b.t_start) <= tol_s or overlap > 0)


def score_events(pred: list[Event], gold: list[Event], tol_s: float = 1.0) -> dict[EventType, PRF]:
    out: dict[EventType, PRF] = defaultdict(PRF)
    unmatched = sorted(gold, key=lambda e: e.t_start)
    for p in sorted(pred, key=lambda e: e.t_start):
        hit = next((g for g in unmatched if _match(p, g, tol_s)), None)
        if hit is None:
            out[p.type].fp += 1
        else:
            out[p.type].tp += 1
            unmatched.remove(hit)
    for g in unmatched:
        out[g.type].fn += 1
    return dict(out)


# ------------------------------------------------------------ VAD / diarización / ASR


def _frames(segs: list[Segment], duration: float, step: float) -> np.ndarray:
    n = int(np.ceil(duration / step))
    mask = np.zeros(n, dtype=bool)
    for s in segs:
        mask[int(s.start / step) : int(np.ceil(s.end / step))] = True
    return mask


def speech_detection(hyp: list[Segment], ref: list[Segment], duration: float, step: float = 0.01) -> PRF:
    """Precisión/cobertura de voz a nivel de trama (VAD). ``tp/fp/fn`` en tramas."""
    h, r = _frames(hyp, duration, step), _frames(ref, duration, step)
    return PRF(tp=int((h & r).sum()), fp=int((h & ~r).sum()), fn=int((~h & r).sum()))


def _by_speaker(segs: list[Segment], duration: float, step: float) -> dict[str, np.ndarray]:
    spks = sorted({s.speaker for s in segs if s.speaker})
    return {k: _frames([s for s in segs if s.speaker == k], duration, step) for k in spks}


def speaker_mapping(
    hyp: list[Segment], ref: list[Segment], duration: float, step: float = 0.01
) -> dict[str, str]:
    """Asigna cada hablante hipótesis al de referencia con más tiempo compartido (1:1, voraz)."""
    H, R = _by_speaker(hyp, duration, step), _by_speaker(ref, duration, step)
    pairs = sorted(((int((H[h] & R[r]).sum()), h, r) for h in H for r in R), reverse=True)
    mapping: dict[str, str] = {}
    for n, h, r in pairs:
        if n > 0 and h not in mapping and r not in mapping.values():
            mapping[h] = r
    return mapping


def single_speaker_accuracy(
    hyp: list[Segment], ref: list[Segment], duration: float, step: float = 0.01
) -> float:
    """En tramas con exactamente un hablante de referencia: % donde la hipótesis lo incluye."""
    mapping = speaker_mapping(hyp, ref, duration, step)
    R = _by_speaker(ref, duration, step)
    mapped = [s.model_copy(update={"speaker": mapping[s.speaker]}) for s in hyp if s.speaker in mapping]
    H = _by_speaker(mapped, duration, step)
    count = sum(R.values()) if R else np.zeros(0)
    single = count == 1
    if not single.any():
        return 0.0
    ok = np.zeros_like(single)
    for k, rk in R.items():
        ok |= single & rk & H.get(k, np.zeros_like(rk))
    return float(ok.sum() / single.sum())


def overlap_recall(
    hyp: list[Segment], ref: list[Segment], duration: float, step: float = 0.01
) -> float | None:
    """% de tramas con >= 2 hablantes en la referencia donde la hipótesis también tiene >= 2."""
    R, H = _by_speaker(ref, duration, step), _by_speaker(hyp, duration, step)
    r_ov = (sum(R.values()) >= 2) if R else np.zeros(0, dtype=bool)
    if not r_ov.any():
        return None
    h_ov = (sum(H.values()) >= 2) if H else np.zeros_like(r_ov)
    return float((r_ov & h_ov).sum() / r_ov.sum())


def word_speaker_accuracy(words: list[Word], ref: list[Segment], mapping: dict[str, str]) -> float | None:
    """% de palabras con hablante correcto, contando solo las que caen enteras en zona de un
    único hablante de referencia (en solapes la verdad por palabra es ambigua)."""
    total = ok = 0
    for w in words:
        covering = {s.speaker for s in ref if s.start <= w.start and w.end <= s.end}
        touching = {s.speaker for s in ref if s.start < w.end and w.start < s.end}
        if len(covering) == 1 and covering == touching:
            total += 1
            ok += mapping.get(w.speaker or "") in covering
    return ok / total if total else None


def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w\s']", " ", text).split()


def wer(ref: str, hyp: str) -> float:
    """Word error rate (Levenshtein sobre tokens normalizados, sin puntuación)."""
    r, h = _tokens(ref), _tokens(hyp)
    prev = list(range(len(h) + 1))
    for i, rt in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, ht in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rt != ht))
        prev = cur
    return prev[-1] / max(1, len(r))
