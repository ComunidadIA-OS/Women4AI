"""Evaluación de eventos predichos frente a anotaciones (funciones puras).

Un evento predicho casa con uno de referencia si es del mismo tipo y sus intervalos
están a <= ``tol_s`` (inicio) o se solapan. Emparejamiento voraz 1:1 por orden temporal.
Los IDs de hablante no se comparan (dependen de la diarización).
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from .schemas import Event, EventType


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
