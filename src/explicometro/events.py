"""Eventos C1–C4 y detección de moderador (funciones puras sobre los esquemas).

- **C1 interrupción competitiva** (CLAUDE.md): B empieza mientras A habla, solape >= 0,5 s
  y A cede (su turno acaba) <= 2 s después de que entre B. Se excluyen asentimientos
  (``Turn.is_backchannel``) tanto de B como de A. Se guarda el solape y si A recupera el turno.
- **C2** (reformulación sin atribución): embeddings; backend configurable (pendiente).
- **C3/C4** llegan ya etiquetados de la etapa ``llm`` y aquí solo se fusionan.
- **Moderador**: metadata (``Speaker.role``) o heurística. Los eventos que emite el moderador
  se marcan ``by_moderator=True``: se reportan aparte y no cuentan en C1–C4, porque
  moderar no es interrumpir.
"""

from __future__ import annotations

import itertools
from collections import Counter

from .backends import not_implemented, resolve
from .config import InterruptionCfg, ModeratorCfg, ReformulationCfg, RuntimeCfg, StageCfg, ThresholdsCfg
from .schemas import Event, EventType, Speaker, Turn


def _floor(turns: list[Turn]) -> list[Turn]:
    """Turnos que toman la palabra (sin asentimientos), ordenados por inicio."""
    return sorted((t for t in turns if not t.is_backchannel), key=lambda t: (t.start, t.end))


# ------------------------------------------------------------------------------- C1


def _recovered(a: Turn, b: Turn, floor: list[Turn], within_s: float) -> bool:
    """¿El siguiente turno (no de B) tras la entrada de B es de A y llega a tiempo?"""
    for t in floor:
        if t.start > b.start and t.speaker != b.speaker and t.id != a.id:
            return t.speaker == a.speaker and t.start <= b.end + within_s
    return False


def detect_interruptions(turns: list[Turn], cfg: InterruptionCfg) -> list[Event]:
    """C1. Emisor = B (quien interrumpe), receptor = A (interrumpido)."""
    floor = _floor(turns)
    events = []
    for b in floor:
        best: tuple[float, Turn] | None = None
        for a in floor:
            if a.speaker == b.speaker or not (a.start < b.start < a.end):
                continue
            overlap = min(a.end, b.end) - b.start
            yields = a.end - b.start <= cfg.yield_within_s
            if overlap >= cfg.min_overlap_s and yields and (best is None or overlap > best[0]):
                best = (overlap, a)
        if best is None:
            continue
        overlap, a = best
        events.append(
            Event(
                type=EventType.INTERRUPTION,
                src_spk=b.speaker,
                dst_spk=a.speaker,
                t_start=b.start,
                t_end=min(a.end, b.end),
                evidence=f"turno {b.id} entra {overlap:.2f} s antes de que acabe el turno {a.id}",
                score=1.0,
                overlap_s=round(overlap, 3),
                recovered=_recovered(a, b, floor, cfg.recover_within_s),
            )
        )
    return events


# ------------------------------------------------------------------------- moderador


def handovers(turns: list[Turn], cfg: ModeratorCfg) -> Counter[str]:
    """Turnos que ceden la palabra: <= ``handover_max_s`` y el siguiente turno (por inicio)
    es de otro hablante y empieza <= ``handover_gap_s`` después de que acabe."""
    floor = _floor(turns)
    count: Counter[str] = Counter()
    for t, nxt in itertools.pairwise(floor):
        if (
            t.duration <= cfg.handover_max_s
            and nxt.speaker != t.speaker
            and nxt.start - t.end <= cfg.handover_gap_s
        ):
            count[t.speaker] += 1
    return count


def detect_moderator(
    turns: list[Turn], speakers: list[Speaker], cfg: ModeratorCfg
) -> tuple[str | None, str | None]:
    """Devuelve ``(spk_id, fuente)`` con fuente ``metadata`` o ``heuristic``.

    Metadata manda: ``role=moderator`` lo fija; ``role=guest`` excluye al hablante. Heurística:
    entre quienes hablan en los ``edge_turns`` primeros **y** últimos turnos, el de más turnos
    que ceden la palabra, si cede >= ``min_handovers`` veces y >= ``min_margin`` × lo que cede
    cualquier otro participante (no invitado por metadata). Empate o ventaja pequeña → nadie:
    mejor sin moderador que uno al azar.
    """
    mods = sorted(s.id for s in speakers if s.role == "moderator")
    if mods:
        return mods[0], "metadata"
    guests = {s.id for s in speakers if s.role == "guest"}
    floor = _floor(turns)
    if not floor:
        return None, None
    opening = {t.speaker for t in floor[: cfg.edge_turns]}
    closing = {t.speaker for t in floor[-cfg.edge_turns :]}
    hand = handovers(turns, cfg)
    eligible = sorted((opening & closing) - guests, key=lambda s: (-hand[s], s))
    if not eligible or hand[eligible[0]] < cfg.min_handovers:
        return None, None
    best = eligible[0]
    others = [hand[t.speaker] for t in floor if t.speaker not in (best, *guests)]
    second = max(others, default=0)
    if hand[best] <= second or hand[best] < cfg.min_margin * second:
        return None, None  # empate o ventaja pequeña: mejor sin moderador que uno dudoso
    return best, "heuristic"


def mark_moderator(events: list[Event], moderator: str | None) -> list[Event]:
    """``by_moderator=True`` en los eventos que emite el moderador (hacia invitados)."""
    if moderator is None:
        return events
    return [e.model_copy(update={"by_moderator": e.src_spk == moderator}) for e in events]


# ------------------------------------------------------------------------------- C2


def _c2_stub(turns: list[Turn], cfg: ReformulationCfg, stage: StageCfg, rt: RuntimeCfg) -> list[Event]:
    return []


def _c2_mrbert(turns: list[Turn], cfg: ReformulationCfg, stage: StageCfg, rt: RuntimeCfg) -> list[Event]:
    raise not_implemented("events", "mrbert", "Instala extras [gpu] (BSC-LT/MrBERT-es).")


BACKENDS = {"stub": _c2_stub, "mrbert": _c2_mrbert}


def run(
    turns: list[Turn],
    speakers: list[Speaker],
    llm_events: list[Event],
    stage: StageCfg,
    thresholds: ThresholdsCfg,
    rt: RuntimeCfg,
) -> list[Event]:
    c1 = detect_interruptions(turns, thresholds.interruption)
    c2 = resolve(BACKENDS, stage.backend, "events")(turns, thresholds.reformulation, stage, rt)
    moderator, _src = detect_moderator(turns, speakers, thresholds.moderator)
    events = mark_moderator([*c1, *c2, *llm_events], moderator)
    return sorted(events, key=lambda e: (e.t_start, e.type.value))
