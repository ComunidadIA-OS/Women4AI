"""Eventos C1–C4.

- C1 (interrupción competitiva): reglas sobre turnos, siempre activo (sin modelos).
- C2 (reformulación sin atribución): embeddings (backend configurable).
- C3/C4: vienen ya etiquetados de la etapa ``llm`` y aquí solo se fusionan.
"""

from __future__ import annotations

import re
import unicodedata

from .backends import not_implemented, resolve
from .config import InterruptionCfg as C1Cfg
from .config import ReformulationCfg, RuntimeCfg, StageCfg, ThresholdsCfg
from .schemas import Event, EventType, Speaker, Turn


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower())
    return " ".join(re.sub(r"[^\w\s']", " ", text).split())


def is_backchannel(turn: Turn, cfg: C1Cfg) -> bool:
    """Asentimiento: turno corto (< backchannel_max_s) o formado solo por léxico de backchannel."""
    if turn.duration < cfg.backchannel_max_s:
        return True
    lexicon = {_norm(x) for x in cfg.backchannel_lexicon}
    text = _norm(turn.text)
    if not text:
        return False
    if text in lexicon:
        return True
    return all(tok in lexicon for tok in text.split())


def detect_interruptions(turns: list[Turn], cfg: C1Cfg) -> list[Event]:
    """C1: B empieza con A hablando, solape >= min_overlap_s y A cede en <= yield_within_s.

    Función pura. Emisor = B (quien interrumpe), receptor = A.
    """
    events: list[Event] = []
    ordered = sorted(turns, key=lambda t: t.start)
    for b in ordered:
        if is_backchannel(b, cfg):
            continue
        for a in ordered:
            if a.speaker == b.speaker or not (a.start < b.start < a.end):
                continue
            overlap = min(a.end, b.end) - b.start
            yielded = a.end - b.start <= cfg.yield_within_s
            if overlap >= cfg.min_overlap_s and yielded:
                events.append(
                    Event(
                        type=EventType.INTERRUPTION,
                        src_spk=b.speaker,
                        dst_spk=a.speaker,
                        t_start=b.start,
                        t_end=min(a.end, b.end),
                        evidence=f"turno {b.id} solapa {overlap:.2f}s con turno {a.id}",
                        score=1.0,
                    )
                )
                break
    return events


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
    return sorted([*c1, *c2, *llm_events], key=lambda e: (e.t_start, e.type.value))
