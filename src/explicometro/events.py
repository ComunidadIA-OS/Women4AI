"""Eventos C1–C4 y detección de moderador (funciones puras sobre los esquemas).

- **C1 interrupción competitiva** (docs/metrics.md): B empieza mientras A habla, solape >= 0,5 s
  y A cede (su turno acaba) <= 2 s después de que entre B. Se excluyen asentimientos
  (``Turn.is_backchannel``) tanto de B como de A. Se guarda el solape y si A recupera el turno.
- **C2** (reformulación sin atribución): embeddings BSC-LT/MrBERT-es (mean pooling, L2) de
  turnos >= 15 palabras; cada turno de B contra los 5 turnos previos de otros hablantes;
  candidato si coseno >= τ y B no atribuye la idea (regex). τ se calibra en el gold set.
- **C3/C4** llegan ya etiquetados de la etapa ``llm`` y aquí solo se fusionan.
- **Moderador**: metadata (``Speaker.role``) o heurística. Los eventos que emite el moderador
  se marcan ``by_moderator=True``: se reportan aparte y no cuentan en C1–C4, porque
  moderar no es interrumpir.
"""

from __future__ import annotations

import itertools
import logging
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pydantic import Field

from .backends import BackendError, resolve
from .config import (
    InterruptionCfg,
    ModeratorCfg,
    Params,
    ReformulationCfg,
    RuntimeCfg,
    StageCfg,
    ThresholdsCfg,
)
from .schemas import Event, EventType, Speaker, Turn

log = logging.getLogger(__name__)


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

Embedder = Callable[[list[str]], np.ndarray]


class EventsParams(Params):
    embedding_model: str = "BSC-LT/MrBERT-es"
    max_length: int = Field(default=512, ge=16)
    batch_size: int = Field(default=32, ge=1)
    device_index: int = Field(default=0, ge=0)
    trust_remote_code: bool = False


@dataclass(frozen=True)
class C2Candidate:
    b: Turn
    a: Turn
    cos: float


def has_attribution(text: str, patterns: list[str]) -> bool:
    """¿B atribuye la idea a otra persona? (regex sin distinguir mayúsculas)."""
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def c2_candidates(turns: list[Turn], embed: Embedder, cfg: ReformulationCfg) -> list[C2Candidate]:
    """Todos los pares (B, A) a comparar, con su coseno y **sin aplicar τ** (sirve para calibrar).

    B = turno (sin asentimientos) con >= ``min_words`` palabras y sin atribución explícita;
    A = cada uno de los ``window_prev_turns`` turnos elegibles anteriores de **otros** hablantes.
    """
    elig = [t for t in _floor(turns) if t.n_words >= cfg.min_words]
    if len(elig) < 2:
        return []
    E = embed([t.text for t in elig])
    out = []
    for j, b in enumerate(elig):
        if has_attribution(b.text, cfg.attribution_patterns):
            continue
        prev = [i for i in range(j - 1, -1, -1) if elig[i].speaker != b.speaker and elig[i].start < b.start]
        for i in prev[: cfg.window_prev_turns]:
            out.append(C2Candidate(b=b, a=elig[i], cos=float(E[j] @ E[i])))
    return out


def reformulations_from(cands: list[C2Candidate], tau: float) -> list[Event]:
    """Un evento por turno B: el turno A más parecido, si coseno >= τ."""
    best: dict[int, C2Candidate] = {}
    for c in cands:
        if c.cos >= tau and (c.b.id not in best or c.cos > best[c.b.id].cos):
            best[c.b.id] = c
    return [
        Event(
            type=EventType.REFORMULATION,
            src_spk=c.b.speaker,
            dst_spk=c.a.speaker,
            t_start=c.b.start,
            t_end=c.b.end,
            evidence=" ".join(c.b.text.split()[:20]) + f" ≈ turno {c.a.id} (cos {c.cos:.2f})",
            score=round(max(0.0, min(1.0, c.cos)), 4),
        )
        for c in sorted(best.values(), key=lambda c: c.b.start)
    ]


def detect_reformulations(turns: list[Turn], embed: Embedder, cfg: ReformulationCfg) -> list[Event]:
    if cfg.cos_threshold is None:
        log.warning(
            "C2: τ (thresholds.reformulation.cos_threshold) sin calibrar → no se emiten eventos C2. "
            "Calíbralo con scripts/calibrate_c2.py"
        )
        return []
    return reformulations_from(c2_candidates(turns, embed, cfg), cfg.cos_threshold)


class MrBertEmbedder:  # pragma: no cover - requiere torch/transformers y el modelo
    """Mean pooling sobre la última capa (con máscara) + normalización L2. fp16 en GPU."""

    def __init__(self, p: EventsParams, rt: RuntimeCfg):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise BackendError("MrBERT necesita torch y transformers: pip install -e '.[gpu]'") from e
        self.torch, self.p = torch, p
        self.device = f"cuda:{p.device_index}" if rt.device == "cuda" else "cpu"
        dtype = torch.float16 if rt.device == "cuda" else torch.float32  # V100S: fp16, nunca bf16
        self.tok = AutoTokenizer.from_pretrained(p.embedding_model, trust_remote_code=p.trust_remote_code)
        kw = {"torch_dtype": dtype, "trust_remote_code": p.trust_remote_code}
        try:  # sdpa (sin flash-attention); si el modelo no lo admite, eager
            self.model = AutoModel.from_pretrained(p.embedding_model, attn_implementation="sdpa", **kw)
        except (ValueError, ImportError):
            self.model = AutoModel.from_pretrained(p.embedding_model, attn_implementation="eager", **kw)
        self.model.to(self.device).eval()

    def __call__(self, texts: list[str]) -> np.ndarray:
        torch, out = self.torch, []
        for i in range(0, len(texts), self.p.batch_size):
            enc = self.tok(
                texts[i : i + self.p.batch_size],
                padding=True,
                truncation=True,
                max_length=self.p.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        E = np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)
        return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)


def _c2_stub(turns: list[Turn], cfg: ReformulationCfg, p: EventsParams, rt: RuntimeCfg) -> list[Event]:
    return []


def _c2_mrbert(turns: list[Turn], cfg: ReformulationCfg, p: EventsParams, rt: RuntimeCfg) -> list[Event]:
    if cfg.cos_threshold is None:
        return detect_reformulations(
            turns, lambda t: np.zeros((len(t), 1)), cfg
        )  # avisa y no carga el modelo
    if sum(1 for t in turns if not t.is_backchannel and t.n_words >= cfg.min_words) < 2:
        return []
    return detect_reformulations(turns, MrBertEmbedder(p, rt), cfg)


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
    c2 = resolve(BACKENDS, stage.backend, "events")(
        turns, thresholds.reformulation, stage.parsed(EventsParams), rt
    )
    moderator, _src = detect_moderator(turns, speakers, thresholds.moderator)
    events = mark_moderator([*c1, *c2, *llm_events], moderator)
    return sorted(events, key=lambda e: (e.t_start, e.type.value))
