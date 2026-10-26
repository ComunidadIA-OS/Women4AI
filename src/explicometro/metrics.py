"""Métricas por episodio y agregadas (funciones puras).

Cada ``EpisodeMetrics`` guarda **estadísticos suficientes** (hablantes, segundos y palabras
por género; recuentos por tipo y díada; turnos), de modo que agregar episodios es sumar y
recalcular, y el bootstrap remuestrea episodios sin volver a leer turnos.

Convenciones:
- Solo cuentan H y M; lo que es U se reporta en ``unknown``.
- Las tasas por díada se normalizan por minuto de habla del **género receptor** (todo su
  tiempo de habla, asentimientos incluidos).
- Los eventos emitidos por el moderador (``by_moderator``) van a ``moderator_events``.
- Exceso de interrupción = %C1 recibidas por M / %tiempo de habla de M (C1 con díada conocida).
- Ratio de asimetría = tasa C1 H→M / tasa C1 H→H.
- Con < ``min_events`` eventos en un ratio: ``status="insuficiente"`` y sin número.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .config import ProtagonismCfg, RatiosCfg, StageCfg, ThresholdsCfg
from .events import detect_moderator
from .schemas import (
    AudioMeta,
    Dyad,
    DyadStats,
    EpisodeMetrics,
    Event,
    EventTable,
    EventType,
    Gender,
    GenderShare,
    GroupMetrics,
    Presence,
    Protagonism,
    ProtagonismAgg,
    RatioStat,
    Speaker,
    Turn,
    UnknownShare,
    dyad_of,
)

KNOWN = (Gender.MALE, Gender.FEMALE)
H, M, U = Gender.MALE, Gender.FEMALE, Gender.UNKNOWN

Stat = Callable[[dict[Dyad, int], dict[Gender, float]], tuple[float | None, int]]


def _ratio(num: float, den: float) -> float | None:
    return num / den if den > 0 else None


def _gmap(speakers: list[Speaker]) -> dict[str, Gender]:
    return {s.id: s.gender for s in speakers}


# ------------------------------------------------------------- estadísticos suficientes


def speakers_by_gender(speakers: list[Speaker]) -> dict[Gender, int]:
    return dict(Counter(s.gender for s in speakers))


def speech_by_gender(turns: list[Turn], speakers: list[Speaker]) -> dict[Gender, float]:
    g = _gmap(speakers)
    out: dict[Gender, float] = defaultdict(float)
    for t in turns:
        out[g.get(t.speaker, U)] += t.duration
    return {k: round(v, 3) for k, v in out.items()}


def words_by_gender(turns: list[Turn], speakers: list[Speaker]) -> dict[Gender, int]:
    g = _gmap(speakers)
    out: dict[Gender, int] = defaultdict(int)
    for t in turns:
        out[g.get(t.speaker, U)] += t.n_words
    return dict(out)


def count_events(
    events: Iterable[Event], speakers: list[Speaker], by_moderator: bool
) -> dict[EventType, dict[Dyad, int]]:
    """Recuento por tipo y díada (solo díadas con ambos géneros conocidos)."""
    g = _gmap(speakers)
    out: dict[EventType, Counter[Dyad]] = defaultdict(Counter)
    for e in events:
        if e.by_moderator != by_moderator:
            continue
        dyad = dyad_of(g.get(e.src_spk, U), g.get(e.dst_spk, U))
        if dyad is not None:
            out[e.type][dyad] += 1
    return {t: dict(c) for t, c in out.items()}


def receptor_gender(d: Dyad) -> Gender:
    return Gender(d.value.split("->")[1])


def event_table(counts: dict[EventType, dict[Dyad, int]], speech: dict[Gender, float]) -> EventTable:
    """Tasas por minuto de habla del género receptor, con recuento y minutos."""
    out: EventTable = {}
    for etype, c in sorted(counts.items()):
        row = {}
        for d, n in sorted(c.items()):
            minutes = speech.get(receptor_gender(d), 0.0) / 60
            rate = _ratio(n, minutes)
            row[d] = DyadStats(
                count=int(n),
                receptor_min=round(minutes, 3),
                rate_per_min=None if rate is None else round(rate, 4),
            )
        out[etype] = row
    return out


def table_counts(table: EventTable) -> dict[EventType, dict[Dyad, int]]:
    return {t: {d: s.count for d, s in row.items()} for t, row in table.items()}


# ----------------------------------------------------------------------- A, U y ratios


def presence_from(spk: dict[Gender, int], speech: dict[Gender, float], words: dict[Gender, int]) -> Presence:
    """A sobre H y M (U fuera de numerador y denominador)."""
    tot_spk = sum(spk.get(g, 0) for g in KNOWN)
    if tot_spk == 0:
        return Presence()
    tot_t = sum(speech.get(g, 0.0) for g in KNOWN)
    tot_w = sum(words.get(g, 0) for g in KNOWN)
    return Presence(
        by_gender={
            g: GenderShare(
                speakers=_ratio(spk.get(g, 0), tot_spk),
                time=_ratio(speech.get(g, 0.0), tot_t),
                words=_ratio(words.get(g, 0), tot_w),
            )
            for g in KNOWN
        }
    )


def unknown_from(
    spk: dict[Gender, int], speech: dict[Gender, float], words: dict[Gender, int]
) -> UnknownShare:
    return UnknownShare(
        speakers=spk.get(U, 0),
        time_s=round(speech.get(U, 0.0), 3),
        time_share=_ratio(speech.get(U, 0.0), sum(speech.values())),
        words_share=_ratio(words.get(U, 0), sum(words.values())),
    )


def excess_from(c1: dict[Dyad, int], speech: dict[Gender, float]) -> tuple[float | None, int]:
    """(exceso de interrupción, nº de C1 con díada conocida)."""
    n = int(sum(c1.values()))
    recv_m = c1.get(Dyad.H_M, 0) + c1.get(Dyad.M_M, 0)
    share_recv = _ratio(recv_m, n)
    share_time = _ratio(speech.get(M, 0.0), speech.get(M, 0.0) + speech.get(H, 0.0))
    if share_recv is None or not share_time:
        return None, n
    return share_recv / share_time, n


def asymmetry_from(c1: dict[Dyad, int], speech: dict[Gender, float]) -> tuple[float | None, int]:
    """(tasa H→M / tasa H→H, nº de eventos H→M + H→H)."""
    hm, hh = c1.get(Dyad.H_M, 0), c1.get(Dyad.H_H, 0)
    rate_hm = _ratio(hm, speech.get(M, 0.0) / 60)
    rate_hh = _ratio(hh, speech.get(H, 0.0) / 60)
    if rate_hm is None or not rate_hh:
        return None, int(hm + hh)
    return rate_hm / rate_hh, int(hm + hh)


def ratio_stat(
    value: float | None, n_events: int, n_episodes: int, cfg: RatiosCfg, ci: tuple[float, float] | None = None
) -> RatioStat:
    """Empaqueta un ratio con su tamaño muestral; < ``min_events`` → insuficiente, sin número."""
    if n_events < cfg.min_events:
        status = "insuficiente" if n_events > 0 else "sin_datos"
        return RatioStat(n_events=n_events, n_episodes=n_episodes, status=status)
    if value is None:
        return RatioStat(n_events=n_events, n_episodes=n_episodes, status="sin_datos")
    lo, hi = ci if ci else (None, None)
    return RatioStat(
        value=round(value, 4), ci_low=lo, ci_high=hi, n_events=n_events, n_episodes=n_episodes, status="ok"
    )


# ------------------------------------------------------------------------------ B


def protagonism(
    turns: list[Turn],
    speakers: list[Speaker],
    cfg: ProtagonismCfg,
    moderator: tuple[str | None, str | None] = (None, None),
) -> Protagonism:
    """B: turnos y duración media por género (sin asentimientos ni U), asentimientos aparte,
    quién abre/cierra (primer/último turno >= ``min_open_close_turn_s``) y quién modera."""
    g = _gmap(speakers)
    n: dict[Gender, int] = defaultdict(int)
    dur: dict[Gender, float] = defaultdict(float)
    bc: dict[Gender, int] = defaultdict(int)
    for t in turns:
        gen = g.get(t.speaker, U)
        if gen not in KNOWN:
            continue
        if t.is_backchannel:
            bc[gen] += 1
        else:
            n[gen] += 1
            dur[gen] += t.duration
    long_turns = sorted(
        (t for t in turns if t.duration >= cfg.min_open_close_turn_s and not t.is_backchannel),
        key=lambda t: t.start,
    )
    opener = long_turns[0].speaker if long_turns else None
    closer = long_turns[-1].speaker if long_turns else None
    mod, src = moderator
    return Protagonism(
        n_turns=dict(n),
        mean_turn_s={k: round(dur[k] / n[k], 3) for k in n},
        n_backchannels=dict(bc),
        opener=opener,
        opener_gender=g.get(opener, U) if opener else None,
        closer=closer,
        closer_gender=g.get(closer, U) if closer else None,
        moderator=mod,
        moderator_gender=g.get(mod, U) if mod else None,
        moderator_source=src,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------------- episodio


def run(
    audio: AudioMeta,
    speakers: list[Speaker],
    turns: list[Turn],
    events: list[Event],
    stage: StageCfg,
    thresholds: ThresholdsCfg,
    *,
    dry_run: bool = False,
    config_hash: str | None = None,
) -> EpisodeMetrics:
    spk = speakers_by_gender(speakers)
    speech = speech_by_gender(turns, speakers)
    words = words_by_gender(turns, speakers)
    table = event_table(count_events(events, speakers, by_moderator=False), speech)
    c1 = table_counts(table).get(EventType.INTERRUPTION, {})
    rc = thresholds.ratios
    return EpisodeMetrics(
        episode_id=audio.episode_id,
        dry_run=dry_run,
        config_hash=config_hash,
        duration_s=audio.duration_s,
        speech_s=round(sum(t.duration for t in turns), 3),
        n_speakers=len(speakers),
        speakers_by_gender=spk,
        speech_by_gender=speech,
        words_by_gender=words,
        presence=presence_from(spk, speech, words),
        unknown=unknown_from(spk, speech, words),
        protagonism=protagonism(
            turns, speakers, thresholds.protagonism, detect_moderator(turns, speakers, thresholds.moderator)
        ),
        events=table,
        moderator_events=event_table(count_events(events, speakers, by_moderator=True), speech),
        interruption_excess=ratio_stat(*excess_from(c1, speech), 1, rc),
        asymmetry_ratio=ratio_stat(*asymmetry_from(c1, speech), 1, rc),
    )


# ------------------------------------------------------------------------ agregación


def _sum(dicts: Iterable[dict]) -> dict:
    out: dict = defaultdict(float)
    for d in dicts:
        for k, v in d.items():
            out[k] += v
    return dict(out)


def _sum_counts(tables: Iterable[EventTable]) -> dict[EventType, dict[Dyad, int]]:
    out: dict[EventType, dict[Dyad, int]] = defaultdict(lambda: defaultdict(int))
    for t in tables:
        for etype, row in t.items():
            for d, s in row.items():
                out[etype][d] += s.count
    return {k: dict(v) for k, v in out.items()}


def _c1(ep: EpisodeMetrics) -> dict[Dyad, int]:
    return {d: s.count for d, s in ep.events.get(EventType.INTERRUPTION, {}).items()}


def bootstrap_ci(eps: Sequence[EpisodeMetrics], stat: Stat, cfg: RatiosCfg) -> tuple[float, float] | None:
    """IC percentil remuestreando **episodios** con reemplazo (``bootstrap_iterations``,
    semilla fija); cada réplica recalcula el ratio agrupado. ``None`` con < 2 episodios o si
    menos de la mitad de las réplicas son calculables (p. ej. sin habla de M)."""
    if len(eps) < 2:
        return None
    dyads = list(Dyad)
    C = np.array([[_c1(e).get(d, 0) for d in dyads] for e in eps], dtype=float)
    T = np.array([[e.speech_by_gender.get(g, 0.0) for g in KNOWN] for e in eps], dtype=float)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.integers(0, len(eps), size=(cfg.bootstrap_iterations, len(eps)))
    Cs, Ts = C[idx].sum(axis=1), T[idx].sum(axis=1)
    vals = []
    for c, t in zip(Cs, Ts, strict=True):
        v, _n = stat(dict(zip(dyads, c.tolist(), strict=True)), {H: float(t[0]), M: float(t[1])})
        if v is not None:
            vals.append(v)
    if len(vals) < cfg.bootstrap_iterations / 2:
        return None
    alpha = (1 - cfg.ci) / 2
    lo, hi = np.quantile(vals, [alpha, 1 - alpha])
    return round(float(lo), 4), round(float(hi), 4)


def aggregate(
    eps: Sequence[EpisodeMetrics],
    thresholds: ThresholdsCfg,
    group_by: list[str] | None = None,
    group: dict[str, str] | None = None,
) -> GroupMetrics:
    """Agrega episodios: suma estadísticos suficientes, recalcula A/B/tasas y ratios
    agrupados (no promedio de ratios), con IC 95 % por bootstrap sobre episodios."""
    rc = thresholds.ratios
    spk = {k: int(v) for k, v in _sum(e.speakers_by_gender for e in eps).items()}
    speech = {k: round(v, 3) for k, v in _sum(e.speech_by_gender for e in eps).items()}
    words = {k: int(v) for k, v in _sum(e.words_by_gender for e in eps).items()}
    counts = _sum_counts(e.events for e in eps)
    c1 = counts.get(EventType.INTERRUPTION, {})

    n_turns = {k: int(v) for k, v in _sum(e.protagonism.n_turns for e in eps).items()}
    tot_dur = _sum(
        {g: e.protagonism.mean_turn_s.get(g, 0.0) * n for g, n in e.protagonism.n_turns.items()} for e in eps
    )
    p = [e.protagonism for e in eps]
    prot = ProtagonismAgg(
        n_turns=n_turns,
        mean_turn_s={g: round(tot_dur[g] / n, 3) for g, n in n_turns.items() if n},
        n_backchannels={k: int(v) for k, v in _sum(x.n_backchannels for x in p).items()},
        openers=dict(Counter(x.opener_gender for x in p if x.opener_gender)),
        closers=dict(Counter(x.closer_gender for x in p if x.closer_gender)),
        moderators=dict(Counter(x.moderator_gender for x in p if x.moderator_gender)),
    )
    return GroupMetrics(
        group_by=group_by or [],
        group=group or {},
        episodes=[e.episode_id for e in eps],
        n_episodes=len(eps),
        duration_s=round(sum(e.duration_s for e in eps), 3),
        speakers_by_gender=spk,
        speech_by_gender=speech,
        words_by_gender=words,
        presence=presence_from(spk, speech, words),
        unknown=unknown_from(spk, speech, words),
        protagonism=prot,
        events=event_table(counts, speech),
        moderator_events=event_table(_sum_counts(e.moderator_events for e in eps), speech),
        interruption_excess=ratio_stat(
            *excess_from(c1, speech), len(eps), rc, bootstrap_ci(eps, excess_from, rc)
        ),
        asymmetry_ratio=ratio_stat(
            *asymmetry_from(c1, speech), len(eps), rc, bootstrap_ci(eps, asymmetry_from, rc)
        ),
        bootstrap={"iterations": rc.bootstrap_iterations, "seed": rc.seed, "ci": rc.ci},
    )
