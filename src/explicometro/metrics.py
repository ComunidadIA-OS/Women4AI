"""Métricas por episodio. Funciones puras: entran tipos del contrato, sale EpisodeMetrics.

Con entradas vacías todo devuelve estructuras vacías o ``None`` (nunca divide por cero).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .config import ProtagonismCfg, StageCfg, ThresholdsCfg
from .schemas import (
    AudioMeta,
    Dyad,
    DyadStats,
    EpisodeMetrics,
    Event,
    EventType,
    Gender,
    GenderShare,
    Presence,
    Protagonism,
    Speaker,
    Turn,
    UnknownShare,
    dyad_of,
)


def _gender_map(speakers: list[Speaker]) -> dict[str, Gender]:
    return {s.id: s.gender for s in speakers}


def _ratio(num: float, den: float) -> float | None:
    return num / den if den > 0 else None


def speech_time_by_gender(turns: list[Turn], speakers: list[Speaker]) -> dict[Gender, float]:
    g = _gender_map(speakers)
    out: dict[Gender, float] = defaultdict(float)
    for t in turns:
        out[g.get(t.speaker, Gender.UNKNOWN)] += t.duration
    return dict(out)


KNOWN = (Gender.MALE, Gender.FEMALE)


def _words_by_gender(turns: list[Turn], speakers: list[Speaker]) -> dict[Gender, int]:
    g = _gender_map(speakers)
    words: dict[Gender, int] = defaultdict(int)
    for t in turns:
        words[g.get(t.speaker, Gender.UNKNOWN)] += t.n_words
    return dict(words)


def presence(turns: list[Turn], speakers: list[Speaker]) -> Presence:
    """A: % hablantes, % tiempo y % palabras por género, **solo sobre H y M**.

    Los hablantes U no entran en numerador ni denominador; lo que queda fuera se reporta
    con ``unknown_share``.
    """
    n_spk = Counter(s.gender for s in speakers if s.gender in KNOWN)
    if not n_spk:
        return Presence()
    time = {k: v for k, v in speech_time_by_gender(turns, speakers).items() if k in KNOWN}
    words = {k: v for k, v in _words_by_gender(turns, speakers).items() if k in KNOWN}
    tot_spk, tot_time, tot_words = sum(n_spk.values()), sum(time.values()), sum(words.values())
    genders = KNOWN
    return Presence(
        by_gender={
            gen: GenderShare(
                speakers=_ratio(n_spk.get(gen, 0), tot_spk),
                time=_ratio(time.get(gen, 0.0), tot_time),
                words=_ratio(words.get(gen, 0), tot_words),
            )
            for gen in sorted(genders)
        }
    )


def protagonism(turns: list[Turn], speakers: list[Speaker], cfg: ProtagonismCfg) -> Protagonism:
    """B: nº turnos y duración media por género (sin U) y quién abre/cierra (primer/último
    turno >= umbral; aquí sí se informa aunque sea U, porque es un hablante concreto).

    TODO: detección de moderador/a (ver ROADMAP.md).
    """
    if not turns:
        return Protagonism()
    g = _gender_map(speakers)
    n: dict[Gender, int] = defaultdict(int)
    dur: dict[Gender, float] = defaultdict(float)
    for t in turns:
        gen = g.get(t.speaker, Gender.UNKNOWN)
        if gen not in KNOWN:  # U excluido de las métricas por género
            continue
        n[gen] += 1
        dur[gen] += t.duration
    long_turns = [t for t in sorted(turns, key=lambda t: t.start) if t.duration >= cfg.min_open_close_turn_s]
    opener = long_turns[0].speaker if long_turns else None
    closer = long_turns[-1].speaker if long_turns else None
    return Protagonism(
        n_turns=dict(n),
        mean_turn_s={k: dur[k] / n[k] for k in n},
        opener=opener,
        opener_gender=g.get(opener, Gender.UNKNOWN) if opener else None,
        closer=closer,
        closer_gender=g.get(closer, Gender.UNKNOWN) if closer else None,
    )


def unknown_share(turns: list[Turn], speakers: list[Speaker]) -> UnknownShare:
    """Cuánto queda fuera de las métricas por género (hablantes U)."""
    time = speech_time_by_gender(turns, speakers)
    words = _words_by_gender(turns, speakers)
    return UnknownShare(
        speakers=sum(1 for s in speakers if s.gender is Gender.UNKNOWN),
        time_s=round(time.get(Gender.UNKNOWN, 0.0), 3),
        time_share=_ratio(time.get(Gender.UNKNOWN, 0.0), sum(time.values())),
        words_share=_ratio(words.get(Gender.UNKNOWN, 0), sum(words.values())),
    )


def event_stats(
    events: list[Event], turns: list[Turn], speakers: list[Speaker]
) -> dict[EventType, dict[Dyad, DyadStats]]:
    """Conteo y tasa por minuto de habla del receptor, por tipo y díada."""
    g = _gender_map(speakers)
    minutes = {k: v / 60 for k, v in speech_time_by_gender(turns, speakers).items()}
    counts: dict[EventType, Counter[Dyad]] = defaultdict(Counter)
    for e in events:
        dyad = dyad_of(g.get(e.src_spk, Gender.UNKNOWN), g.get(e.dst_spk, Gender.UNKNOWN))
        if dyad is not None:
            counts[e.type][dyad] += 1
    out: dict[EventType, dict[Dyad, DyadStats]] = {}
    for etype, c in counts.items():
        out[etype] = {
            d: DyadStats(count=k, rate_per_min=_ratio(k, minutes.get(Gender(d.value[-1]), 0.0)))
            for d, k in c.items()
        }
    return out


def interruption_excess(events: list[Event], turns: list[Turn], speakers: list[Speaker]) -> float | None:
    """% interrupciones (C1) recibidas por mujeres / % tiempo de habla de mujeres."""
    g = _gender_map(speakers)
    recv = [g.get(e.dst_spk, Gender.UNKNOWN) for e in events if e.type is EventType.INTERRUPTION]
    recv_known = [x for x in recv if x is not Gender.UNKNOWN]
    time = speech_time_by_gender(turns, speakers)
    time_known = time.get(Gender.MALE, 0.0) + time.get(Gender.FEMALE, 0.0)
    share_recv = _ratio(recv_known.count(Gender.FEMALE), len(recv_known))
    share_time = _ratio(time.get(Gender.FEMALE, 0.0), time_known)
    if share_recv is None or not share_time:
        return None
    return share_recv / share_time


def asymmetry_ratio(stats: dict[EventType, dict[Dyad, DyadStats]]) -> float | None:
    """Tasa C1 H->M / tasa C1 H->H (punto). El IC 95 % se calcula agregando por programa."""
    c1 = stats.get(EventType.INTERRUPTION, {})
    hm, hh = c1.get(Dyad.H_M), c1.get(Dyad.H_H)
    if hm is None or hh is None or hm.rate_per_min is None or not hh.rate_per_min:
        return None
    return hm.rate_per_min / hh.rate_per_min


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
    stats = event_stats(events, turns, speakers)
    return EpisodeMetrics(
        episode_id=audio.episode_id,
        dry_run=dry_run,
        config_hash=config_hash,
        duration_s=audio.duration_s,
        speech_s=round(sum(t.duration for t in turns), 3),
        n_speakers=len(speakers),
        presence=presence(turns, speakers),
        unknown=unknown_share(turns, speakers),
        protagonism=protagonism(turns, speakers, thresholds.protagonism),
        events=stats,
        interruption_excess=interruption_excess(events, turns, speakers),
        asymmetry_ratio=asymmetry_ratio(stats),
    )
