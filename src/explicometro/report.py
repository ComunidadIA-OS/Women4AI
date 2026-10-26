"""Informe legible (Markdown) a partir de EpisodeMetrics. Sin nombres de personas."""

from __future__ import annotations

from .backends import resolve
from .config import StageCfg
from .schemas import EpisodeMetrics, Gender

_LABEL = {Gender.MALE: "Hombres", Gender.FEMALE: "Mujeres", Gender.UNKNOWN: "Desconocido"}


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f} %"


def _num(x: float | None) -> str:
    return "—" if x is None else f"{x:.2f}"


def to_markdown(m: EpisodeMetrics) -> str:
    lines = [
        f"# Explicómetro — {m.episode_id}",
        "",
        f"Duración: {m.duration_s:.1f} s · Habla: {m.speech_s:.1f} s · Hablantes: {m.n_speakers}"
        + (" · **dry-run**" if m.dry_run else ""),
        "",
        "## A. Presencia",
        "",
        "| Género | Hablantes | Tiempo | Palabras |",
        "|---|---|---|---|",
    ]
    for gen, s in m.presence.by_gender.items():
        lines.append(f"| {_LABEL[gen]} | {_pct(s.speakers)} | {_pct(s.time)} | {_pct(s.words)} |")
    if not m.presence.by_gender:
        lines.append("| (sin datos) | — | — | — |")
    u = m.unknown
    lines += [
        "",
        f"Fuera de las métricas por género (U): {u.speakers} hablantes, {u.time_s:.1f} s "
        f"({_pct(u.time_share)} del tiempo de habla, {_pct(u.words_share)} de las palabras).",
    ]
    lines += ["", "## C. Conductas por díada (emisor→receptor)", ""]
    if not m.events:
        lines.append("Sin eventos detectados.")
    for etype, dyads in m.events.items():
        lines.append(
            f"**{etype.value}**: "
            + ", ".join(f"{d.value} {s.count} ({_num(s.rate_per_min)}/min)" for d, s in dyads.items())
        )
    lines += [
        "",
        f"Exceso de interrupción: {_num(m.interruption_excess)} · "
        f"Ratio de asimetría (punto): {_num(m.asymmetry_ratio)}",
        "",
    ]
    return "\n".join(lines)


def _markdown(m: EpisodeMetrics, cfg: StageCfg) -> str:
    return to_markdown(m)


BACKENDS = {"markdown": _markdown, "stub": _markdown}


def run(m: EpisodeMetrics, cfg: StageCfg) -> str:
    return resolve(BACKENDS, cfg.backend, "report")(m, cfg)
