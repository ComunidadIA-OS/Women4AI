"""Informe legible (Markdown) de un episodio o de un grupo de episodios. Sin nombres."""

from __future__ import annotations

from .backends import resolve
from .config import StageCfg
from .schemas import EpisodeMetrics, EventTable, Gender, GroupMetrics, RatioStat

_LABEL = {Gender.MALE: "Hombres", Gender.FEMALE: "Mujeres", Gender.UNKNOWN: "Desconocido"}


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f} %"


def _num(x: float | None, nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def fmt3(x: float | None) -> str:
    """Número con 3 decimales o un guion de ancho fijo (tablas P/R/F1)."""
    return "  —  " if x is None else f"{x:.3f}"


def fmt_ratio(r: RatioStat) -> str:
    """'1.84 [IC 95 %: 1.20–2.61] (n=37 eventos, 12 episodios)' o 'insuficiente (n=4 ...)'."""
    n = f"n={r.n_events} eventos, {r.n_episodes} episodio{'s' if r.n_episodes != 1 else ''}"
    if r.status == "insuficiente":
        return f"insuficiente ({n})"
    if r.status == "sin_datos":
        return f"sin datos ({n})"
    ci = f" [IC 95 %: {_num(r.ci_low)}–{_num(r.ci_high)}]" if r.ci_low is not None else ""
    return f"{_num(r.value)}{ci} ({n})"


def _events(title: str, table: EventTable) -> list[str]:
    lines = ["", f"### {title}", ""]
    if not table:
        return [*lines, "Sin eventos."]
    lines += ["| Conducta | Díada | Eventos | Min. habla receptor | Tasa/min |", "|---|---|---|---|---|"]
    for etype, row in table.items():
        for d, s in row.items():
            rate = _num(s.rate_per_min, 3)
            lines.append(f"| {etype.value} | {d.value} | {s.count} | {s.receptor_min:.1f} | {rate} |")
    return lines


def _who(spk: str | None, g: Gender | None) -> str:
    return "—" if spk is None else f"{spk} ({_LABEL[g] if g else '?'})"


def to_markdown(m: EpisodeMetrics | GroupMetrics) -> str:
    if isinstance(m, GroupMetrics):
        name = ", ".join(f"{k}={v}" for k, v in m.group.items()) or "corpus"
        head = [f"# Explicómetro — {name}", "", f"{m.n_episodes} episodios · {m.duration_s / 3600:.1f} h"]
    else:
        head = [
            f"# Explicómetro — {m.episode_id}",
            "",
            f"Duración: {m.duration_s:.1f} s · Habla: {m.speech_s:.1f} s · Hablantes: {m.n_speakers}"
            + (" · **dry-run**" if m.dry_run else ""),
        ]
    lines = [*head, "", "## A. Presencia (solo H y M)", ""]
    lines += ["| Género | Hablantes | Tiempo | Palabras |", "|---|---|---|---|"]
    for gen, s in m.presence.by_gender.items():
        lines.append(f"| {_LABEL[gen]} | {_pct(s.speakers)} | {_pct(s.time)} | {_pct(s.words)} |")
    if not m.presence.by_gender:
        lines.append("| (sin datos) | — | — | — |")
    u = m.unknown
    lines += [
        "",
        f"Fuera de las métricas por género (U): {u.speakers} hablantes, {u.time_s:.1f} s "
        f"({_pct(u.time_share)} del tiempo de habla, {_pct(u.words_share)} de las palabras).",
        "",
        "## B. Protagonismo",
        "",
        "| Género | Turnos | Duración media (s) | Asentimientos |",
        "|---|---|---|---|",
    ]
    p = m.protagonism
    for gen in (Gender.MALE, Gender.FEMALE):
        mean = _num(p.mean_turn_s.get(gen), 1)
        lines.append(f"| {_LABEL[gen]} | {p.n_turns.get(gen, 0)} | {mean} | {p.n_backchannels.get(gen, 0)} |")
    lines.append("")
    if isinstance(m, GroupMetrics):
        pa = m.protagonism
        for label, d in (("Abre", pa.openers), ("Cierra", pa.closers), ("Modera", pa.moderators)):
            lines.append(f"- {label}: " + (", ".join(f"{_LABEL[g]} {n}" for g, n in d.items()) or "—"))
    else:
        pe = m.protagonism
        lines += [
            f"- Abre: {_who(pe.opener, pe.opener_gender)} · Cierra: {_who(pe.closer, pe.closer_gender)}",
            f"- Modera: {_who(pe.moderator, pe.moderator_gender)}"
            + (f" (fuente: {pe.moderator_source})" if pe.moderator_source else ""),
        ]
    lines += ["", "## C. Conductas por díada (emisor→receptor)"]
    lines += _events("Entre participantes (cuentan en C1–C4)", m.events)
    lines += _events("Del moderador hacia invitados (aparte: moderar no es interrumpir)", m.moderator_events)
    lines += [
        "",
        "## Ratios",
        "",
        f"- Exceso de interrupción: {fmt_ratio(m.interruption_excess)}",
        f"- Ratio de asimetría (H→M / H→H): {fmt_ratio(m.asymmetry_ratio)}",
        "",
    ]
    return "\n".join(lines)


def _markdown(m: EpisodeMetrics, cfg: StageCfg) -> str:
    return to_markdown(m)


BACKENDS = {"markdown": _markdown, "stub": _markdown}


def run(m: EpisodeMetrics, cfg: StageCfg) -> str:
    return resolve(BACKENDS, cfg.backend, "report")(m, cfg)
