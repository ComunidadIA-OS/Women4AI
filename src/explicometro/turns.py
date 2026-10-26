"""Construcción de turnos a partir de palabras alineadas (determinista)."""

from __future__ import annotations

from .backends import resolve
from .config import StageCfg
from .schemas import Turn, Word


def build_turns(words: list[Word], max_gap_s: float = 0.75) -> list[Turn]:
    """Agrupa palabras consecutivas del mismo hablante separadas por <= ``max_gap_s``.

    Las palabras sin hablante se descartan. Los turnos de hablantes distintos pueden
    solaparse (habla simultánea) porque se construyen por hablante.
    """
    by_spk: dict[str, list[list[Word]]] = {}
    for w in sorted((w for w in words if w.speaker), key=lambda w: w.start):
        groups = by_spk.setdefault(w.speaker, [])  # type: ignore[arg-type]
        if groups and w.start - groups[-1][-1].end <= max_gap_s:
            groups[-1].append(w)
        else:
            groups.append([w])
    chunks = sorted((g for gs in by_spk.values() for g in gs), key=lambda g: (g[0].start, g[-1].end))
    return [
        Turn(
            id=i,
            speaker=g[0].speaker,  # type: ignore[arg-type]
            start=g[0].start,
            end=max(w.end for w in g),
            text=" ".join(w.text for w in g),
            n_words=len(g),
        )
        for i, g in enumerate(chunks)
    ]


def _words_backend(words: list[Word], cfg: StageCfg) -> list[Turn]:
    return build_turns(words, float(cfg.params.get("max_gap_s", 0.75)))


BACKENDS = {"words": _words_backend, "stub": _words_backend}


def run(words: list[Word], cfg: StageCfg) -> list[Turn]:
    return resolve(BACKENDS, cfg.backend, "turns")(words, cfg)
