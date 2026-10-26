"""Género por hablante. Backend real: SpeechBrain ECAPA + LogisticRegression.

Por debajo de ``min_conf`` la etiqueta es U. Nunca se infiere género a partir del nombre.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .backends import not_implemented, resolve
from .config import RuntimeCfg, StageCfg
from .schemas import AudioMeta, Segment, Speaker


def speaking_time(diar: list[Segment]) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for s in diar:
        if s.speaker is not None:
            total[s.speaker] += s.duration
    return dict(total)


def _stub(audio: AudioMeta, wav: Path, diar: list[Segment], cfg: StageCfg, rt: RuntimeCfg) -> list[Speaker]:
    """Todos los hablantes con género desconocido."""
    return [Speaker(id=spk, total_s=round(t, 3)) for spk, t in speaking_time(diar).items()]


def _ecapa_logreg(
    audio: AudioMeta, wav: Path, diar: list[Segment], cfg: StageCfg, rt: RuntimeCfg
) -> list[Speaker]:
    raise not_implemented(
        "gender", "ecapa_logreg", "Instala extras [gpu] y entrena el clasificador (classifier_path)."
    )


BACKENDS = {"stub": _stub, "ecapa_logreg": _ecapa_logreg}


def run(audio: AudioMeta, wav: Path, diar: list[Segment], cfg: StageCfg, rt: RuntimeCfg) -> list[Speaker]:
    speakers = resolve(BACKENDS, cfg.backend, "gender")(audio, wav, diar, cfg, rt)
    return sorted(speakers, key=lambda s: s.id)
