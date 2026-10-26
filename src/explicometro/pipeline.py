"""Orquestación con caché por artefacto en ``data/interim/<episodio>/``.

Cada etapa:
- lee solo artefactos de sus dependencias (tipos de ``schemas``),
- escribe un único artefacto,
- se salta si el artefacto existe y su clave de caché coincide (config de la etapa +
  claves de sus dependencias). ``force`` la recalcula.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import align, asr, diarize, events, gender, ingest, llm, metrics, report, turns, vad
from . import io as aio
from .config import Config
from .schemas import AudioMeta, EpisodeMetrics, Event, Segment, Speaker, Turn, Word

log = logging.getLogger(__name__)

STAGES: tuple[str, ...] = (
    "ingest", "vad", "diarize", "asr", "align", "gender", "turns", "llm", "events", "metrics", "report",
)  # fmt: skip

DEPS: dict[str, tuple[str, ...]] = {
    "ingest": (),
    "vad": ("ingest",),
    "diarize": ("ingest", "vad"),
    "asr": ("ingest", "vad"),
    "align": ("asr", "diarize"),
    "gender": ("ingest", "diarize"),
    "turns": ("align",),
    "llm": ("turns", "gender"),
    "events": ("turns", "gender", "llm"),
    "metrics": ("ingest", "gender", "turns", "events"),
    "report": ("metrics",),
}


@dataclass(frozen=True)
class Artifact:
    filename: str
    save: Callable[[Path, Any, str], None]
    load: Callable[[Path], Any]


def _json_list(filename: str, cls: type) -> Artifact:
    return Artifact(filename, lambda p, v, _: aio.write_models(p, v), lambda p: aio.read_models(p, cls))


ARTIFACTS: dict[str, Artifact] = {
    "ingest": Artifact(
        "audio.json", lambda p, v, _: aio.write_model(p, v), lambda p: aio.read_model(p, AudioMeta)
    ),
    "vad": _json_list("vad.json", Segment),
    "diarize": Artifact("diarization.rttm", aio.write_rttm, aio.read_rttm),
    "asr": _json_list("asr_words.json", Word),
    "align": _json_list("transcript.json", Word),
    "gender": _json_list("speakers.json", Speaker),
    "turns": Artifact(
        "turns.parquet",
        lambda p, v, _: aio.write_parquet(p, v, Turn),
        lambda p: aio.read_parquet(p, Turn),
    ),
    "llm": _json_list("llm_events.json", Event),
    "events": Artifact(
        "events.parquet",
        lambda p, v, _: aio.write_parquet(p, v, Event),
        lambda p: aio.read_parquet(p, Event),
    ),
    "metrics": Artifact(
        "metrics.json",
        lambda p, v, _: aio.write_model(p, v),
        lambda p: aio.read_model(p, EpisodeMetrics),
    ),
    "report": Artifact("report.md", lambda p, v, _: aio.write_text(p, v), lambda p: p.read_text()),
}


def upstream(targets: Iterable[str]) -> list[str]:
    """Targets + todas sus dependencias, en orden de ejecución."""
    needed: set[str] = set()

    def visit(s: str) -> None:
        if s not in DEPS:
            raise ValueError(f"Etapa desconocida: {s}. Válidas: {', '.join(STAGES)}")
        if s not in needed:
            needed.add(s)
            for d in DEPS[s]:
                visit(d)

    for t in targets:
        visit(t)
    return [s for s in STAGES if s in needed]


@dataclass
class Pipeline:
    cfg: Config
    episode_id: str
    source: Path | None = None
    dry_run: bool = False
    _values: dict[str, Any] = field(default_factory=dict, init=False)
    _keys: dict[str, str] = field(default_factory=dict, init=False)

    @property
    def workdir(self) -> Path:
        return self.cfg.paths.interim / self.episode_id

    def artifact_path(self, stage: str) -> Path:
        return self.workdir / ARTIFACTS[stage].filename

    def _key_file(self, stage: str) -> Path:
        return self.workdir / ".cache" / f"{stage}.key"

    def key(self, stage: str) -> str:
        if stage in self._keys:
            return self._keys[stage]
        parts: list[Any] = [
            stage,
            getattr(self.cfg.stages, stage).model_dump(mode="json"),
            self.cfg.runtime.model_dump(mode="json"),
            [self.key(d) for d in DEPS[stage]],
        ]
        if stage in ("events", "metrics"):
            parts.append(self.cfg.thresholds.model_dump(mode="json"))
        if stage == "ingest":
            if self.source is None:
                kf = self._key_file(stage)
                if not kf.exists():
                    raise FileNotFoundError("Falta --audio: no hay ingest previa para este episodio")
                self._keys[stage] = kf.read_text().strip()
                return self._keys[stage]
            st = self.source.stat()
            parts += [str(self.source.resolve()), st.st_size, st.st_mtime_ns, self._max_seconds()]
        self._keys[stage] = self.cfg.digest(*parts)
        return self._keys[stage]

    def _max_seconds(self) -> float | None:
        return self.cfg.dry_run.max_seconds if self.dry_run else None

    def _cached(self, stage: str) -> bool:
        kf = self._key_file(stage)
        return (
            self.artifact_path(stage).exists() and kf.exists() and kf.read_text().strip() == self.key(stage)
        )

    def _compute(self, stage: str) -> Any:
        c, rt, th, v = self.cfg, self.cfg.runtime, self.cfg.thresholds, self._values
        st = getattr(c.stages, stage)
        wav = self.workdir / ingest.AUDIO_NAME
        match stage:
            case "ingest":
                if self.source is None:
                    raise FileNotFoundError("Falta --audio para la etapa ingest")
                return ingest.run(self.source, self.workdir, self.episode_id, st, self._max_seconds())
            case "vad":
                return vad.run(v["ingest"], wav, st, rt)
            case "diarize":
                return diarize.run(v["ingest"], wav, v["vad"], st, rt)
            case "asr":
                return asr.run(v["ingest"], wav, v["vad"], st, rt)
            case "align":
                return align.run(v["asr"], v["diarize"], st)
            case "gender":
                return gender.run(v["ingest"], wav, v["diarize"], st, rt)
            case "turns":
                return turns.run(v["align"], st)
            case "llm":
                return llm.run(v["turns"], v["gender"], st, rt)
            case "events":
                return events.run(v["turns"], v["gender"], v["llm"], st, th, rt)
            case "metrics":
                return metrics.run(
                    v["ingest"], v["gender"], v["turns"], v["events"], st, th,
                    dry_run=self.dry_run, config_hash=self.key("metrics"),
                )  # fmt: skip
            case "report":
                return report.run(v["metrics"], st)
        raise ValueError(stage)

    def run(self, targets: Iterable[str] | None = None, force: Iterable[str] = ()) -> dict[str, Any]:
        """Ejecuta ``targets`` (por defecto, todas) y lo que necesiten. Devuelve los valores."""
        targets = list(targets or STAGES)
        force = set(force)
        self.workdir.mkdir(parents=True, exist_ok=True)
        for stage in upstream(targets):
            art, path = ARTIFACTS[stage], self.artifact_path(stage)
            if stage not in force and self._cached(stage):
                log.info("[%s] caché → %s", stage, path.name)
                self._values[stage] = art.load(path)
                continue
            log.info("[%s] ejecutando (backend=%s)", stage, getattr(self.cfg.stages, stage).backend)
            value = self._compute(stage)
            art.save(path, value, self.episode_id)
            self._values[stage] = art.load(path)  # round-trip: valida lo escrito contra el esquema
            kf = self._key_file(stage)
            kf.parent.mkdir(exist_ok=True)
            kf.write_text(self.key(stage) + "\n")
        return dict(self._values)
