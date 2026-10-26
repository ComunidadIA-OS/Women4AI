"""Configuración tipada. Todo sale de YAML (configs/default.yaml) + secretos de .env."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

# V100S: sin bf16 ni flash-attention. Solo estos tipos de cómputo son válidos.
ComputeType = Literal["float32", "float16", "int8", "int8_float16"]


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsCfg(_Cfg):
    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    annotations: Path = Path("data/annotations")


class RuntimeCfg(_Cfg):
    device: Literal["cpu", "cuda"] = "cuda"
    compute_type: ComputeType = "float16"
    seed: int = 13
    chunk_s: float = Field(default=600.0, gt=0, description="Trozos para VAD/ASR en episodios largos")
    chunk_margin_s: float = Field(default=5.0, ge=0, description="Margen a cada lado de cada trozo")


class StageCfg(_Cfg):
    backend: str = "stub"
    params: dict[str, Any] = Field(default_factory=dict)

    def parsed(self, model: type[P]) -> P:
        """Valida ``params`` contra el modelo tipado de la etapa (rechaza claves desconocidas)."""
        return model.model_validate(self.params)


class Params(BaseModel):
    """Base para los parámetros tipados de cada etapa."""

    model_config = ConfigDict(extra="forbid")


P = TypeVar("P", bound=Params)


class StagesCfg(_Cfg):
    ingest: StageCfg = StageCfg(backend="ffmpeg")
    vad: StageCfg = StageCfg(backend="silero")
    diarize: StageCfg = StageCfg(backend="pyannote")
    asr: StageCfg = StageCfg(backend="faster_whisper")
    align: StageCfg = StageCfg(backend="overlap")
    gender: StageCfg = StageCfg(backend="ecapa_logreg")
    turns: StageCfg = StageCfg(backend="segments")
    llm: StageCfg = StageCfg(backend="openai_compatible")
    events: StageCfg = StageCfg(backend="mrbert")
    metrics: StageCfg = StageCfg(backend="default")
    report: StageCfg = StageCfg(backend="markdown")


class InterruptionCfg(_Cfg):
    """C1 — definición cerrada en CLAUDE.md. Los asentimientos los marca ``turns``."""

    min_overlap_s: float = 0.5
    yield_within_s: float = 2.0
    recover_within_s: float = Field(
        default=5.0, ge=0, description="A recupera si su siguiente turno llega <= esto tras el fin de B"
    )


class ModeratorCfg(_Cfg):
    """Heurística de moderador (la metadata ``rol`` tiene prioridad)."""

    handover_max_s: float = Field(default=15.0, gt=0, description="Turno 'corto' que cede la palabra")
    handover_gap_s: float = Field(default=2.0, ge=0, description="Otro hablante entra <= esto después")
    edge_turns: int = Field(default=3, ge=1, description="Turnos de apertura y de cierre a mirar")
    min_handovers: int = Field(default=3, ge=1)
    min_margin: float = Field(
        default=1.5, ge=1.0, description="El 1º debe ceder >= este factor × lo que cede el 2º (y más que él)"
    )


class RatiosCfg(_Cfg):
    min_events: int = Field(default=10, ge=1, description="Por debajo: 'insuficiente'")
    bootstrap_iterations: int = Field(default=1000, ge=100)
    seed: int = 13
    ci: float = Field(default=0.95, gt=0, lt=1)


DEFAULT_ATTRIBUTION = [
    r"como (decía|ha dicho|apuntaba) \w+",
    r"lo que dice \w+",
    r"tiene razón",
    r"té raó",
    r"com deia",
]


class ReformulationCfg(_Cfg):
    """C2 — ``cos_threshold`` (τ) se calibra en el gold set con scripts/calibrate_c2.py.

    Mientras sea ``None``, C2 no emite eventos (y lo avisa): nada de umbrales a ojo.
    """

    cos_threshold: float | None = Field(default=None, ge=-1, le=1)
    window_prev_turns: int = Field(default=5, ge=1)
    min_words: int = Field(default=15, ge=1, description="Solo turnos con >= estas palabras")
    attribution_patterns: list[str] = Field(default_factory=lambda: list(DEFAULT_ATTRIBUTION))


class ProtagonismCfg(_Cfg):
    min_open_close_turn_s: float = 10.0


class ThresholdsCfg(_Cfg):
    interruption: InterruptionCfg = InterruptionCfg()
    reformulation: ReformulationCfg = ReformulationCfg()
    protagonism: ProtagonismCfg = ProtagonismCfg()
    moderator: ModeratorCfg = ModeratorCfg()
    ratios: RatiosCfg = RatiosCfg()


class PrivacyCfg(_Cfg):
    use_public_names: bool = False


class DryRunCfg(_Cfg):
    max_seconds: float = 120.0
    device: Literal["cpu"] = "cpu"
    compute_type: ComputeType = "float32"
    backend: str = "stub"
    keep_backends: list[str] = Field(
        default_factory=lambda: ["ingest", "align", "turns", "metrics", "report"],
        description="Etapas deterministas sin modelos: mantienen su backend real en dry-run",
    )


class Config(_Cfg):
    paths: PathsCfg = PathsCfg()
    runtime: RuntimeCfg = RuntimeCfg()
    stages: StagesCfg = StagesCfg()
    thresholds: ThresholdsCfg = ThresholdsCfg()
    privacy: PrivacyCfg = PrivacyCfg()
    dry_run: DryRunCfg = DryRunCfg()

    def apply_dry_run(self) -> Config:
        """Copia con CPU, stubs para etapas con modelos y recorte a ``max_seconds``."""
        stages = self.stages.model_copy(deep=True)
        for name in StagesCfg.model_fields:
            if name not in self.dry_run.keep_backends:
                getattr(stages, name).backend = self.dry_run.backend
        runtime = self.runtime.model_copy(
            update={"device": self.dry_run.device, "compute_type": self.dry_run.compute_type}
        )
        return self.model_copy(update={"stages": stages, "runtime": runtime})

    def digest(self, *parts: Any) -> str:
        """Hash estable de partes de la config (para la caché)."""
        blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load_config(path: Path | str | None = None, *, dry_run: bool = False) -> Config:
    """Carga YAML (o defaults) y variables de ``.env``. No lee secretos al objeto Config."""
    load_dotenv(override=False)
    data: dict[str, Any] = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = Config.model_validate(data)
    return cfg.apply_dry_run() if dry_run else cfg


def env(name: str, default: str | None = None) -> str | None:
    """Acceso único a secretos/variables de entorno (HF_TOKEN, LLM_*)."""
    value = os.environ.get(name)
    return value if value else default
