"""Configuración tipada. Todo sale de YAML (configs/default.yaml) + secretos de .env."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

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


class StageCfg(_Cfg):
    backend: str = "stub"
    params: dict[str, Any] = Field(default_factory=dict)


class StagesCfg(_Cfg):
    ingest: StageCfg = StageCfg(backend="auto", params={"sample_rate": 16000})
    vad: StageCfg = StageCfg(backend="silero")
    diarize: StageCfg = StageCfg(backend="pyannote")
    asr: StageCfg = StageCfg(backend="faster_whisper")
    align: StageCfg = StageCfg(backend="overlap")
    gender: StageCfg = StageCfg(backend="ecapa_logreg")
    turns: StageCfg = StageCfg(backend="words", params={"max_gap_s": 0.75})
    llm: StageCfg = StageCfg(backend="openai_compatible")
    events: StageCfg = StageCfg(backend="mrbert")
    metrics: StageCfg = StageCfg(backend="default")
    report: StageCfg = StageCfg(backend="markdown")


class InterruptionCfg(_Cfg):
    """C1 — definición cerrada en CLAUDE.md."""

    min_overlap_s: float = 0.5
    yield_within_s: float = 2.0
    backchannel_max_s: float = 1.0
    backchannel_lexicon: list[str] = Field(default_factory=list)


class ReformulationCfg(_Cfg):
    """C2 — el umbral se calibra con anotaciones."""

    cos_threshold: float = 0.8
    window_prev_turns: int = 5
    attribution_patterns: list[str] = Field(default_factory=list)


class ProtagonismCfg(_Cfg):
    min_open_close_turn_s: float = 10.0


class ThresholdsCfg(_Cfg):
    interruption: InterruptionCfg = InterruptionCfg()
    reformulation: ReformulationCfg = ReformulationCfg()
    protagonism: ProtagonismCfg = ProtagonismCfg()


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
