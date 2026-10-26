"""VAD: regiones de voz (Segment sin speaker). Backend real: Silero VAD."""

from __future__ import annotations

from pathlib import Path

from .backends import not_implemented, resolve
from .config import RuntimeCfg, StageCfg
from .schemas import AudioMeta, Segment


def _stub(audio: AudioMeta, wav: Path, cfg: StageCfg, rt: RuntimeCfg) -> list[Segment]:
    return []


def _silero(audio: AudioMeta, wav: Path, cfg: StageCfg, rt: RuntimeCfg) -> list[Segment]:
    raise not_implemented("vad", "silero", "Instala extras [gpu] (silero-vad).")


BACKENDS = {"stub": _stub, "silero": _silero}


def run(audio: AudioMeta, wav: Path, cfg: StageCfg, rt: RuntimeCfg) -> list[Segment]:
    segs = resolve(BACKENDS, cfg.backend, "vad")(audio, wav, cfg, rt)
    return sorted(segs, key=lambda s: s.start)
