"""Diarización: Segment con speaker. Backend real: pyannote 3.1 (+ segmentación RTVE)."""

from __future__ import annotations

from pathlib import Path

from .backends import not_implemented, resolve
from .config import RuntimeCfg, StageCfg
from .schemas import AudioMeta, Segment


def relabel_speakers(segments: list[Segment]) -> list[Segment]:
    """Anonimiza etiquetas a SPK_00, SPK_01... por orden de primera aparición."""
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    mapping: dict[str, str] = {}
    out = []
    for s in ordered:
        raw = s.speaker or "?"
        mapping.setdefault(raw, f"SPK_{len(mapping):02d}")
        out.append(s.model_copy(update={"speaker": mapping[raw]}))
    return out


def _stub(audio: AudioMeta, wav: Path, speech: list[Segment], cfg: StageCfg, rt: RuntimeCfg) -> list[Segment]:
    return []


def _pyannote(
    audio: AudioMeta, wav: Path, speech: list[Segment], cfg: StageCfg, rt: RuntimeCfg
) -> list[Segment]:
    raise not_implemented(
        "diarize", "pyannote", "Instala extras [gpu] y define HF_TOKEN en .env (fp16, sin bf16)."
    )


BACKENDS = {"stub": _stub, "pyannote": _pyannote}


def run(audio: AudioMeta, wav: Path, speech: list[Segment], cfg: StageCfg, rt: RuntimeCfg) -> list[Segment]:
    segs = resolve(BACKENDS, cfg.backend, "diarize")(audio, wav, speech, cfg, rt)
    return relabel_speakers(segs)
