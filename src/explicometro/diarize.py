"""Diarización con pyannote/speaker-diarization-3.1. Conserva los solapes.

- ``use_rtve_segmentation: true`` sustituye el modelo de segmentación por
  HiTZ/pyannote-segmentation-3.0-RTVE manteniendo embeddings e hiperparámetros de 3.1.
- No se trocea el audio: con trozos independientes las etiquetas de hablante no serían
  coherentes entre trozos. pyannote ya usa ventanas deslizantes internas; la memoria de GPU
  se controla con ``segmentation_batch_size`` y ``embedding_batch_size``.
- Precisión fp32 (pyannote no soporta fp16 de forma fiable; bf16 está prohibido en V100S).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pydantic import Field

from .backends import BackendError, resolve
from .config import Params, RuntimeCfg, StageCfg, env
from .schemas import AudioMeta, Segment

log = logging.getLogger(__name__)


class DiarizeParams(Params):
    pipeline: str = "pyannote/speaker-diarization-3.1"
    use_rtve_segmentation: bool = False
    rtve_segmentation: str = "HiTZ/pyannote-segmentation-3.0-RTVE"
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)
    num_speakers: int | None = Field(default=None, ge=1)
    segmentation_batch_size: int = Field(default=32, ge=1)
    embedding_batch_size: int = Field(default=32, ge=1)
    device_index: int = Field(default=0, ge=0)
    min_segment_s: float = Field(default=0.0, ge=0, description="Descarta segmentos más cortos")


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


def tracks_to_segments(tracks: Iterable[tuple[Any, Any, Any]], min_segment_s: float = 0.0) -> list[Segment]:
    """Convierte ``Annotation.itertracks(yield_label=True)`` en segmentos SIN aplanar solapes.

    Cada pista es un segmento propio; dos hablantes a la vez dan dos segmentos solapados.
    Las etiquetas originales se conservan aquí y se anonimizan en ``run``.
    """
    out = []
    for turn, _track, label in tracks:
        if turn.end - turn.start >= max(min_segment_s, 1e-3):
            out.append(
                Segment.model_construct(
                    start=round(turn.start, 3), end=round(turn.end, 3), speaker=str(label)
                )
            )
    return out


def _speaker_kwargs(p: DiarizeParams) -> dict[str, int]:
    if p.num_speakers is not None:
        return {"num_speakers": p.num_speakers}
    kw = {}
    if p.min_speakers is not None:
        kw["min_speakers"] = p.min_speakers
    if p.max_speakers is not None:
        kw["max_speakers"] = p.max_speakers
    if kw.get("min_speakers", 0) > kw.get("max_speakers", 10**9):
        raise ValueError("min_speakers > max_speakers")
    return kw


def load_pipeline(p: DiarizeParams, rt: RuntimeCfg) -> Any:  # pragma: no cover - requiere GPU/HF
    try:
        import torch
        from pyannote.audio import Model, Pipeline
        from pyannote.audio.pipelines import SpeakerDiarization
    except ImportError as e:
        raise BackendError("pyannote necesita los extras de GPU: pip install -e '.[gpu]'") from e

    token = env("HF_TOKEN")
    base = Pipeline.from_pretrained(p.pipeline, use_auth_token=token)
    if base is None:
        raise BackendError(f"No se pudo cargar {p.pipeline}: acepta sus condiciones en HF y define HF_TOKEN")
    if p.use_rtve_segmentation:
        seg_model = Model.from_pretrained(p.rtve_segmentation, use_auth_token=token)
        pipe = SpeakerDiarization(
            segmentation=seg_model,
            embedding=base.embedding,
            embedding_exclude_overlap=base.embedding_exclude_overlap,
            clustering=base.klustering,
            segmentation_batch_size=p.segmentation_batch_size,
            embedding_batch_size=p.embedding_batch_size,
            use_auth_token=token,
        )
        pipe.instantiate(base.parameters(instantiated=True))
        log.info("diarize: segmentación sustituida por %s", p.rtve_segmentation)
    else:
        pipe = base
        pipe.segmentation_batch_size = p.segmentation_batch_size
        pipe.embedding_batch_size = p.embedding_batch_size
    if rt.device == "cuda":
        pipe.to(torch.device(f"cuda:{p.device_index}"))
    return pipe


def _stub(
    audio: AudioMeta, wav: Path, speech: list[Segment], p: DiarizeParams, rt: RuntimeCfg
) -> list[Segment]:
    return []


def _pyannote(
    audio: AudioMeta, wav: Path, speech: list[Segment], p: DiarizeParams, rt: RuntimeCfg
) -> list[Segment]:  # pragma: no cover - requiere GPU/HF
    from .audio import read_window

    pipe = load_pipeline(p, rt)
    import torch

    samples, sr = read_window(wav, 0.0, audio.duration_s)
    waveform = torch.from_numpy(samples).unsqueeze(0)
    try:
        annotation = pipe({"waveform": waveform, "sample_rate": sr}, **_speaker_kwargs(p))
        return tracks_to_segments(annotation.itertracks(yield_label=True), p.min_segment_s)
    finally:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


BACKENDS = {"stub": _stub, "pyannote": _pyannote}


def run(audio: AudioMeta, wav: Path, speech: list[Segment], cfg: StageCfg, rt: RuntimeCfg) -> list[Segment]:
    p = cfg.parsed(DiarizeParams)
    _speaker_kwargs(p)  # valida min/max antes de cargar modelos
    segs = resolve(BACKENDS, cfg.backend, "diarize")(audio, wav, speech, p, rt)
    return [Segment.model_validate(s.model_dump()) for s in relabel_speakers(segs)]
