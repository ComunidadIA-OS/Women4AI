"""VAD: regiones de voz (``Segment`` sin speaker) con Silero VAD.

Backend ``silero``: el modelo Silero en ONNX que viene empaquetado con faster-whisper
(``faster_whisper.vad``). Corre en CPU, no descarga nada y no necesita torch, así que
también funciona en máquinas sin GPU. Se procesa por trozos (``runtime.chunk_s``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field

from .audio import read_window
from .backends import BackendError, resolve
from .chunking import merge_segments, plan_chunks
from .config import Params, RuntimeCfg, StageCfg
from .schemas import AudioMeta, Segment

log = logging.getLogger(__name__)


class VadParams(Params):
    threshold: float = Field(default=0.5, gt=0, lt=1)
    min_speech_ms: int = Field(default=250, ge=0)
    min_silence_ms: int = Field(default=100, ge=0)
    speech_pad_ms: int = Field(default=30, ge=0)
    max_speech_s: float = Field(default=float("inf"), gt=0)


def _stub(audio: AudioMeta, wav: Path, p: VadParams, rt: RuntimeCfg) -> list[Segment]:
    return []


def _silero(audio: AudioMeta, wav: Path, p: VadParams, rt: RuntimeCfg) -> list[Segment]:
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except ImportError as e:  # pragma: no cover - depende del entorno
        raise BackendError("Silero VAD necesita faster-whisper: pip install -e '.[audio]'") from e

    opts = VadOptions(
        threshold=p.threshold,
        min_speech_duration_ms=p.min_speech_ms,
        min_silence_duration_ms=p.min_silence_ms,
        speech_pad_ms=p.speech_pad_ms,
        max_speech_duration_s=p.max_speech_s,
    )
    per_chunk = []
    for chunk in plan_chunks(audio.duration_s, rt.chunk_s, rt.chunk_margin_s):
        samples, sr = read_window(wav, chunk.start, chunk.end)
        stamps = get_speech_timestamps(samples, vad_options=opts, sampling_rate=sr)
        segs = [Segment(start=chunk.start + t["start"] / sr, end=chunk.start + t["end"] / sr) for t in stamps]
        per_chunk.append((chunk, segs))
        log.debug("vad trozo %d: %d segmentos", chunk.index, len(segs))
    return merge_segments(per_chunk, join_gap_s=p.min_silence_ms / 1000)


BACKENDS = {"stub": _stub, "silero": _silero}


def run(audio: AudioMeta, wav: Path, cfg: StageCfg, rt: RuntimeCfg) -> list[Segment]:
    segs = resolve(BACKENDS, cfg.backend, "vad")(audio, wav, cfg.parsed(VadParams), rt)
    return [s.model_copy(update={"start": round(s.start, 3), "end": round(s.end, 3)}) for s in segs]
