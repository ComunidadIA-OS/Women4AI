"""Ingesta: ffmpeg → WAV 16 kHz mono PCM16 con normalización de volumen.

Normalización por defecto: ``loudnorm`` EBU R128 en dos pasadas y modo **lineal** (una
ganancia constante para todo el episodio). La pasada única de ``loudnorm`` es dinámica y
comprime: subiría el volumen de voces de fondo y alteraría la relación entre hablantes
que se solapan, justo lo que queremos medir.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Literal

from pydantic import Field

from .backends import BackendError, resolve
from .config import Params, StageCfg
from .schemas import AudioMeta

log = logging.getLogger(__name__)
AUDIO_NAME = "audio.wav"
DOWNMIX = "aformat=channel_layouts=mono"


class IngestParams(Params):
    sample_rate: int = 16000
    normalize: Literal["loudnorm_2pass", "loudnorm", "none"] = "loudnorm_2pass"
    target_lufs: float = Field(default=-23.0, le=-5, ge=-70)
    true_peak_db: float = Field(default=-2.0, le=0)
    lra: float = Field(default=11.0, gt=0)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise BackendError("ffmpeg no está instalado (apt install ffmpeg)")
    return exe


def _input_args(src: Path, max_seconds: float | None) -> list[str]:
    args = [_ffmpeg_bin(), "-nostdin", "-hide_banner", "-y", "-i", str(src)]
    if max_seconds is not None:
        args += ["-t", f"{max_seconds}"]
    return args


def measure_loudness(src: Path, p: IngestParams, max_seconds: float | None) -> dict[str, float]:
    """Primera pasada de loudnorm: mide el audio original y devuelve el JSON de ffmpeg."""
    flt = f"{DOWNMIX},loudnorm=I={p.target_lufs}:TP={p.true_peak_db}:LRA={p.lra}:print_format=json"
    cmd = [*_input_args(src, max_seconds), "-vn", "-af", flt, "-f", "null", "-"]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", res.stderr, re.S)
    if match is None:
        raise BackendError("No se pudo leer la medida de loudnorm de ffmpeg")
    return {k: float(v) for k, v in json.loads(match.group(0)).items() if k.startswith(("input", "target"))}


def loudnorm_filter(p: IngestParams, measured: dict[str, float] | None) -> str:
    base = f"loudnorm=I={p.target_lufs}:TP={p.true_peak_db}:LRA={p.lra}"
    if measured is None:
        return base
    # Silencio digital: input_i = -inf → ffmpeg no puede normalizar; se deja tal cual.
    if measured["input_i"] == float("-inf") or measured["input_i"] < -70:
        return "anull"
    return (
        f"{base}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


def convert(src: Path, dst: Path, p: IngestParams, max_seconds: float | None) -> None:
    filters: list[str] = [DOWNMIX]  # mezclar a mono ANTES de medir: en estéreo la medida sale +3 dB
    if p.normalize == "loudnorm_2pass":
        filters.append(loudnorm_filter(p, measure_loudness(src, p, max_seconds)))
    elif p.normalize == "loudnorm":
        filters.append(loudnorm_filter(p, None))
    filters.append(f"aresample={p.sample_rate}")  # loudnorm trabaja a 192 kHz internamente
    cmd = [
        *_input_args(src, max_seconds),
        "-vn", "-loglevel", "error", "-af", ",".join(filters),
        "-ac", "1", "-ar", str(p.sample_rate), "-c:a", "pcm_s16le", str(dst),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)


def _ffmpeg(src: Path, workdir: Path, episode_id: str, p: IngestParams, max_s: float | None) -> AudioMeta:
    dst = workdir / AUDIO_NAME
    tmp = workdir / f".{AUDIO_NAME}.tmp.wav"
    convert(src, tmp, p, max_s)
    tmp.replace(dst)
    with wave.open(str(dst), "rb") as w:
        duration, sr, ch = w.getnframes() / w.getframerate(), w.getframerate(), w.getnchannels()
    return AudioMeta(
        episode_id=episode_id,
        path=AUDIO_NAME,
        sample_rate=sr,
        channels=ch,
        duration_s=round(duration, 3),
        sha256=sha256_file(dst),
        source_sha256=sha256_file(src),
        truncated_to_s=max_s,
    )


BACKENDS = {"ffmpeg": _ffmpeg}


def run(
    src: Path, workdir: Path, episode_id: str, cfg: StageCfg, max_seconds: float | None = None
) -> AudioMeta:
    """Escribe ``workdir/audio.wav`` (16 kHz mono PCM16 normalizado) y devuelve sus metadatos."""
    if not src.exists():
        raise FileNotFoundError(src)
    workdir.mkdir(parents=True, exist_ok=True)
    params = cfg.parsed(IngestParams)
    return resolve(BACKENDS, cfg.backend, "ingest")(src, workdir, episode_id, params, max_seconds)
