"""Ingesta: normaliza el audio a WAV 16 kHz mono PCM16 y opcionalmente lo recorta."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import wave
from pathlib import Path

from .backends import BackendError
from .config import StageCfg
from .schemas import AudioMeta

AUDIO_NAME = "audio.wav"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _is_target_wav(path: Path, sample_rate: int) -> bool:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate() == sample_rate and w.getnchannels() == 1 and w.getsampwidth() == 2
    except (wave.Error, EOFError):
        return False


def _copy_wav(src: Path, dst: Path, max_seconds: float | None) -> None:
    with wave.open(str(src), "rb") as r:
        n = r.getnframes()
        if max_seconds is not None:
            n = min(n, int(max_seconds * r.getframerate()))
        frames = r.readframes(n)
        params = r.getparams()
    with wave.open(str(dst), "wb") as w:
        w.setparams(params)
        w.writeframes(frames)


def _ffmpeg(src: Path, dst: Path, sample_rate: int, max_seconds: float | None) -> None:
    if shutil.which("ffmpeg") is None:
        raise BackendError("ffmpeg no está instalado y el audio no es WAV 16 kHz mono PCM16")
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if max_seconds is not None:
        cmd += ["-t", f"{max_seconds}"]
    cmd += ["-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def run(
    src: Path, workdir: Path, episode_id: str, cfg: StageCfg, max_seconds: float | None = None
) -> AudioMeta:
    """Escribe ``workdir/audio.wav`` y devuelve sus metadatos."""
    sample_rate = int(cfg.params.get("sample_rate", 16000))
    workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / AUDIO_NAME
    backend = cfg.backend
    if backend == "auto":
        backend = "wave" if _is_target_wav(src, sample_rate) else "ffmpeg"
    if backend == "wave":
        _copy_wav(src, dst, max_seconds)
    elif backend == "ffmpeg":
        _ffmpeg(src, dst, sample_rate, max_seconds)
    else:
        raise BackendError(f"Backend de ingest desconocido: {cfg.backend}")
    with wave.open(str(dst), "rb") as w:
        duration = w.getnframes() / w.getframerate()
        sr, ch = w.getframerate(), w.getnchannels()
    return AudioMeta(
        episode_id=episode_id,
        path=AUDIO_NAME,
        sample_rate=sr,
        channels=ch,
        duration_s=round(duration, 3),
        sha256=sha256_file(dst),
        source_sha256=sha256_file(src),
        truncated_to_s=max_seconds,
    )
