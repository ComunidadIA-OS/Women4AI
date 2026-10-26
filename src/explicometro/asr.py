"""ASR con timestamps por palabra (Word sin speaker).

Backends reales: faster-whisper (large-v3) o BSC-LT/whisper-large-v3-LoS-punctuated.
"""

from __future__ import annotations

from pathlib import Path

from .backends import not_implemented, resolve
from .config import RuntimeCfg, StageCfg
from .schemas import AudioMeta, Segment, Word


def _stub(audio: AudioMeta, wav: Path, speech: list[Segment], cfg: StageCfg, rt: RuntimeCfg) -> list[Word]:
    return []


def _faster_whisper(
    audio: AudioMeta, wav: Path, speech: list[Segment], cfg: StageCfg, rt: RuntimeCfg
) -> list[Word]:
    raise not_implemented(
        "asr", "faster_whisper", "Instala extras [gpu]; compute_type float16/int8 en V100S."
    )


BACKENDS = {"stub": _stub, "faster_whisper": _faster_whisper}


def run(audio: AudioMeta, wav: Path, speech: list[Segment], cfg: StageCfg, rt: RuntimeCfg) -> list[Word]:
    words = resolve(BACKENDS, cfg.backend, "asr")(audio, wav, speech, cfg, rt)
    return sorted(words, key=lambda w: w.start)
