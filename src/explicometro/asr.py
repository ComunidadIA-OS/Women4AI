"""ASR con timestamps por palabra, por trozos (``runtime.chunk_s`` ± ``chunk_margin_s``).

Backends intercambiables (``stages.asr.backend`` + ``params.model``):

=====================  ==========================================  ==========================
backend                model                                       notas
=====================  ==========================================  ==========================
``faster_whisper``     ``large-v3-turbo``                          multilingüe, rápido
``faster_whisper``     ``BSC-LT/faster-whisper-bsc-large-v3-cat``  catalán (CTranslate2)
``transformers``       ``BSC-LT/whisper-large-v3-LoS-punctuated``  es/ca con puntuación
=====================  ==========================================  ==========================

Por cada trozo: se lee solo su ventana, se silencia lo que el VAD marca como no-voz (evita
alucinaciones sin mover timestamps), se transcribe, se pasan los tiempos a absolutos y se
fusiona conservando solo las palabras del núcleo del trozo (``chunking.merge_words``).
El modelo se carga una vez por etapa y se libera al terminar.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from .audio import mute_outside, read_window
from .backends import BackendError, resolve
from .chunking import merge_words, offset_words, plan_chunks
from .config import Params, RuntimeCfg, StageCfg
from .schemas import AudioMeta, Segment, Word

log = logging.getLogger(__name__)

KNOWN_MODELS = {
    "large-v3-turbo": "faster_whisper",
    "turbo": "faster_whisper",
    "large-v3": "faster_whisper",
    "BSC-LT/faster-whisper-bsc-large-v3-cat": "faster_whisper",
    "BSC-LT/whisper-large-v3-LoS-punctuated": "transformers",
}


class AsrParams(Params):
    model: str = "large-v3-turbo"
    language: Literal["es", "ca"] | None = "es"
    compute_type: Literal["float16", "int8_float16", "int8", "float32"] | None = Field(
        default=None, description="None = runtime.compute_type. V100S: float16 o int8_float16"
    )
    device_index: int = Field(default=0, ge=0)
    beam_size: int = Field(default=5, ge=1)
    condition_on_previous_text: bool = False
    initial_prompt: str | None = None
    hallucination_silence_threshold: float | None = None
    vad_pad_s: float = Field(default=0.2, ge=0, description="Holgura alrededor de la voz del VAD")
    batch_size: int = Field(default=8, ge=1, description="Solo backend transformers")
    alignment_heads_from: str | None = Field(
        default="openai/whisper-large-v3",
        description="transformers: copia alignment_heads si el modelo no los trae (timestamps por palabra)",
    )


@dataclass
class Transcriber:
    """Transcribe una ventana (tiempos relativos a la ventana) y libera recursos al cerrar."""

    transcribe: Callable[[np.ndarray, int], list[Word]]
    close: Callable[[], None] = lambda: None


def effective_compute_type(p: AsrParams, rt: RuntimeCfg, backend: str) -> str:
    ct = p.compute_type or rt.compute_type
    if rt.device == "cpu" and ct in ("float16", "int8_float16"):
        fallback = "int8" if backend == "faster_whisper" else "float32"
        log.warning("asr: %s no está soportado en CPU; uso %s", ct, fallback)
        return fallback
    if backend == "transformers" and ct in ("int8", "int8_float16"):
        log.warning("asr: transformers no soporta %s aquí; uso float16", ct)
        return "float16" if rt.device == "cuda" else "float32"
    return ct


def check_model_backend(model: str, backend: str) -> None:
    expected = KNOWN_MODELS.get(model)
    if expected is not None and backend not in ("stub", expected):
        raise BackendError(f"El modelo {model} va con backend '{expected}', no '{backend}'")


def _clean(text: str) -> str:
    return text.strip()


# ------------------------------------------------------------------------------ backends


def _stub_factory(p: AsrParams, rt: RuntimeCfg) -> Transcriber:
    return Transcriber(lambda samples, sr: [])


def _faster_whisper_factory(p: AsrParams, rt: RuntimeCfg) -> Transcriber:  # pragma: no cover - GPU
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise BackendError("Instala faster-whisper: pip install -e '.[audio]'") from e

    ct = effective_compute_type(p, rt, "faster_whisper")
    model = WhisperModel(p.model, device=rt.device, device_index=p.device_index, compute_type=ct)
    log.info("asr: faster-whisper %s (%s, %s)", p.model, rt.device, ct)

    def transcribe(samples: np.ndarray, sr: int) -> list[Word]:
        segments, _info = model.transcribe(
            samples,
            language=p.language,
            task="transcribe",
            beam_size=p.beam_size,
            word_timestamps=True,
            vad_filter=False,  # ya silenciamos con nuestro VAD
            condition_on_previous_text=p.condition_on_previous_text,
            initial_prompt=p.initial_prompt,
            hallucination_silence_threshold=p.hallucination_silence_threshold,
        )
        out = []
        for seg in segments:  # generador: la decodificación ocurre aquí
            for w in seg.words or []:
                text = _clean(w.word)
                if text and w.end >= w.start:
                    out.append(
                        Word(start=max(0.0, w.start), end=max(0.0, w.end), text=text, prob=w.probability)
                    )
        return out

    def close() -> None:
        nonlocal model
        del model

    return Transcriber(transcribe, close)


def _transformers_factory(p: AsrParams, rt: RuntimeCfg) -> Transcriber:  # pragma: no cover - GPU
    try:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig, pipeline
    except ImportError as e:
        raise BackendError("Instala torch y transformers: pip install -e '.[gpu]'") from e

    ct = effective_compute_type(p, rt, "transformers")
    dtype = torch.float16 if ct == "float16" else torch.float32
    device = f"cuda:{p.device_index}" if rt.device == "cuda" else "cpu"
    # eager: las word timestamps necesitan las atenciones cruzadas (y nada de flash-attention).
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        p.model, torch_dtype=dtype, low_cpu_mem_usage=True, attn_implementation="eager"
    )
    if not getattr(model.generation_config, "alignment_heads", None):
        if p.alignment_heads_from is None:
            raise BackendError(f"{p.model} no trae alignment_heads; define alignment_heads_from")
        ref = GenerationConfig.from_pretrained(p.alignment_heads_from)
        model.generation_config.alignment_heads = ref.alignment_heads
        log.info("asr: alignment_heads copiados de %s", p.alignment_heads_from)
    proc = AutoProcessor.from_pretrained(p.model)
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        batch_size=p.batch_size,
    )
    log.info("asr: transformers %s (%s, %s)", p.model, device, ct)
    gen_kwargs = {"task": "transcribe", "num_beams": p.beam_size}
    if p.language:
        gen_kwargs["language"] = p.language

    def transcribe(samples: np.ndarray, sr: int) -> list[Word]:
        out = asr_pipe(
            {"raw": samples, "sampling_rate": sr}, return_timestamps="word", generate_kwargs=gen_kwargs
        )
        words = []
        for c in out.get("chunks", []):
            text = _clean(c["text"])
            start, end = c["timestamp"]
            if not text or start is None:
                continue
            end = start + 0.3 if end is None or end < start else end
            words.append(Word(start=float(start), end=float(end), text=text))
        return words

    def close() -> None:
        nonlocal model, asr_pipe
        del asr_pipe, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return Transcriber(transcribe, close)


BACKENDS: dict[str, Callable[[AsrParams, RuntimeCfg], Transcriber]] = {
    "stub": _stub_factory,
    "faster_whisper": _faster_whisper_factory,
    "transformers": _transformers_factory,
}


# ------------------------------------------------------------------------------ troceado


def in_speech(w: Word, speech: list[Segment], pad_s: float) -> bool:
    mid = (w.start + w.end) / 2
    return any(s.start - pad_s <= mid <= s.end + pad_s for s in speech)


def transcribe_chunked(
    wav: Path, duration: float, speech: list[Segment], tr: Transcriber, rt: RuntimeCfg, pad_s: float
) -> list[Word]:
    """Bucle de trozos común a todos los backends. Función testeable con un Transcriber falso."""
    chunks = plan_chunks(duration, rt.chunk_s, rt.chunk_margin_s, speech)
    per_chunk = []
    for chunk in chunks:
        local = [s for s in speech if s.end > chunk.start and s.start < chunk.end]
        if not local:
            per_chunk.append((chunk, []))
            continue
        t0 = time.perf_counter()
        samples, sr = read_window(wav, chunk.start, chunk.end)
        samples = mute_outside(samples, sr, chunk.start, local, pad_s)
        words = offset_words(tr.transcribe(samples, sr), chunk.start)
        words = [w for w in words if in_speech(w, local, pad_s)]
        per_chunk.append((chunk, words))
        dt = time.perf_counter() - t0
        log.info(
            "asr trozo %d/%d [%.0f–%.0f s]: %d palabras en %.1f s",
            chunk.index + 1, len(chunks), chunk.core_start, chunk.core_end, len(words), dt,
        )  # fmt: skip
    return merge_words(per_chunk)


def run(audio: AudioMeta, wav: Path, speech: list[Segment], cfg: StageCfg, rt: RuntimeCfg) -> list[Word]:
    p = cfg.parsed(AsrParams)
    check_model_backend(p.model, cfg.backend)
    factory = resolve(BACKENDS, cfg.backend, "asr")
    if not speech:
        log.warning("asr: el VAD no encontró voz; no se transcribe nada")
        return []
    tr = factory(p, rt)
    try:
        words = transcribe_chunked(wav, audio.duration_s, speech, tr, rt, p.vad_pad_s)
    finally:
        tr.close()
    return [w.model_copy(update={"start": round(w.start, 3), "end": round(w.end, 3)}) for w in words]
