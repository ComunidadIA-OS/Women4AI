"""Etapas ingest → vad → diarize → asr → align.

- CPU (aquí y en CI): ffmpeg + loudnorm, Silero VAD real, adaptadores con modelos falsos.
- GPU (``-m gpu``): pyannote + Whisper reales sobre el diálogo TTS con verdad conocida.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from conftest import ROOT
from explicometro import asr, diarize, ingest, vad
from explicometro import io as aio
from explicometro.backends import BackendError
from explicometro.chunking import plan_chunks
from explicometro.cli import main as cli_main
from explicometro.config import RuntimeCfg, StageCfg, load_config
from explicometro.evaluation import (
    overlap_recall,
    single_speaker_accuracy,
    speaker_mapping,
    speech_detection,
    wer,
)
from explicometro.schemas import AudioMeta, Segment, Word

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no instalado")
CPU = RuntimeCfg(device="cpu", compute_type="float32")


def _ingest(src: Path, workdir: Path, **params) -> AudioMeta:
    return ingest.run(src, workdir, "ep", StageCfg(backend="ffmpeg", params=params))


# ------------------------------------------------------------------------------- ingest


@needs_ffmpeg
def test_ingest_loudnorm_and_hash(tts_dialogue, tmp_path):
    quiet = tmp_path / "quiet.flac"  # otro formato, -26 dB y estéreo 44.1 kHz
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(tts_dialogue["wav"]),
         "-af", "volume=0.05", "-ac", "2", "-ar", "44100", str(quiet)],
        check=True,
    )  # fmt: skip
    meta = _ingest(quiet, tmp_path / "w")
    out = tmp_path / "w" / "audio.wav"
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
    assert meta.source_sha256 == ingest.sha256_file(quiet)
    assert meta.sha256 == ingest.sha256_file(out)
    measured = ingest.measure_loudness(out, ingest.IngestParams(), None)
    assert measured["input_i"] == pytest.approx(-23.0, abs=1.0)


@needs_ffmpeg
def test_ingest_truncates_and_survives_silence(tmp_path):
    silent = tmp_path / "silence.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "5",
         str(silent)],
        check=True,
    )  # fmt: skip
    meta = ingest.run(silent, tmp_path / "w", "ep", StageCfg(backend="ffmpeg"), max_seconds=3.0)
    assert meta.duration_s == pytest.approx(3.0, abs=0.01) and meta.truncated_to_s == 3.0


def test_ingest_rejects_unknown_params(tmp_path):
    with pytest.raises(Exception, match="extra"):
        StageCfg(backend="ffmpeg", params={"normalise": "x"}).parsed(ingest.IngestParams)


# ---------------------------------------------------------------------------------- VAD


@needs_ffmpeg
def test_silero_vad_matches_truth(tts_dialogue, tmp_path):
    pytest.importorskip("faster_whisper")
    meta = _ingest(tts_dialogue["wav"], tmp_path)
    segs = vad.run(meta, tmp_path / "audio.wav", StageCfg(backend="silero"), CPU)
    ref = [s.model_copy(update={"speaker": None}) for s in tts_dialogue["rttm"]]
    prf = speech_detection(segs, ref, meta.duration_s)
    assert prf.precision > 0.9 and prf.recall > 0.9, prf


@needs_ffmpeg
def test_silero_vad_chunked_equals_unchunked(tts_dialogue, tmp_path):
    pytest.importorskip("faster_whisper")
    meta = _ingest(tts_dialogue["wav"], tmp_path)
    full = vad.run(meta, tmp_path / "audio.wav", StageCfg(backend="silero"), CPU)
    small = CPU.model_copy(update={"chunk_s": 12.0, "chunk_margin_s": 2.0})
    assert len(plan_chunks(meta.duration_s, 12.0, 2.0)) >= 3
    chunked = vad.run(meta, tmp_path / "audio.wav", StageCfg(backend="silero"), small)
    agree = speech_detection(chunked, full, meta.duration_s)
    assert agree.precision > 0.97 and agree.recall > 0.97, agree


# ---------------------------------------------------------------------------- diarize


def test_tracks_to_segments_preserves_overlap():
    core = pytest.importorskip("pyannote.core")
    ann = core.Annotation()
    ann[core.Segment(0.0, 10.0), "a"] = "SPEAKER_01"
    ann[core.Segment(9.0, 15.0), "b"] = "SPEAKER_00"
    ann[core.Segment(4.0, 4.6), "c"] = "SPEAKER_00"
    segs = diarize.relabel_speakers(diarize.tracks_to_segments(ann.itertracks(yield_label=True)))
    assert [(s.start, s.end, s.speaker) for s in segs] == [
        (0.0, 10.0, "SPK_00"), (4.0, 4.6, "SPK_01"), (9.0, 15.0, "SPK_01"),
    ]  # fmt: skip


def test_diarize_speaker_bounds_validated(tmp_path):
    meta = AudioMeta(episode_id="ep", path="a.wav", sample_rate=16000, channels=1, duration_s=1,
                     sha256="0" * 64, source_sha256="0" * 64)  # fmt: skip
    cfg = StageCfg(backend="stub", params={"min_speakers": 4, "max_speakers": 2})
    with pytest.raises(ValueError, match="min_speakers"):
        diarize.run(meta, tmp_path / "a.wav", [], cfg, CPU)


# -------------------------------------------------------------------------------- ASR


def test_asr_model_backend_and_compute_type():
    with pytest.raises(BackendError):
        asr.check_model_backend("BSC-LT/whisper-large-v3-LoS-punctuated", "faster_whisper")
    asr.check_model_backend("BSC-LT/faster-whisper-bsc-large-v3-cat", "faster_whisper")
    p = asr.AsrParams(compute_type="int8_float16")
    assert asr.effective_compute_type(p, RuntimeCfg(device="cuda"), "faster_whisper") == "int8_float16"
    assert asr.effective_compute_type(p, CPU, "faster_whisper") == "int8"
    assert asr.effective_compute_type(p, RuntimeCfg(device="cuda"), "transformers") == "float16"
    with pytest.raises(ValidationError):
        asr.AsrParams(compute_type="bfloat16")


def _write_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(int(seconds * 16000), dtype="<i2").tobytes())


def test_transcribe_chunked_offsets_merge_and_mute(tmp_path, monkeypatch):
    """Transcriber falso que 'oye' la verdad en su ventana: sin duplicados ni pérdidas."""
    wav = tmp_path / "a.wav"
    _write_wav(wav, 65.0)
    truth = [Word(start=t, end=t + 0.3, text=f"w{i}") for i, t in enumerate(np.arange(0.5, 64, 0.9).round(2))]
    speech = [Segment(start=0.3, end=64.6)]
    noise = Word(start=0.0, end=0.1, text="alucinación")  # fuera de la voz → se filtra
    windows: list[float] = []
    real_read = asr.read_window

    def spy(path, start, end):
        windows.append((start, end))
        return real_read(path, start, end)

    def fake(samples, sr):
        start, end = windows[-1]
        jitter = 0.03 * len(windows)
        seen = [Word(start=w.start - start + jitter, end=w.end - start + jitter, text=w.text)
                for w in truth if start <= w.start and w.end <= end]  # fmt: skip
        return [*seen, noise] if start == 0 else seen

    monkeypatch.setattr(asr, "read_window", spy)
    rt = CPU.model_copy(update={"chunk_s": 20.0, "chunk_margin_s": 5.0})
    out = asr.transcribe_chunked(wav, 65.0, speech, asr.Transcriber(fake), rt, pad_s=0.2)
    assert len(windows) >= 3
    assert [w.text for w in out] == [w.text for w in truth]


def test_asr_run_without_speech_skips_model(tmp_path):
    meta = AudioMeta(episode_id="ep", path="a.wav", sample_rate=16000, channels=1, duration_s=1,
                     sha256="0" * 64, source_sha256="0" * 64)  # fmt: skip
    assert asr.run(meta, tmp_path / "a.wav", [], StageCfg(backend="faster_whisper"), CPU) == []


# ---------------------------------------------------------------- pipeline --until align


@needs_ffmpeg
def test_cli_until_align_logs_timings(tts_dialogue, tmp_path):
    pytest.importorskip("faster_whisper")
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copy(tts_dialogue["wav"], raw / "dlg.wav")
    cfg = tmp_path / "cfg.yaml"
    base = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    interim = tmp_path / "i"
    cfg.write_text(
        base.replace("raw: data/raw", f"raw: {raw}").replace("interim: data/interim", f"interim: {interim}")
        .replace("device: cuda ", "device: cpu ")
        .replace("backend: pyannote", "backend: stub").replace("backend: faster_whisper ", "backend: stub "),
        encoding="utf-8",
    )  # fmt: skip
    assert cli_main(["run", "--episode", "dlg", "--until", "align", "--config", str(cfg)]) == 0
    wd = tmp_path / "i" / "dlg"
    for name in ("audio.json", "vad.json", "diarization.rttm", "asr_words.json", "transcript.json"):
        assert (wd / name).exists(), name
    assert not (wd / "turns.parquet").exists()
    stages = [json.loads(line)["stage"] for line in (wd / "timings.jsonl").read_text().splitlines()]
    assert stages == ["ingest", "vad", "diarize", "asr", "align"]
    log = (wd / "pipeline.log").read_text(encoding="utf-8")
    assert "[vad] terminada en" in log
    assert len(aio.read_models(wd / "vad.json", Segment)) >= 6
    # Segunda vez: todo de caché
    assert cli_main(["run", "--episode", "dlg", "--until", "align", "--config", str(cfg)]) == 0
    last = [json.loads(line) for line in (wd / "timings.jsonl").read_text().splitlines()][-5:]
    assert {r["status"] for r in last} == {"cache"}


# --------------------------------------------------------------------------------- GPU


def _gpu_ready() -> str | None:
    try:
        import torch
    except ImportError:
        return "torch no instalado"
    if not torch.cuda.is_available():
        return "sin CUDA"
    if not os.environ.get("HF_TOKEN"):
        return "falta HF_TOKEN"
    return None


@pytest.mark.gpu
@pytest.mark.parametrize("rtve", [False, True], ids=["seg-3.1", "seg-RTVE"])
def test_gpu_until_align_against_truth(tts_dialogue, tmp_path, rtve):
    reason = _gpu_ready()
    if reason:
        pytest.skip(reason)
    from explicometro.pipeline import Pipeline

    cfg = load_config(ROOT / "configs" / "default.yaml")
    stages = cfg.stages.model_copy(deep=True)
    stages.diarize.params["use_rtve_segmentation"] = rtve
    cfg = cfg.model_copy(
        update={"stages": stages, "paths": cfg.paths.model_copy(update={"interim": tmp_path})}
    )
    values = Pipeline(cfg=cfg, episode_id="dlg", source=tts_dialogue["wav"]).run(["align"])
    ref, dur = tts_dialogue["rttm"], values["ingest"].duration_s
    diar, words = values["diarize"], values["align"]

    assert len({s.speaker for s in diar}) == 2
    assert single_speaker_accuracy(diar, ref, dur) > 0.85
    assert (overlap_recall(diar, ref, dur) or 0) > 0.3, "la diarización ha aplanado los solapes"
    from explicometro.evaluation import word_speaker_accuracy

    acc = word_speaker_accuracy(words, ref, speaker_mapping(diar, ref, dur))
    assert acc is not None and acc > 0.85
    if tts_dialogue["meta"]["tts"] == "styletts2":  # flite pronuncia en inglés: WER no tiene sentido
        ref_text = " ".join(u["text"] for u in tts_dialogue["meta"]["utterances"])
        assert wer(ref_text, " ".join(w.text for w in words)) < 0.3


def test_faster_whisper_adapter_against_real_signature(monkeypatch):
    """Sin descargar modelos: un WhisperModel falso comprueba que los kwargs existen en la API real."""
    fw = pytest.importorskip("faster_whisper")
    import inspect
    from types import SimpleNamespace

    real_init = set(inspect.signature(fw.WhisperModel.__init__).parameters)
    real_transcribe = set(inspect.signature(fw.WhisperModel.transcribe).parameters)
    calls = {}

    class FakeModel:
        def __init__(self, model, **kw):
            calls["init"] = {"model_size_or_path", *kw}

        def transcribe(self, audio, **kw):
            calls["transcribe"] = {"audio", *kw}
            words = [SimpleNamespace(start=0.5, end=0.9, word=" Hola", probability=0.9),
                     SimpleNamespace(start=1.0, end=1.4, word=" mundo.", probability=0.8),
                     SimpleNamespace(start=1.4, end=1.4, word=" ", probability=0.1)]  # fmt: skip
            return iter([SimpleNamespace(words=words)]), None

    monkeypatch.setattr(fw, "WhisperModel", FakeModel)
    tr = asr.BACKENDS["faster_whisper"](asr.AsrParams(language="ca"), CPU)
    out = tr.transcribe(np.zeros(16000 * 2, dtype=np.float32), 16000)
    tr.close()
    assert calls["init"] <= real_init and calls["transcribe"] <= real_transcribe
    assert [(w.text, w.start) for w in out] == [("Hola", 0.5), ("mundo.", 1.0)]
