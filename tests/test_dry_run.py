"""Criterio de éxito: dry-run en CPU sobre clip_120s.wav -> metrics.json vacío pero válido."""

import wave

import pytest

from conftest import ROOT
from explicometro.backends import BackendError
from explicometro.config import load_config
from explicometro.pipeline import ARTIFACTS, STAGES, Pipeline
from explicometro.schemas import EpisodeMetrics


@pytest.fixture
def cfg(tmp_path):
    c = load_config(ROOT / "configs" / "default.yaml", dry_run=True)
    paths = c.paths.model_copy(update={"interim": tmp_path / "interim"})
    return c.model_copy(update={"paths": paths})


def test_dry_run_config_is_cpu_and_stub(cfg):
    assert cfg.runtime.device == "cpu"
    assert cfg.stages.diarize.backend == "stub" and cfg.stages.asr.backend == "stub"
    assert cfg.stages.turns.backend == "words"  # determinista: se mantiene


def test_dry_run_end_to_end(cfg, clip_120s):
    pipe = Pipeline(cfg=cfg, episode_id="clip_120s", source=clip_120s, dry_run=True)
    values = pipe.run()
    for stage in STAGES:
        assert pipe.artifact_path(stage).exists(), stage
    m = EpisodeMetrics.model_validate_json((pipe.workdir / "metrics.json").read_text())
    assert m.dry_run and m.episode_id == "clip_120s"
    assert m.duration_s == pytest.approx(120.0, abs=0.01)
    assert m.n_speakers == 0 and m.events == {} and m.presence.by_gender == {}
    assert values["metrics"] == m
    with wave.open(str(pipe.workdir / "audio.wav")) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1)


def test_cache_and_single_stage_rerun(cfg, clip_120s):
    Pipeline(cfg=cfg, episode_id="ep", source=clip_120s, dry_run=True).run()
    metrics_path = cfg.paths.interim / "ep" / ARTIFACTS["metrics"].filename
    mtime = metrics_path.stat().st_mtime_ns
    # Re-ejecutar sin --audio: todo sale de caché
    Pipeline(cfg=cfg, episode_id="ep", source=None, dry_run=True).run(["metrics"])
    assert metrics_path.stat().st_mtime_ns == mtime
    # Forzar solo metrics la reescribe
    Pipeline(cfg=cfg, episode_id="ep", source=None, dry_run=True).run(["metrics"], force=["metrics"])
    assert metrics_path.stat().st_mtime_ns != mtime


def test_real_backends_fail_loudly(tmp_path, clip_120s):
    c = load_config(ROOT / "configs" / "default.yaml")
    c = c.model_copy(update={"paths": c.paths.model_copy(update={"interim": tmp_path})})
    with pytest.raises(BackendError, match="pendiente"):
        Pipeline(cfg=c, episode_id="ep", source=clip_120s).run(["vad"])


def test_max_seconds_truncates(cfg, tmp_path):
    from conftest import _load_script

    mf = _load_script("make_fixture")
    long = tmp_path / "long.wav"
    mf.write_wav(long, mf.synth(130.0))
    pipe = Pipeline(cfg=cfg, episode_id="long", source=long, dry_run=True)
    meta = pipe.run(["ingest"])["ingest"]
    assert meta.duration_s == pytest.approx(120.0) and meta.truncated_to_s == 120.0
