"""gender: prioridad metadata > clasificador > unknown, voz limpia, entrenamiento sin fugas.

Todo con datos sintéticos: el embedder ECAPA se sustituye por uno falso basado en la
frecuencia dominante (tonos graves = H, agudos = M), así se ejercita toda la lógica real
(lectura de audio, ventanas, promedio, LogisticRegression, umbral) sin descargar modelos.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import tarfile
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, S, T
from explicometro import gender
from explicometro.config import RuntimeCfg, StageCfg, load_config
from explicometro.metrics import (
    presence_from,
    speakers_by_gender,
    speech_by_gender,
    unknown_from,
    words_by_gender,
)
from explicometro.schemas import AudioMeta, Gender, GenderSource, Segment

pytest.importorskip("sklearn")
from explicometro import gender_training as gt

SR = 16000
CPU = RuntimeCfg(device="cpu", compute_type="float32")
H, M, U = Gender.MALE, Gender.FEMALE, Gender.UNKNOWN


def fake_embed(clips: list[np.ndarray]) -> np.ndarray:
    """'Embedding' = frecuencia dominante (+ dimensiones constantes)."""
    out = []
    for c in clips:
        spec = np.abs(np.fft.rfft(c * np.hanning(len(c))))
        f = np.fft.rfftfreq(len(c), 1 / SR)[spec.argmax()]
        out.append([f / 100.0, 1.0, 0.5])
    return gender.l2norm(np.array(out))


def tone_wav(path: Path, plan: list[tuple[float, float, float]], total_s: float) -> None:
    """WAV con tonos ``(inicio, fin, frecuencia)``."""
    x = np.zeros(int(total_s * SR), dtype=np.float32)
    for a, b, f in plan:
        t = np.arange(int(a * SR), int(b * SR)) / SR
        x[int(a * SR) : int(a * SR) + len(t)] += 0.3 * np.sin(2 * np.pi * f * t)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """Clasificador entrenado con tonos: H ~ 90–150 Hz, M ~ 190–260 Hz."""
    rng = np.random.default_rng(0)
    fh, fm = rng.uniform(90, 150, 200), rng.uniform(190, 260, 200)
    X = gender.l2norm(np.array([[f / 100, 1.0, 0.5] for f in np.r_[fh, fm]]))
    y = np.array(["H"] * 200 + ["M"] * 200)
    b = {"model": gt.train(X, y), "embedding_model": "fake/ecapa"}
    path = tmp_path_factory.mktemp("clf") / "gender_lr.joblib"
    import joblib

    joblib.dump(b, path)
    return {"bundle": b, "path": path}


# ----------------------------------------------------------------------------- metadata


def write_csv(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["spk_id,nombre_publico,genero_publico", *(",".join(r) for r in rows)]
    path.write_text("﻿" + "\n".join(lines) + "\n", encoding="utf-8")  # con BOM, como Excel
    return path


def test_metadata_parsing(tmp_path):
    p = write_csv(
        tmp_path / "s.csv",
        [
            ("SPK_00", "Ana Pérez", "Mujer"),
            ("SPK_01", "", ""),
            ("SPK_02", "Alex", "no binario"),
            ("SPK_03", "Joan", "H"),
        ],
    )
    meta = gender.load_metadata(p)
    assert meta["SPK_00"] == gender.MetaRow("Ana Pérez", M)
    assert meta["SPK_01"] == gender.MetaRow(None, None)  # vacío → clasificador
    assert meta["SPK_02"].gender is U  # U explícito: no se clasifica por voz
    assert meta["SPK_03"].gender is H


@pytest.mark.parametrize(
    ("rows", "msg"),
    [
        ([("SPK_00", "a", "M"), ("SPK_00", "b", "H")], "repetido"),
        ([("SPK_00", "a", "quizá")], "no reconocido"),
    ],
)
def test_metadata_errors(tmp_path, rows, msg):
    with pytest.raises(ValueError, match=msg):
        gender.load_metadata(write_csv(tmp_path / "s.csv", rows))


def test_metadata_missing_columns(tmp_path):
    (tmp_path / "s.csv").write_text("spk,genero\nSPK_00,M\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columnas"):
        gender.load_metadata(tmp_path / "s.csv")


# --------------------------------------------------------------------------- voz limpia


def test_clean_intervals_remove_overlaps_with_guard():
    diar = [
        Segment(start=0, end=10, speaker="SPK_00"),
        Segment(start=4, end=5, speaker="SPK_01"),
        Segment(start=9, end=12, speaker="SPK_01"),
        Segment(start=11, end=20, speaker="SPK_00"),
    ]
    clean = gender.clean_intervals(diar, "SPK_00", guard_s=0.1)
    assert clean == [(0, 3.9), (5.1, 8.9), (12.1, 20)]


def test_windows_cap_and_min_length():
    w = gender.windows([(0, 10.5), (20, 20.5)], window_s=4, min_window_s=1, max_windows=100)
    assert w == [(0, 4), (4, 8), (8, 10.5)]
    many = gender.windows([(0, 1000)], window_s=4, min_window_s=1, max_windows=10)
    assert len(many) == 10 and many[0][0] == 0 and many[-1][0] == 996


# ------------------------------------------------------------------------- clasificador


@pytest.fixture
def episode(tmp_path):
    """SPK_00 grave 40 s · SPK_01 agudo 40 s · SPK_02 agudo 20 s · SPK_03 ambiguo 40 s ·
    con un solape SPK_00/SPK_01 en 38–42 s."""
    plan = [(0, 40, 120), (38, 80, 230), (80, 100, 230), (100, 140, 170)]
    wav = tmp_path / "audio.wav"
    tone_wav(wav, plan, 141)
    diar = [Segment(start=a, end=b, speaker=f"SPK_0{i}") for i, (a, b, _f) in enumerate(plan)]
    meta = AudioMeta(
        episode_id="ep1",
        path="audio.wav",
        sample_rate=SR,
        channels=1,
        duration_s=141,
        sha256="0" * 64,
        source_sha256="0" * 64,
    )
    return wav, diar, meta


def test_classify_thresholds(episode, bundle):
    wav, diar, _ = episode
    p = gender.GenderParams()
    res = gender.classify(
        wav, diar, ["SPK_00", "SPK_01", "SPK_02", "SPK_03"], p, fake_embed, bundle["bundle"]
    )
    assert res["SPK_00"][0] is H and res["SPK_00"][1] >= 0.8
    assert res["SPK_01"][0] is M
    assert res["SPK_00"][2] == pytest.approx(37.9, abs=0.01)  # 40 s menos el solape y la guarda
    assert res["SPK_02"] == (U, 0.0, pytest.approx(20.0, abs=0.3))  # < 30 s de voz limpia
    assert res["SPK_03"][0] is U and 0 < res["SPK_03"][1] < 0.8  # ambiguo → U


def test_load_classifier_checks_embedding_model(bundle, tmp_path):
    assert gender.load_classifier(bundle["path"], "fake/ecapa")["model"] is not None
    with pytest.raises(Exception, match="se entrenó con"):
        gender.load_classifier(bundle["path"], "speechbrain/spkrec-ecapa-voxceleb")
    with pytest.raises(Exception, match="train_gender"):
        gender.load_classifier(tmp_path / "no.joblib", "x")


def test_priority_metadata_over_classifier_and_privacy(episode, bundle):
    wav, diar, _ = episode
    meta = {
        "SPK_00": gender.MetaRow("Persona Pública", M),  # metadata gana al clasificador (H)
        "SPK_01": gender.MetaRow("Otra", None),  # vacío → clasificador
        "SPK_03": gender.MetaRow(None, U),
    }  # U explícito → sin clasificar
    pending = ["SPK_01", "SPK_02"]
    classified = gender.classify(wav, diar, pending, gender.GenderParams(), fake_embed, bundle["bundle"])
    sp = {s.id: s for s in gender.build_speakers(diar, meta, classified, use_public_names=False)}
    assert (sp["SPK_00"].gender, sp["SPK_00"].gender_source, sp["SPK_00"].gender_conf) == (
        M,
        GenderSource.METADATA,
        1.0,
    )
    assert (sp["SPK_01"].gender, sp["SPK_01"].gender_source) == (M, GenderSource.MODEL)
    assert (sp["SPK_02"].gender, sp["SPK_02"].gender_source) == (U, GenderSource.NONE)
    assert (sp["SPK_03"].gender, sp["SPK_03"].gender_source) == (U, GenderSource.METADATA)
    assert all(s.public_name is None for s in sp.values())  # privacidad por defecto
    named = gender.build_speakers(diar, meta, classified, use_public_names=True)
    assert named[0].public_name == "Persona Pública"


def test_run_stub_uses_metadata_and_pipeline_cache(episode, tmp_path):
    wav, diar, meta = episode
    ann = tmp_path / "ann"
    write_csv(ann / "speakers_ep1.csv", [("SPK_01", "", "M")])
    out = gender.run(meta, wav, diar, StageCfg(backend="stub"), CPU, annotations_dir=ann)
    by = {s.id: s for s in out}
    assert by["SPK_01"].gender is M and by["SPK_00"].gender is U and by["SPK_00"].clean_s > 30

    from explicometro.pipeline import Pipeline

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={"annotations": ann})})
    k1 = Pipeline(cfg=cfg, episode_id="ep1", source=wav).key("gender")
    write_csv(ann / "speakers_ep1.csv", [("SPK_01", "", "H")])
    assert Pipeline(cfg=cfg, episode_id="ep1", source=wav).key("gender") != k1  # editar CSV invalida


# ------------------------------------------------------------------------------ métricas


def test_metrics_exclude_unknown_and_report_it():
    spk = [S("SPK_00", H), S("SPK_01", M), S("SPK_02", U)]
    turns = [
        T(0, "SPK_00", 0, 30, "a " * 30),
        T(1, "SPK_01", 30, 60, "b " * 30),
        T(2, "SPK_02", 60, 100, "c " * 40),
    ]
    stats = speakers_by_gender(spk), speech_by_gender(turns, spk), words_by_gender(turns, spk)
    p = presence_from(*stats)
    assert set(p.by_gender) == {H, M}
    assert p.by_gender[M].time == pytest.approx(0.5) and p.by_gender[M].speakers == pytest.approx(0.5)
    u = unknown_from(*stats)
    assert (u.speakers, u.time_s) == (1, 40.0)
    assert u.time_share == pytest.approx(0.4) and u.words_share == pytest.approx(0.4)


# --------------------------------------------------------------------------- entrenamiento

TSV_HEADER = "\t".join(
    [
        "client_id",
        "path",
        "sentence_id",
        "sentence",
        "sentence_domain",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accents",
        "variant",
        "locale",
        "segment",
    ]
)


def cv_rows(locale: str, n_spk: int, clips: int, gender_value: str, prefix: str) -> list[str]:
    sentence = "Dijo \"hola\" y 'adiós'"  # comillas sin escapar, como en Common Voice
    return [
        "\t".join(
            [
                f"{prefix}{s}",
                f"common_voice_{locale}_{prefix}{s}_{c}.mp3",
                "x",
                sentence,
                "",
                "2",
                "0",
                "",
                gender_value,
                "",
                "",
                locale,
                "",
            ]
        )
        for s in range(n_spk)
        for c in range(clips)
    ]


def test_parse_cv_tsv_maps_and_filters(tmp_path):
    rows = [
        *cv_rows("es", 2, 2, "male_masculine", "a"),
        *cv_rows("es", 2, 2, "female_feminine", "b"),
        *cv_rows("es", 1, 2, "non-binary", "c"),
        *cv_rows("es", 1, 2, "", "d"),
        *cv_rows("es", 1, 1, "female", "e"),
    ]
    df = gt.parse_cv_tsv(("\n".join([TSV_HEADER, *rows]) + "\n").encode(), "es")
    assert df["gender"].value_counts().to_dict() == {"H": 4, "M": 5}
    assert set(df.columns) == {"client_id", "path", "gender", "locale"}


def test_select_balanced_and_speaker_split():
    import pandas as pd

    rows = []
    for loc in ("es", "ca"):
        for g, pre in (("H", "h"), ("M", "m")):
            for s in range(40):
                rows += [
                    {"client_id": f"{pre}{s}", "path": f"{loc}_{pre}{s}_{c}.mp3", "gender": g, "locale": loc}
                    for c in range(20 if s < 2 else 3)
                ]
    df = pd.DataFrame(rows)  # h0, h1... graban en es y ca (bilingües) y dos son muy prolíficos
    sel = gt.select_balanced(df, per_class=200, max_per_speaker=5)
    counts = sel.groupby(["gender", "locale"]).size().to_dict()
    assert counts == {("H", "ca"): 100, ("H", "es"): 100, ("M", "ca"): 100, ("M", "es"): 100}
    assert sel.groupby(["locale", "client_id"]).size().max() <= 5
    split = gt.speaker_split(sel, test_frac=0.2)
    train_ids = set(split.loc[split.split == "train", "client_id"])
    test_ids = set(split.loc[split.split == "test", "client_id"])
    assert train_ids and test_ids and not (train_ids & test_ids)


def make_cv_archive(
    root: Path, locale: str, plan: dict[str, tuple[str, float]], clips_per_spk: int = 3
) -> Path:
    """Archivo .tar.gz con estructura de Common Voice; cada hablante = un tono."""
    base = root / f"cv-corpus-25.0/{locale}"
    (base / "clips").mkdir(parents=True)
    rows = []
    for spk, (g, f0) in plan.items():
        for c in range(clips_per_spk):
            name = f"common_voice_{locale}_{spk}_{c}.mp3"
            f = f0 * (1 + 0.02 * c)
            subprocess.run(
                [
                    "ffmpeg",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    f"sine=frequency={f}:duration=2",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(base / "clips" / name),
                ],
                check=True,
            )
            rows.append(f"{spk}\t{name}\tx\tfrase\t\t2\t0\t\t{g}\t\t\t{locale}\t")
    (base / "validated.tsv").write_text("\n".join([TSV_HEADER, *rows]) + "\n", encoding="utf-8")
    (base / "other.tsv").write_text(TSV_HEADER + "\n", encoding="utf-8")
    archive = root / f"cv_{locale}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root / "cv-corpus-25.0", arcname="cv-corpus-25.0")
    shutil.rmtree(root / "cv-corpus-25.0")
    return archive


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg no instalado")
def test_train_gender_script_end_to_end(tmp_path):
    rng = np.random.default_rng(1)
    archives = {}
    for loc in ("es", "ca"):
        plan = {f"{loc}h{i}": ("male_masculine", float(rng.uniform(90, 150))) for i in range(10)}
        plan |= {f"{loc}m{i}": ("female_feminine", float(rng.uniform(190, 260))) for i in range(10)}
        plan |= {f"{loc}x0": ("do_not_wish_to_say", 170.0)}
        archives[loc] = make_cv_archive(tmp_path / loc, loc, plan)
    cfg = tmp_path / "gt.yaml"
    base = (ROOT / "configs" / "gender_train.yaml").read_text(encoding="utf-8")
    cfg.write_text(
        base.replace("archive: null", "archive: ARCH", 2)
        .replace("ARCH", str(archives["es"]), 1)
        .replace("ARCH", str(archives["ca"]), 1)
        .replace("data/processed/gender_cv", str(tmp_path / "work"))
        .replace("models/gender_lr", str(tmp_path / "models" / "gender_lr"))
        .replace("per_class: 2000", "per_class: 40"),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("train_gender", ROOT / "scripts" / "train_gender.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_gender"] = mod
    spec.loader.exec_module(mod)
    assert mod.main(["--config", str(cfg)], embedder_factory=lambda c: fake_embed) == 0

    import json

    import joblib

    rep = json.loads((tmp_path / "models" / "gender_lr.report.json").read_text())
    assert rep["test"]["speaker"]["accuracy"] == 1.0
    assert set(rep["test"]["clip"]["f1"]) == {"H", "M"}
    assert len(rep["test"]["clip"]["confusion"]["matrix"]) == 2
    # Solo se extrajeron los clips elegidos (el hablante sin género declarado, no)
    extracted = {p.name for p in (tmp_path / "work" / "clips").rglob("*.mp3")}
    assert extracted and not any("x0" in n for n in extracted)
    b = joblib.load(tmp_path / "models" / "gender_lr.joblib")
    assert b["embedding_model"] == "speechbrain/spkrec-ecapa-voxceleb"
    # Segunda ejecución: embeddings desde caché
    calls = []
    assert (
        mod.main(
            ["--config", str(cfg)],
            embedder_factory=lambda c: lambda clips: calls.append(1) or fake_embed(clips),
        )
        == 0
    )
    assert calls == []


def test_metadata_role_column(tmp_path, episode):
    _wav, diar, _ = episode
    p = tmp_path / "s.csv"
    p.write_text(
        "spk_id,nombre_publico,genero_publico,rol\nSPK_00,,M,Moderadora\nSPK_01,,,invitado\n",
        encoding="utf-8",
    )
    meta = gender.load_metadata(p)
    assert (meta["SPK_00"].role, meta["SPK_01"].role) == ("moderator", "guest")
    by = {s.id: s for s in gender.build_speakers(diar, meta, {}, use_public_names=False)}
    assert by["SPK_00"].role == "moderator" and by["SPK_02"].role is None
    p.write_text("spk_id,nombre_publico,genero_publico,rol\nSPK_00,,M,jefa\n", encoding="utf-8")
    with pytest.raises(ValueError, match="rol no reconocido"):
        gender.load_metadata(p)
