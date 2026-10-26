from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from explicometro.schemas import Gender, GenderSource, Speaker, Turn

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def clip_120s() -> Path:
    """Fixture de audio sintético; se genera si no existe (no se versiona audio)."""
    path = FIXTURES / "clip_120s.wav"
    if not path.exists():
        mf = _load_script("make_fixture")
        mf.write_wav(path, mf.synth(120.0))
    return path


def T(i: int, spk: str, start: float, end: float, text: str = "bla bla bla bla", bc: bool = False) -> Turn:
    return Turn(
        id=i, speaker=spk, start=start, end=end, text=text, n_words=len(text.split()), is_backchannel=bc
    )


def S(spk: str, gender: Gender) -> Speaker:
    src = GenderSource.NONE if gender is Gender.UNKNOWN else GenderSource.MANUAL
    return Speaker(
        id=spk, gender=gender, gender_source=src, gender_conf=1.0 if src != GenderSource.NONE else 0.0
    )


def has_flite() -> bool:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return " flite " in out


@pytest.fixture(scope="session")
def tts_dialogue(tmp_path_factory):
    """Diálogo sintético de dos voces con verdad conocida (flite; ver make_tts_fixture.py).

    Si existe tests/fixtures/tts_dialogue/ generado con StyleTTS2, se usa ese.
    """
    import json

    from explicometro import io as aio

    existing = FIXTURES / "tts_dialogue"
    if (existing / "truth.json").exists() and (existing / "dialogue.wav").exists():
        d = existing
    else:
        if not has_flite():
            pytest.skip("ffmpeg sin libflite: genera el diálogo con scripts/make_tts_fixture.py")
        d = tmp_path_factory.mktemp("tts_dialogue")
        _load_script("make_tts_fixture").build(d)
    meta = json.loads((d / "truth.json").read_text(encoding="utf-8"))
    return {"dir": d, "wav": d / "dialogue.wav", "rttm": aio.read_rttm(d / "truth.rttm"), "meta": meta}
