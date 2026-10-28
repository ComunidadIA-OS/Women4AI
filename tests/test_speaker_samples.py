"""scripts/speaker_samples.py: agrupación de palabras en intervenciones por hablante."""

from conftest import _load_script


def w(start: float, text: str, spk: str) -> dict:
    return {"start": start, "end": start + 0.3, "text": text, "speaker": spk}


def test_utterances_split_by_speaker_and_gap():
    ss = _load_script("speaker_samples")
    words = [
        w(0.0, "hola", "SPK_00"),
        w(0.4, "a", "SPK_00"),
        w(0.8, "todos", "SPK_00"),
        w(1.2, "gracias", "SPK_01"),
        w(5.0, "otra", "SPK_01"),  # hueco > 1 s → nueva intervención
        w(5.4, "vez", "SPK_01"),
    ]
    out = ss.utterances(words)
    assert out["SPK_00"] == [(0.0, "hola a todos")]
    assert out["SPK_01"] == [(1.2, "gracias"), (5.0, "otra vez")]


def test_utterances_ignore_words_without_speaker():
    ss = _load_script("speaker_samples")
    out = ss.utterances([{"start": 0.0, "end": 0.2, "text": "eh", "speaker": None}, w(0.3, "sí", "SPK_00")])
    assert set(out) == {"SPK_00"}
