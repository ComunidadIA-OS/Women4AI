"""Casos sintéticos para diarización, alineación + turnos y evaluación de eventos."""

from explicometro.align import assign_speakers
from explicometro.diarize import relabel_speakers
from explicometro.evaluation import score_events
from explicometro.schemas import Event, EventType, Segment, Word
from explicometro.turns import build_turns


def test_relabel_speakers_by_first_appearance():
    segs = [Segment(start=5, end=6, speaker="SPK_07"), Segment(start=0, end=1, speaker="SPK_03")]
    out = relabel_speakers(segs)
    assert [s.speaker for s in out] == ["SPK_00", "SPK_01"]


def test_align_then_turns():
    diar = [Segment(start=0, end=3, speaker="SPK_00"), Segment(start=3, end=6, speaker="SPK_01")]
    words = [
        Word(start=0.1, end=0.5, text="hola"),
        Word(start=0.6, end=1.0, text="qué"),
        Word(start=3.2, end=3.6, text="bien"),
        Word(start=5.0, end=5.4, text="vale"),
    ]
    aligned = assign_speakers(words, diar)
    assert [w.speaker for w in aligned] == ["SPK_00", "SPK_00", "SPK_01", "SPK_01"]
    turns = build_turns(diar, aligned)
    assert [(t.speaker, t.text, t.is_backchannel) for t in turns] == [
        ("SPK_00", "hola qué", False),
        ("SPK_01", "bien vale", False),
    ]


def test_score_events():
    def e(t):
        return Event(type=EventType.INTERRUPTION, src_spk="SPK_00", dst_spk="SPK_01", t_start=t, t_end=t + 1)

    s = score_events([e(10), e(50)], [e(10.4), e(30)], tol_s=1.0)[EventType.INTERRUPTION]
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    assert s.precision == 0.5 and s.recall == 0.5
