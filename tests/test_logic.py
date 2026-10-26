"""Casos sintéticos para la lógica determinista (align, turns, C1, métricas, evaluación)."""

import pytest

from conftest import S, T
from explicometro.align import assign_speakers
from explicometro.config import InterruptionCfg, ProtagonismCfg
from explicometro.diarize import relabel_speakers
from explicometro.evaluation import score_events
from explicometro.events import detect_interruptions
from explicometro.metrics import (
    asymmetry_ratio,
    event_stats,
    interruption_excess,
    presence,
    protagonism,
)
from explicometro.schemas import Dyad, Event, EventType, Gender, Segment, Word
from explicometro.turns import build_turns

C1 = InterruptionCfg(backchannel_lexicon=["sí", "claro", "vale", "mhm"])
H, M = Gender.MALE, Gender.FEMALE


def test_relabel_speakers_by_first_appearance():
    segs = [Segment(start=5, end=6, speaker="SPK_07"), Segment(start=0, end=1, speaker="SPK_03")]
    out = relabel_speakers(segs)
    assert [s.speaker for s in out] == ["SPK_00", "SPK_01"]


def test_align_and_turns():
    diar = [Segment(start=0, end=3, speaker="SPK_00"), Segment(start=3, end=6, speaker="SPK_01")]
    words = [
        Word(start=0.1, end=0.5, text="hola"),
        Word(start=0.6, end=1.0, text="qué"),
        Word(start=3.2, end=3.6, text="bien"),
        Word(start=5.0, end=5.4, text="vale"),
    ]
    aligned = assign_speakers(words, diar)
    assert [w.speaker for w in aligned] == ["SPK_00", "SPK_00", "SPK_01", "SPK_01"]
    turns = build_turns(aligned, max_gap_s=0.75)
    assert [(t.speaker, t.text) for t in turns] == [
        ("SPK_00", "hola qué"),
        ("SPK_01", "bien"),
        ("SPK_01", "vale"),
    ]


def test_c1_competitive_interruption():
    # A habla 0–10; B entra en 9.0 (solape 1.0 s) y A cede en 1.0 s <= 2 s.
    turns = [T(0, "SPK_00", 0, 10), T(1, "SPK_01", 9.0, 15)]
    ev = detect_interruptions(turns, C1)
    assert len(ev) == 1 and ev[0].src_spk == "SPK_01" and ev[0].dst_spk == "SPK_00"


@pytest.mark.parametrize(
    "b",
    [
        T(1, "SPK_01", 9.8, 15),  # solape 0.2 s < 0.5 s
        T(1, "SPK_01", 5.0, 15),  # A sigue 5 s: no cede en <= 2 s
        T(1, "SPK_01", 9.0, 9.6, "sí"),  # asentimiento corto
        T(1, "SPK_01", 9.0, 11.0, "sí claro vale"),  # léxico de backchannel
    ],
)
def test_c1_negatives(b):
    assert detect_interruptions([T(0, "SPK_00", 0, 10), b], C1) == []


def test_presence_and_protagonism():
    spk = [S("SPK_00", H), S("SPK_01", M)]
    turns = [T(0, "SPK_00", 0, 30, "a " * 30), T(1, "SPK_01", 30, 40, "b " * 10), T(2, "SPK_00", 40, 52)]
    p = presence(turns, spk)
    assert p.by_gender[H].speakers == 0.5
    assert p.by_gender[M].time == pytest.approx(10 / 52)
    pr = protagonism(turns, spk, ProtagonismCfg())
    assert pr.n_turns == {H: 2, M: 1}
    assert pr.opener == "SPK_00" and pr.closer == "SPK_00" and pr.opener_gender is H


def test_event_rates_excess_and_asymmetry():
    spk = [S("SPK_00", H), S("SPK_01", M), S("SPK_02", H)]
    # 120 s de habla de M (2 min) y 120 s de H (2 min)
    turns = [T(0, "SPK_01", 0, 120), T(1, "SPK_00", 120, 180), T(2, "SPK_02", 180, 240)]
    ev = [
        Event(type=EventType.INTERRUPTION, src_spk="SPK_00", dst_spk="SPK_01", t_start=1, t_end=2),
        Event(type=EventType.INTERRUPTION, src_spk="SPK_02", dst_spk="SPK_01", t_start=3, t_end=4),
        Event(type=EventType.INTERRUPTION, src_spk="SPK_00", dst_spk="SPK_02", t_start=5, t_end=6),
    ]
    stats = event_stats(ev, turns, spk)
    assert stats[EventType.INTERRUPTION][Dyad.H_M].count == 2
    assert stats[EventType.INTERRUPTION][Dyad.H_M].rate_per_min == pytest.approx(1.0)
    assert stats[EventType.INTERRUPTION][Dyad.H_H].rate_per_min == pytest.approx(0.5)
    assert asymmetry_ratio(stats) == pytest.approx(2.0)
    # M recibe 2/3 de las interrupciones con 1/2 del tiempo -> 4/3
    assert interruption_excess(ev, turns, spk) == pytest.approx(4 / 3)


def test_metrics_empty_inputs():
    assert presence([], []).by_gender == {}
    assert event_stats([], [], []) == {}
    assert interruption_excess([], [], []) is None
    assert asymmetry_ratio({}) is None


def test_score_events():
    def e(t):
        return Event(type=EventType.INTERRUPTION, src_spk="SPK_00", dst_spk="SPK_01", t_start=t, t_end=t + 1)

    s = score_events([e(10), e(50)], [e(10.4), e(30)], tol_s=1.0)[EventType.INTERRUPTION]
    assert (s.tp, s.fp, s.fn) == (1, 1, 1)
    assert s.precision == 0.5 and s.recall == 0.5
