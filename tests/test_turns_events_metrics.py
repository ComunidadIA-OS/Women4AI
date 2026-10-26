"""turns, events y metrics sobre transcripciones sintéticas cuyo resultado se calcula a mano.

Episodio de referencia (EP): 10 turnos, 4 hablantes.
    SPK_00 moderadora (M) · SPK_01 invitado (H) · SPK_02 invitada (M) · SPK_03 invitado (H)

    id  hablante  inicio  fin    dur   nota
    0   00        0.0     8.0    8.0   pregunta de apertura → cede a 01 (hueco 0,5 s)
    1   01        8.5     40.0   31.5
    2   00        40.5    46.0   5.5   cede a 02 (hueco 0,2 s)
    3   02        46.2    70.0   23.8
    4   01        69.0    90.0   21.0  ← C1 01→02 (H→M): solape 1,0 s, 02 cede en 1,0 s
    5   02        91.0    100.0  9.0   02 recupera el turno (91 ≤ 90 + 5); cede a 00 (−1,0 s)
    6   03        95.0    95.6   0.6   asentimiento ("sí")
    7   00        99.0    104.0  5.0   ← C1 00→02 (M→M) de la moderadora: aparte; cede a 03
    8   03        104.5   130.0  25.5
    9   00        129.6   135.0  5.4   cierre: solape 0,4 s < 0,5 → no es C1

Cálculos a mano:
- Cesiones (≤ 15 s y otro entra ≤ 2 s después): 00 → t0, t2, t7 = 3; 02 → t5 = 1.
  Apertura (3 primeros turnos sin asentimientos) = {00, 01}; cierre = {00, 03}.
  → moderadora 00 (heurística).
- Habla M = 00 (8 + 5,5 + 5 + 5,4 = 23,9) + 02 (23,8 + 9 = 32,8) = 56,7 s
  Habla H = 01 (31,5 + 21 = 52,5) + 03 (0,6 + 25,5 = 26,1) = 78,6 s
- C1 H→M: 1 evento / (56,7 / 60 min) = 1,0582 por minuto. Moderadora M→M: 1 aparte.
- Exceso (con min_events=1): (1/1) / (56,7/135,3) = 2,3862.
- B: turnos M = {0,2,3,5,7,9} = 6, media 56,7/6 = 9,45 s; H = {1,4,8} = 3, media 78/3 = 26 s;
  asentimientos H = 1. Abre (primer turno ≥ 10 s) = 01; cierra = 03.
"""

from __future__ import annotations

import pytest

from conftest import S, T
from explicometro.config import InterruptionCfg, ModeratorCfg, RatiosCfg, StageCfg, ThresholdsCfg
from explicometro.events import detect_interruptions, detect_moderator, handovers, mark_moderator
from explicometro.events import run as events_run
from explicometro.metrics import aggregate, ratio_stat
from explicometro.metrics import run as metrics_run
from explicometro.schemas import (
    AudioMeta,
    DyadStats,
    EpisodeMetrics,
    Event,
    EventType,
    Gender,
    Protagonism,
    Segment,
    Word,
)
from explicometro.turns import TurnsParams, build_turns, merge_segments, normalize_text

H, M = Gender.MALE, Gender.FEMALE
C1 = EventType.INTERRUPTION
SPEAKERS = [S("SPK_00", M), S("SPK_01", H), S("SPK_02", M), S("SPK_03", H)]
EP = [
    T(0, "SPK_00", 0.0, 8.0),
    T(1, "SPK_01", 8.5, 40.0),
    T(2, "SPK_00", 40.5, 46.0),
    T(3, "SPK_02", 46.2, 70.0),
    T(4, "SPK_01", 69.0, 90.0),
    T(5, "SPK_02", 91.0, 100.0),
    T(6, "SPK_03", 95.0, 95.6, "sí", bc=True),
    T(7, "SPK_00", 99.0, 104.0),
    T(8, "SPK_03", 104.5, 130.0),
    T(9, "SPK_00", 129.6, 135.0),
]
AUDIO = AudioMeta(episode_id="ep", path="a.wav", sample_rate=16000, channels=1, duration_s=140.0,
                  sha256="0" * 64, source_sha256="0" * 64)  # fmt: skip


def thresholds(min_events: int = 10) -> ThresholdsCfg:
    return ThresholdsCfg(ratios=RatiosCfg(min_events=min_events))


# ------------------------------------------------------------------------------- turns


def test_merge_gap_strictly_below_threshold():
    diar = [Segment(start=0, end=2, speaker="SPK_00"), Segment(start=2.2, end=4, speaker="SPK_00"),
            Segment(start=4.3, end=6, speaker="SPK_00")]  # fmt: skip
    # 2,0→2,2 (0,2 s) se une; 4,0→4,3 (0,3 s) no, porque la regla es < 0,3 s
    assert [(s.start, s.end) for s in merge_segments(diar, 0.3)] == [(0, 4), (4.3, 6)]


def test_build_turns_backchannel_by_duration_and_lexicon():
    diar = [
        Segment(start=0, end=2, speaker="SPK_00"), Segment(start=2.2, end=4, speaker="SPK_00"),
        Segment(start=4.5, end=6, speaker="SPK_00"),
        Segment(start=6.1, end=6.6, speaker="SPK_01"),   # 0,5 s → asentimiento por duración
        Segment(start=7.0, end=9.0, speaker="SPK_01"),   # "Sí, sí." 2 s → por léxico
        Segment(start=10.0, end=12.0, speaker="SPK_01"),  # "claro que no" → no
    ]  # fmt: skip
    words = [
        Word(start=0.5, end=0.9, text="Bueno", speaker="SPK_00"),
        Word(start=3.0, end=3.4, text="pues", speaker="SPK_00"),
        Word(start=5.0, end=5.4, text="eso.", speaker="SPK_00"),
        Word(start=6.2, end=6.5, text="Ajá", speaker="SPK_01"),
        Word(start=7.2, end=7.6, text="Sí,", speaker="SPK_01"),
        Word(start=7.8, end=8.2, text="sí.", speaker="SPK_01"),
        Word(start=10.2, end=10.6, text="Claro", speaker="SPK_01"),
        Word(start=10.7, end=11.0, text="que", speaker="SPK_01"),
        Word(start=11.1, end=11.4, text="no", speaker="SPK_01"),
        Word(start=20.0, end=20.3, text="huérfana", speaker="SPK_01"),  # sin turno cerca → fuera
    ]
    turns = build_turns(diar, words, TurnsParams())
    got = [(t.speaker, t.start, t.end, t.text, t.n_words, t.is_backchannel) for t in turns]
    assert got == [
        ("SPK_00", 0, 4, "Bueno pues", 2, False),
        ("SPK_00", 4.5, 6, "eso.", 1, False),
        ("SPK_01", 6.1, 6.6, "Ajá", 1, True),
        ("SPK_01", 7.0, 9.0, "Sí, sí.", 2, True),
        ("SPK_01", 10.0, 12.0, "Claro que no", 3, False),
    ]
    assert [t.id for t in turns] == [0, 1, 2, 3, 4]


def test_normalize_text_strips_accents_and_punctuation():
    assert normalize_text("¡Sí, SÍ!") == "si si"
    assert normalize_text("Ajá...") == "aja"


def test_turns_keep_overlaps_between_speakers():
    diar = [Segment(start=0, end=10, speaker="SPK_00"), Segment(start=9, end=15, speaker="SPK_01")]
    turns = build_turns(diar, [])
    assert [(t.start, t.end) for t in turns] == [(0, 10), (9, 15)]


# --------------------------------------------------------------------------------- C1


def test_c1_on_reference_episode():
    ev = detect_interruptions(EP, InterruptionCfg())
    got = [(e.src_spk, e.dst_spk, e.t_start, e.t_end, e.overlap_s, e.recovered) for e in ev]
    assert got == [
        ("SPK_01", "SPK_02", 69.0, 70.0, 1.0, True),  # 02 recupera en 91 ≤ 95
        ("SPK_00", "SPK_02", 99.0, 100.0, 1.0, False),  # el siguiente es 03, no 02
    ]


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (T(0, "SPK_00", 0, 10), T(1, "SPK_01", 9.0, 15), 1),  # solape 1 s, cede en 1 s
        (T(0, "SPK_00", 0, 10), T(1, "SPK_01", 9.6, 15), 0),  # solape 0,4 s < 0,5
        (T(0, "SPK_00", 0, 10), T(1, "SPK_01", 7.5, 15), 0),  # A sigue 2,5 s > 2
        (T(0, "SPK_00", 0, 10), T(1, "SPK_01", 8.0, 15), 1),  # justo 2,0 s: cuenta
        (T(0, "SPK_00", 0, 10), T(1, "SPK_01", 9.0, 9.8, "sí", bc=True), 0),  # B asiente
        (T(0, "SPK_00", 0, 0.9, "ya", bc=True), T(1, "SPK_01", 0.2, 15), 0),  # A asentía
        (T(0, "SPK_00", 0, 10), T(1, "SPK_01", 9.0, 9.5), 1),  # B breve no-asentimiento
    ],
)
def test_c1_definition_edges(a, b, expected):
    assert len(detect_interruptions([a, b], InterruptionCfg())) == expected


def test_c1_recovery_window_and_third_speaker():
    base = [T(0, "SPK_00", 0, 10), T(1, "SPK_01", 9, 20)]
    late = detect_interruptions([*base, T(2, "SPK_00", 25.5, 30)], InterruptionCfg())
    assert late[0].recovered is False  # 25,5 > 20 + 5
    third = detect_interruptions([*base, T(2, "SPK_02", 21, 24), T(3, "SPK_00", 24, 30)], InterruptionCfg())
    assert third[0].recovered is False  # habla antes una tercera persona


def test_c1_picks_interrupted_speaker_with_largest_overlap():
    turns = [T(0, "SPK_00", 0, 10), T(1, "SPK_02", 2, 10.5), T(2, "SPK_01", 9.0, 15)]
    ev = detect_interruptions(turns, InterruptionCfg())
    assert [(e.src_spk, e.dst_spk) for e in ev] == [("SPK_01", "SPK_02")]  # 1,5 s > 1,0 s


# -------------------------------------------------------------------------- moderador


def test_moderator_heuristic_on_reference_episode():
    assert handovers(EP, ModeratorCfg()) == {"SPK_00": 3, "SPK_02": 1}
    assert detect_moderator(EP, SPEAKERS, ModeratorCfg()) == ("SPK_00", "heuristic")


def test_moderator_needs_min_handovers_and_no_tie():
    assert detect_moderator(EP, SPEAKERS, ModeratorCfg(min_handovers=4)) == (None, None)
    tie = [T(0, "SPK_00", 0, 5), T(1, "SPK_01", 5.5, 10), T(2, "SPK_00", 10.5, 15), T(3, "SPK_01", 15.5, 20)]
    # 00 cede 2 veces, 01 cede 1 vez → con min 1: 00; si ambos 1 → empate
    assert detect_moderator(tie, [], ModeratorCfg(min_handovers=1, edge_turns=2)) == ("SPK_00", "heuristic")
    assert detect_moderator(tie[:2] + tie[3:], [], ModeratorCfg(min_handovers=1, edge_turns=2)) == (
        None,
        None,
    )


def test_moderator_metadata_override():
    spk = [*SPEAKERS[:3], SPEAKERS[3].model_copy(update={"role": "moderator"})]
    assert detect_moderator(EP, spk, ModeratorCfg()) == ("SPK_03", "metadata")
    guest = [SPEAKERS[0].model_copy(update={"role": "guest"}), *SPEAKERS[1:]]
    assert detect_moderator(EP, guest, ModeratorCfg()) == (None, None)  # 00 excluida y nadie más


def test_events_run_marks_moderator_events_including_llm():
    llm = [
        Event(type=EventType.DIRECT_CORRECTION, src_spk="SPK_00", dst_spk="SPK_01", t_start=41, t_end=45),
        Event(
            type=EventType.UNSOLICITED_EXPLANATION, src_spk="SPK_01", dst_spk="SPK_02", t_start=70, t_end=80
        ),
    ]
    ev = events_run(EP, SPEAKERS, llm, StageCfg(backend="stub"), thresholds(), None)
    assert [(e.type.value, e.src_spk, e.by_moderator) for e in ev] == [
        ("C4_direct_correction", "SPK_00", True),
        ("C1_interruption", "SPK_01", False),
        ("C3_unsolicited_explanation", "SPK_01", False),
        ("C1_interruption", "SPK_00", True),
    ]
    assert mark_moderator(ev, None) == ev


# --------------------------------------------------------------------- métricas episodio


def test_episode_metrics_by_hand():
    ev = events_run(EP, SPEAKERS, [], StageCfg(backend="stub"), thresholds(), None)
    m = metrics_run(AUDIO, SPEAKERS, EP, ev, StageCfg(), thresholds(min_events=1))
    assert m.speech_by_gender == {M: 56.7, H: 78.6}
    # A
    assert m.presence.by_gender[M].speakers == 0.5
    assert m.presence.by_gender[M].time == pytest.approx(56.7 / 135.3)
    assert m.presence.by_gender[H].words == pytest.approx(13 / 37)  # H: 4+4+1+4; M: 6 turnos × 4
    # B
    p = m.protagonism
    assert p.n_turns == {M: 6, H: 3} and p.n_backchannels == {H: 1}
    assert p.mean_turn_s == {M: pytest.approx(9.45), H: pytest.approx(26.0)}
    assert (p.opener, p.closer) == ("SPK_01", "SPK_03")
    assert (p.moderator, p.moderator_gender, p.moderator_source) == ("SPK_00", M, "heuristic")
    # C: la interrupción de la moderadora va aparte
    hm = m.events[C1]
    assert list(hm) == ["H->M"] and hm["H->M"].count == 1
    assert hm["H->M"].receptor_min == pytest.approx(0.945)
    assert hm["H->M"].rate_per_min == pytest.approx(1 / 0.945, abs=1e-4)
    assert m.moderator_events[C1]["M->M"].count == 1
    # Ratios (min_events=1 para poder ver el número)
    assert m.interruption_excess.value == pytest.approx(1 / (56.7 / 135.3), abs=1e-4)
    assert m.interruption_excess.status == "ok" and m.interruption_excess.n_events == 1
    assert m.asymmetry_ratio.status == "sin_datos"  # no hay H→H: tasa de referencia 0


def test_episode_ratios_insufficient_with_default_min_events():
    ev = events_run(EP, SPEAKERS, [], StageCfg(backend="stub"), thresholds(), None)
    m = metrics_run(AUDIO, SPEAKERS, EP, ev, StageCfg(), thresholds())
    r = m.interruption_excess
    assert (r.status, r.value, r.n_events, r.n_episodes) == ("insuficiente", None, 1, 1)


def test_unknown_speakers_out_of_dyads():
    spk = [S("SPK_00", M), S("SPK_01", Gender.UNKNOWN), S("SPK_02", M), S("SPK_03", H)]
    ev = events_run(EP, spk, [], StageCfg(backend="stub"), thresholds(), None)
    m = metrics_run(AUDIO, spk, EP, ev, StageCfg(), thresholds(min_events=1))
    assert m.events == {}  # la única C1 entre invitados la emite U
    assert m.unknown.speakers == 1 and m.unknown.time_s == pytest.approx(52.5)
    assert m.presence.by_gender[H].speakers == pytest.approx(1 / 3)


# ------------------------------------------------------------------------- agregación


def ep_stats(eid: str, h_min: float, m_min: float, hm: int, hh: int, mh: int, mm: int,
             turns_h: int = 0, mean_h: float = 0.0) -> EpisodeMetrics:  # fmt: skip
    """Episodio solo con estadísticos suficientes (lo que usa la agregación)."""
    counts = {"H->M": hm, "H->H": hh, "M->H": mh, "M->M": mm}
    row = {d: DyadStats(count=n) for d, n in counts.items() if n}
    return EpisodeMetrics(
        episode_id=eid,
        duration_s=(h_min + m_min) * 60,
        speakers_by_gender={H: 2, M: 1},
        speech_by_gender={H: h_min * 60, M: m_min * 60},
        words_by_gender={H: 100, M: 50},
        events={C1: row} if row else {},
        protagonism=Protagonism(n_turns={H: turns_h}, mean_turn_s={H: mean_h}, opener_gender=H),
    )


EPS = [
    ep_stats("e1", 10, 5, hm=6, hh=4, mh=1, mm=0, turns_h=10, mean_h=30.0),
    ep_stats("e2", 20, 10, hm=9, hh=6, mh=2, mm=1, turns_h=20, mean_h=15.0),
    ep_stats("e3", 5, 5, hm=0, hh=2, mh=0, mm=0),
]


def test_aggregate_pooled_values_by_hand():
    g = aggregate(EPS, thresholds(), ["channel"], {"channel": "La 1"})
    # Habla: H 35 min, M 20 min. C1: H→M 15, H→H 12, M→H 3, M→M 1 (31 en total).
    assert g.events[C1]["H->M"].rate_per_min == pytest.approx(15 / 20)
    assert g.events[C1]["H->H"].receptor_min == pytest.approx(35)
    a = g.asymmetry_ratio
    assert a.value == pytest.approx((15 / 20) / (12 / 35), abs=1e-4)  # 2,1875
    assert (a.n_events, a.n_episodes, a.status) == (27, 3, "ok")
    x = g.interruption_excess
    assert x.value == pytest.approx((16 / 31) / (20 / 55), abs=1e-4)  # 1,4194
    assert x.n_events == 31
    assert g.protagonism.mean_turn_s[H] == pytest.approx((10 * 30 + 20 * 15) / 30)  # 20 s, no media de medias
    assert g.protagonism.openers == {H: 3}
    assert g.presence.by_gender[M].time == pytest.approx(20 / 55)


def test_bootstrap_ci_reproducible_and_brackets_value():
    g1 = aggregate(EPS, thresholds())
    g2 = aggregate(EPS, thresholds())
    a = g1.asymmetry_ratio
    assert a.ci_low is not None and a.ci_low <= a.value <= a.ci_high
    assert (a.ci_low, a.ci_high) == (g2.asymmetry_ratio.ci_low, g2.asymmetry_ratio.ci_high)  # semilla fija
    assert g1.bootstrap == {"iterations": 1000, "seed": 13, "ci": 0.95}
    other = aggregate(EPS, ThresholdsCfg(ratios=RatiosCfg(seed=7)))
    assert other.asymmetry_ratio.value == a.value  # el punto no depende de la semilla


def test_aggregate_insufficient_and_single_episode():
    small = aggregate([EPS[2]], thresholds())
    assert small.asymmetry_ratio.status == "insuficiente" and small.asymmetry_ratio.value is None
    assert small.asymmetry_ratio.n_events == 2
    one = aggregate([EPS[1]], thresholds())  # 15 eventos: número sí, IC no (1 episodio)
    assert one.asymmetry_ratio.status == "ok" and one.asymmetry_ratio.ci_low is None


def test_ratio_stat_rules():
    rc = RatiosCfg()
    assert ratio_stat(1.5, 9, 3, rc).status == "insuficiente"
    assert ratio_stat(None, 0, 3, rc).status == "sin_datos"
    assert ratio_stat(1.5, 10, 3, rc, (1.1, 2.0)).model_dump(include={"value", "ci_low", "status"}) == {
        "value": 1.5, "ci_low": 1.1, "status": "ok"
    }  # fmt: skip


def test_events_parquet_roundtrip_with_new_fields(tmp_path):
    from explicometro import io as aio

    ev = events_run(EP, SPEAKERS, [], StageCfg(backend="stub"), thresholds(), None)
    aio.write_parquet(tmp_path / "e.parquet", ev, Event)
    assert aio.read_parquet(tmp_path / "e.parquet", Event) == ev


def test_aggregate_script_groups_by_manifest(tmp_path):
    import csv
    import importlib.util
    import json
    import sys

    from conftest import ROOT
    from explicometro import io as aio

    interim = tmp_path / "interim"
    for e in [*EPS, EPS[0].model_copy(update={"episode_id": "dry", "dry_run": True})]:
        aio.write_model(interim / e.episode_id / "metrics.json", e)
    manifest = tmp_path / "episodes.yaml"
    manifest.write_text(
        "episodes:\n"
        "  - {id: e1, channel: La1, year: 2025}\n"
        "  - {id: e2, channel: La1, year: 2026}\n"
        "  - {id: e3, channel: TV3, year: 2026}\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"paths: {{interim: {interim}, processed: {tmp_path / 'out'}}}\n", encoding="utf-8")
    spec = importlib.util.spec_from_file_location("aggregate_script", ROOT / "scripts" / "aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_script"] = mod
    spec.loader.exec_module(mod)
    assert mod.main(["--config", str(cfg), "--manifest", str(manifest)]) == 0

    corpus = json.loads((tmp_path / "out" / "metrics_corpus.json").read_text())
    assert len(corpus) == 1 and corpus[0]["n_episodes"] == 3  # el dry-run no entra
    assert corpus[0]["asymmetry_ratio"]["value"] == pytest.approx(2.1875)
    by_channel = {
        g["group"]["channel"]: g for g in json.loads((tmp_path / "out" / "metrics_channel.json").read_text())
    }
    assert by_channel["La1"]["episodes"] == ["e1", "e2"]
    assert by_channel["TV3"]["asymmetry_ratio"]["status"] == "insuficiente"
    rows = list(csv.DictReader((tmp_path / "out" / "metrics_channel_year.csv").open()))
    assert [(r["channel"], r["year"]) for r in rows] == [("La1", "2025"), ("La1", "2026"), ("TV3", "2026")]
    assert "insuficiente" in (tmp_path / "out" / "report_channel.md").read_text()


def test_moderator_requires_clear_margin():
    # 00 cede 4 veces y 01 cede 2 (01 no llega al mínimo, pero cuenta como rival):
    # 4 ≥ 1,5 × 2 → 00; con margen 2,5 (4 < 5) → nadie
    plan = [("00", 0, 5), ("01", 5.5, 10), ("00", 10.5, 15), ("01", 15.5, 20),
            ("00", 20.5, 25), ("02", 25.5, 60), ("00", 60.5, 63), ("01", 63.5, 90)]  # fmt: skip
    turns = [T(i, f"SPK_{s}", a, b) for i, (s, a, b) in enumerate(plan)]
    assert handovers(turns, ModeratorCfg()) == {"SPK_00": 4, "SPK_01": 2}
    cfg = ModeratorCfg(edge_turns=8)
    assert detect_moderator(turns, [], cfg) == ("SPK_00", "heuristic")  # 4 ≥ 1,5 × 2
    assert detect_moderator(turns, [], ModeratorCfg(edge_turns=8, min_margin=2.5)) == (None, None)
