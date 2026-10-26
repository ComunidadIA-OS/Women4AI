"""Troceado, fusión sin duplicados y regla de alineación (casos sintéticos, sin modelos)."""

import itertools

import pytest

from explicometro.align import assign_speakers, speaker_for
from explicometro.chunking import Chunk, merge_segments, merge_words, plan_chunks
from explicometro.schemas import Segment, Word


def W(s, e, t="x", spk=None):
    return Word(start=s, end=e, text=t, speaker=spk)


def D(s, e, spk):
    return Segment(start=s, end=e, speaker=spk)


# ---------------------------------------------------------------------------- troceado


def test_short_episode_is_one_chunk():
    assert plan_chunks(604.0, 600, 5) == [Chunk(0, 0.0, 604.0, 0.0, 604.0)]


def test_chunks_cover_without_overlap_and_have_margins():
    chunks = plan_chunks(3600.0, 600, 5)
    assert len(chunks) == 6
    assert chunks[0].core_start == 0 and chunks[-1].core_end == 3600
    for a, b in itertools.pairwise(chunks):
        assert a.core_end == b.core_start  # núcleos contiguos, sin hueco ni solape
        assert b.start == pytest.approx(b.core_start - 5) and a.end == pytest.approx(a.core_end + 5)


def test_boundaries_snap_to_silence():
    speech = [Segment(start=0, end=597.0), Segment(start=599.0, end=1300)]  # silencio 597–599
    chunks = plan_chunks(1300.0, 600, 5, speech)
    assert chunks[0].core_end == pytest.approx(598.0)


def test_merge_words_no_duplicates_no_losses_with_jitter():
    truth = [W(t, t + 0.3, f"w{i}") for i, t in enumerate([1.0, 9.0, 9.8, 10.1, 10.6, 15.0, 19.9, 21.0])]
    chunks = plan_chunks(30.0, 10, 2)
    per_chunk = []
    for c in chunks:
        jitter = 0.04 * (c.index + 1)  # cada ventana da timestamps algo distintos
        seen = [W(w.start + jitter, w.end + jitter, w.text) for w in truth if c.start <= w.start < c.end]
        per_chunk.append((c, seen))
    merged = merge_words(per_chunk)
    assert [w.text for w in merged] == [w.text for w in truth]


def test_merge_words_dedups_boundary_straddler():
    c0, c1 = Chunk(0, 0, 10, 0, 12), Chunk(1, 10, 20, 8, 20)
    # La misma palabra vista por ambas ventanas, con midpoints a cada lado de la frontera.
    merged = merge_words([(c0, [W(9.80, 10.18, "hola")]), (c1, [W(9.84, 10.30, "Hola")])])
    assert len(merged) == 1


def test_merge_segments_joins_across_boundary():
    c0, c1 = Chunk(0, 0, 10, 0, 12), Chunk(1, 10, 20, 8, 20)
    out = merge_segments([(c0, [Segment(start=8, end=11.5)]), (c1, [Segment(start=8.1, end=13)])])
    assert out == [Segment(start=8, end=13)]


# ---------------------------------------------------------------------------- alineación

A0 = D(0.0, 10.0, "SPK_00")  # A habla 0–10
B1 = D(9.0, 15.0, "SPK_01")  # B entra en 9 (solape 9–10)


def test_single_speaker_by_overlap():
    assert speaker_for(W(2, 2.4), [A0, B1]) == "SPK_00"
    assert speaker_for(W(12, 12.4), [A0, B1]) == "SPK_01"


def test_overlap_zone_word_entirely_in_new_goes_to_new():
    assert speaker_for(W(9.2, 9.6), [A0, B1]) == "SPK_01"


def test_overlap_zone_word_straddling_new_start_stays_with_earlier():
    assert speaker_for(W(8.8, 9.3), [A0, B1]) == "SPK_00"


def test_backchannel_inside_long_turn():
    bc = D(4.0, 4.6, "SPK_01")
    assert speaker_for(W(4.1, 4.5), [A0, bc]) == "SPK_01"  # entera en el nuevo
    assert speaker_for(W(3.8, 4.3), [A0, bc]) == "SPK_00"  # empieza antes del nuevo


def test_no_overlap_nearest_within_gap_else_none():
    assert speaker_for(W(15.2, 15.4), [A0, B1], max_gap_s=0.5) == "SPK_01"
    assert speaker_for(W(17.0, 17.3), [A0, B1], max_gap_s=0.5) is None


def test_assign_speakers_index_matches_bruteforce():
    diar = [D(i * 1.7, i * 1.7 + 2.5, f"SPK_0{i % 3}") for i in range(60)]
    words = [W(t / 10, t / 10 + 0.25) for t in range(0, 1050, 3)]
    fast = assign_speakers(words, diar)
    slow = [speaker_for(w, sorted(diar, key=lambda s: s.start)) for w in words]
    assert [w.speaker for w in fast] == slow
