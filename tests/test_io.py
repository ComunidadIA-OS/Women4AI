from explicometro import io as aio
from explicometro.schemas import Event, EventType, Segment, Turn


def test_rttm_roundtrip(tmp_path):
    segs = [Segment(start=0.5, end=2.25, speaker="SPK_00"), Segment(start=2.0, end=4.0, speaker="SPK_01")]
    p = tmp_path / "d.rttm"
    aio.write_rttm(p, segs, "ep")
    assert aio.read_rttm(p) == segs


def test_parquet_roundtrip_and_empty(tmp_path):
    ev = [
        Event(
            type=EventType.INTERRUPTION, src_spk="SPK_01", dst_spk="SPK_00", t_start=1, t_end=1.6, score=1.0
        ),
        Event(type=EventType.DIRECT_CORRECTION, src_spk="SPK_00", dst_spk="SPK_01", t_start=5, t_end=9),
    ]
    p = tmp_path / "e.parquet"
    aio.write_parquet(p, ev, Event)
    assert aio.read_parquet(p, Event) == ev

    pe = tmp_path / "t.parquet"
    aio.write_parquet(pe, [], Turn)
    assert aio.read_parquet(pe, Turn) == []


def test_json_list_roundtrip(tmp_path):
    segs = [Segment(start=0, end=1), Segment(start=2, end=3, score=0.9)]
    p = tmp_path / "v.json"
    aio.write_models(p, segs)
    assert aio.read_models(p, Segment) == segs
