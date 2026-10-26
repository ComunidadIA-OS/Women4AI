import pytest
from pydantic import ValidationError

from explicometro.schemas import (
    Dyad,
    EpisodeMetrics,
    Event,
    EventType,
    Gender,
    GenderSource,
    Segment,
    Speaker,
    Word,
    dyad_of,
)


def test_interval_order():
    with pytest.raises(ValidationError):
        Segment(start=2.0, end=1.0)
    assert Segment(start=1.0, end=3.5).duration == 2.5


def test_speaker_ids_are_anonymous():
    with pytest.raises(ValidationError):
        Speaker(id="Ana García")
    Speaker(id="SPK_00")
    Speaker(id="SPK_123")


def test_gender_requires_source():
    with pytest.raises(ValidationError):
        Speaker(id="SPK_00", gender=Gender.FEMALE)  # source=none
    with pytest.raises(ValidationError):
        Speaker(id="SPK_00", gender_conf=1.5)
    s = Speaker(id="SPK_00", gender=Gender.FEMALE, gender_source=GenderSource.MODEL, gender_conf=0.9)
    assert s.gender == "M"


def test_event_validation():
    with pytest.raises(ValidationError):
        Event(type=EventType.INTERRUPTION, src_spk="SPK_00", dst_spk="SPK_00", t_start=0, t_end=1)
    with pytest.raises(ValidationError):
        Event(type="C9", src_spk="SPK_00", dst_spk="SPK_01", t_start=0, t_end=1)
    e = Event(type="C1_interruption", src_spk="SPK_00", dst_spk="SPK_01", t_start=1, t_end=2)
    assert e.type is EventType.INTERRUPTION


def test_extra_fields_forbidden_and_frozen():
    with pytest.raises(ValidationError):
        Word(start=0, end=1, text="hola", nombre="x")
    w = Word(start=0, end=1, text="hola")
    with pytest.raises(ValidationError):
        w.text = "adiós"


def test_dyad_of():
    assert dyad_of(Gender.MALE, Gender.FEMALE) is Dyad.H_M
    assert dyad_of(Gender.FEMALE, Gender.FEMALE) is Dyad.M_M
    assert dyad_of(Gender.UNKNOWN, Gender.FEMALE) is None


def test_empty_metrics_roundtrip():
    m = EpisodeMetrics(episode_id="ep1")
    back = EpisodeMetrics.model_validate_json(m.model_dump_json())
    assert back == m
    assert back.events == {} and back.asymmetry_ratio.status == "sin_datos"
