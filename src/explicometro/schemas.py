"""Contrato de datos entre módulos.

Toda etapa del pipeline recibe y devuelve **solo** estos tipos. Si una etapa necesita
información nueva, se añade aquí (y se sube ``SCHEMA_VERSION``), nunca como dict suelto.

Convenciones:
- Tiempos en segundos desde el inicio del audio procesado (float >= 0).
- Hablantes anonimizados: ``SPK_00``, ``SPK_01``... (regex ``^SPK_\\d{2,}$``).
- Género: ``H`` (hombre), ``M`` (mujer), ``U`` (desconocido). Díadas emisor->receptor
  con la notación de CLAUDE.md: ``H->M``, ``M->H``, ``H->H``, ``M->M``.
- Todos los modelos son inmutables y rechazan campos extra (``extra="forbid"``).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "0.3.0"

SpeakerId = Annotated[str, Field(pattern=r"^SPK_\d{2,}$")]
EpisodeId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$", min_length=1)]
Seconds = Annotated[float, Field(ge=0)]
Prob = Annotated[float, Field(ge=0, le=1)]


class Gender(StrEnum):
    MALE = "H"
    FEMALE = "M"
    UNKNOWN = "U"


class GenderSource(StrEnum):
    """De dónde sale la etiqueta de género."""

    MODEL = "model"  # clasificador acústico (ECAPA + LogReg)
    METADATA = "metadata"  # metadatos públicos del programa
    MANUAL = "manual"  # anotación humana
    NONE = "none"  # sin etiqueta


class EventType(StrEnum):
    INTERRUPTION = "C1_interruption"
    REFORMULATION = "C2_reformulation"
    UNSOLICITED_EXPLANATION = "C3_unsolicited_explanation"
    DIRECT_CORRECTION = "C4_direct_correction"


class Dyad(StrEnum):
    H_M = "H->M"
    M_H = "M->H"
    H_H = "H->H"
    M_M = "M->M"


def dyad_of(src: Gender, dst: Gender) -> Dyad | None:
    """Díada emisor->receptor; ``None`` si algún género es desconocido."""
    if Gender.UNKNOWN in (src, dst):
        return None
    return Dyad(f"{src.value}->{dst.value}")


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Interval(_Base):
    start: Seconds
    end: Seconds

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) < start ({self.start})")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class AudioMeta(_Base):
    """Salida de ``ingest``: audio normalizado (16 kHz mono PCM16) listo para el resto."""

    episode_id: EpisodeId
    path: str = Field(description="Ruta al WAV normalizado, relativa al directorio del episodio")
    sample_rate: int = Field(gt=0)
    channels: int = Field(ge=1)
    duration_s: Seconds
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="Hash del WAV normalizado")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="Hash del fichero original")
    truncated_to_s: Seconds | None = None


class Segment(_Interval):
    """Región de audio. Sin ``speaker`` = región de voz (VAD); con ``speaker`` = diarización."""

    speaker: SpeakerId | None = None
    score: Prob | None = None


class Word(_Interval):
    text: str = Field(min_length=1)
    speaker: SpeakerId | None = None
    prob: Prob | None = None


class Turn(_Interval):
    """Intervención continua de un hablante (segmentos unidos si el hueco es < 0,3 s).

    ``is_backchannel``: asentimiento (< 1 s o texto del léxico). No cuenta como tener la
    palabra: se excluye de C1 y de los turnos del bloque B.
    """

    id: int = Field(ge=0)
    speaker: SpeakerId
    text: str = ""
    n_words: int = Field(default=0, ge=0)
    is_backchannel: bool = False


class Speaker(_Base):
    """Hablante anonimizado.

    ``gender`` es una etiqueta binaria de voz (o la pública de los metadatos), no una
    identidad: ver docs/gender.md. ``public_name`` solo se rellena desde metadatos públicos
    y con ``privacy.use_public_names: true``.
    """

    id: SpeakerId
    gender: Gender = Gender.UNKNOWN
    gender_source: GenderSource = GenderSource.NONE
    gender_conf: Prob = 0.0
    total_s: Seconds = 0.0
    clean_s: Seconds | None = Field(default=None, description="Voz sin solapes usada para clasificar")
    public_name: str | None = None
    role: Literal["moderator", "guest"] | None = Field(
        default=None, description="Solo desde metadata; si es None, el moderador se detecta"
    )

    @model_validator(mode="after")
    def _check_gender(self) -> Self:
        if self.gender_source is GenderSource.NONE and self.gender is not Gender.UNKNOWN:
            raise ValueError("gender_source='none' exige gender='U'")
        return self


class Event(_Base):
    """Conducta detectada de ``src_spk`` (emisor) hacia ``dst_spk`` (receptor)."""

    type: EventType
    src_spk: SpeakerId
    dst_spk: SpeakerId
    t_start: Seconds
    t_end: Seconds
    evidence: str = Field(default="", description="Cita textual o descripción verificable")
    score: Prob | None = Field(default=None, description="Confianza/similitud en [0, 1]")
    overlap_s: Seconds | None = Field(default=None, description="C1: solape entre B y A")
    recovered: bool | None = Field(default=None, description="C1: A recupera el turno")
    by_moderator: bool = Field(
        default=False, description="Emitido por el moderador: se reporta aparte, no en C1–C4"
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.t_end < self.t_start:
            raise ValueError("t_end < t_start")
        if self.src_spk == self.dst_spk:
            raise ValueError("src_spk y dst_spk deben ser distintos")
        return self


# --------------------------------------------------------------------------- métricas

RatioStatus = Literal["ok", "insuficiente", "sin_datos"]


class GenderShare(_Base):
    speakers: Prob | None = None
    time: Prob | None = None
    words: Prob | None = None


class Presence(_Base):
    """Bloque A: proporciones por género, calculadas solo sobre hablantes H y M."""

    by_gender: dict[Gender, GenderShare] = Field(default_factory=dict)


class Protagonism(_Base):
    """Bloque B de un episodio. Turnos sin backchannels (que se cuentan aparte)."""

    n_turns: dict[Gender, int] = Field(default_factory=dict)
    mean_turn_s: dict[Gender, Seconds] = Field(default_factory=dict)
    n_backchannels: dict[Gender, int] = Field(default_factory=dict)
    opener: SpeakerId | None = None
    opener_gender: Gender | None = None
    closer: SpeakerId | None = None
    closer_gender: Gender | None = None
    moderator: SpeakerId | None = None
    moderator_gender: Gender | None = None
    moderator_source: Literal["metadata", "heuristic"] | None = None


class ProtagonismAgg(_Base):
    """Bloque B agregado: turnos sumados y recuentos de quién abre/cierra/modera."""

    n_turns: dict[Gender, int] = Field(default_factory=dict)
    mean_turn_s: dict[Gender, Seconds] = Field(default_factory=dict)
    n_backchannels: dict[Gender, int] = Field(default_factory=dict)
    openers: dict[Gender, int] = Field(default_factory=dict)
    closers: dict[Gender, int] = Field(default_factory=dict)
    moderators: dict[Gender, int] = Field(default_factory=dict)


class UnknownShare(_Base):
    """Lo que queda fuera de las métricas por género (hablantes U)."""

    speakers: int = Field(default=0, ge=0)
    time_s: Seconds = 0.0
    time_share: Prob | None = Field(default=None, description="Sobre el tiempo total de habla")
    words_share: Prob | None = None


class DyadStats(_Base):
    """Tasa de una conducta en una díada, con su tamaño muestral."""

    count: int = Field(default=0, ge=0)
    receptor_min: float = Field(default=0.0, ge=0, description="Minutos de habla del género receptor")
    rate_per_min: float | None = Field(default=None, ge=0, description="count / receptor_min")


class RatioStat(_Base):
    """Ratio con tamaño muestral. Con < ``min_events`` eventos: ``insuficiente`` y sin número."""

    value: float | None = Field(default=None, ge=0)
    ci_low: float | None = None
    ci_high: float | None = None
    n_events: int = Field(default=0, ge=0)
    n_episodes: int = Field(default=0, ge=0)
    status: RatioStatus = "sin_datos"


EventTable = dict[EventType, dict[Dyad, DyadStats]]


class EpisodeMetrics(_Base):
    """Métricas de un episodio + estadísticos suficientes para agregarlo con otros."""

    schema_version: str = SCHEMA_VERSION
    episode_id: EpisodeId
    dry_run: bool = False
    config_hash: str | None = None
    duration_s: Seconds = 0.0
    speech_s: Seconds = 0.0
    n_speakers: int = Field(default=0, ge=0)
    speakers_by_gender: dict[Gender, int] = Field(default_factory=dict)
    speech_by_gender: dict[Gender, Seconds] = Field(default_factory=dict)
    words_by_gender: dict[Gender, int] = Field(default_factory=dict)
    presence: Presence = Field(default_factory=Presence)
    unknown: UnknownShare = Field(default_factory=UnknownShare)
    protagonism: Protagonism = Field(default_factory=Protagonism)
    events: EventTable = Field(default_factory=dict, description="Sin eventos del moderador")
    moderator_events: EventTable = Field(default_factory=dict, description="Moderador → invitados")
    interruption_excess: RatioStat = Field(default_factory=RatioStat)
    asymmetry_ratio: RatioStat = Field(default_factory=RatioStat)


class GroupMetrics(_Base):
    """Métricas agregadas de un grupo de episodios (corpus, cadena, año...)."""

    schema_version: str = SCHEMA_VERSION
    group_by: list[str] = Field(default_factory=list)
    group: dict[str, str] = Field(default_factory=dict)
    episodes: list[EpisodeId] = Field(default_factory=list)
    n_episodes: int = Field(default=0, ge=0)
    duration_s: Seconds = 0.0
    speakers_by_gender: dict[Gender, int] = Field(default_factory=dict)
    speech_by_gender: dict[Gender, Seconds] = Field(default_factory=dict)
    words_by_gender: dict[Gender, int] = Field(default_factory=dict)
    presence: Presence = Field(default_factory=Presence)
    unknown: UnknownShare = Field(default_factory=UnknownShare)
    protagonism: ProtagonismAgg = Field(default_factory=ProtagonismAgg)
    events: EventTable = Field(default_factory=dict)
    moderator_events: EventTable = Field(default_factory=dict)
    interruption_excess: RatioStat = Field(default_factory=RatioStat)
    asymmetry_ratio: RatioStat = Field(default_factory=RatioStat)
    bootstrap: dict[str, int | float] = Field(default_factory=dict)
