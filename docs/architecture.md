# Arquitectura

## Contrato de datos (`src/explicometro/schemas.py`)

Los módulos **solo** leen y escriben estos tipos pydantic (inmutables, `extra="forbid"`):

| Tipo | Campos clave | Producido por |
|---|---|---|
| `AudioMeta` | `episode_id`, `path`, `sample_rate`, `duration_s`, `sha256`, `source_sha256` | ingest |
| `Segment` | `start`, `end`, `speaker?`, `score?` | vad (sin speaker), diarize (con speaker) |
| `Word` | `start`, `end`, `text`, `speaker?`, `prob?` | asr (sin speaker), align (con speaker) |
| `Turn` | `id`, `speaker`, `start`, `end`, `text`, `n_words` | turns |
| `Speaker` | `id`, `gender`, `gender_source`, `gender_conf`, `total_s` | gender |
| `Event` | `type`, `src_spk`, `dst_spk`, `t_start`, `t_end`, `evidence`, `score` | llm (C3, C4), events (C1, C2) |
| `EpisodeMetrics` | `presence`, `protagonism`, `events[tipo][díada]`, `interruption_excess`, `asymmetry_ratio` | metrics |

`AudioMeta` es la única adición a la lista inicial: la etapa `ingest` necesita un tipo de
salida y así las demás no dependen de rutas sueltas.

Invariantes que el esquema garantiza:

- `end >= start` en todo intervalo; `src_spk != dst_spk` en eventos.
- IDs de hablante `^SPK_\d{2,}$` (anonimización obligatoria; `diarize` reetiqueta por orden de aparición).
- `gender_source="none"` ⇒ `gender="U"`; `gender_conf` y `score` en [0, 1].
- Género `H`/`M`/`U`; díadas `H->M`, `M->H`, `H->H`, `M->M` (eventos con `U` no cuentan en díadas).

## Etapas y dependencias (`pipeline.py`)

```
ingest ── vad ──┬── diarize ──┬── align ── turns ──┬── llm ──┐
                └── asr ──────┘                    │         ├── events ── metrics ── report
                    diarize ── gender ─────────────┴─────────┘
```

Firma uniforme: cada módulo expone `run(...)` con tipos del contrato y un dict `BACKENDS`
(`stub` + reales). Los backends reales importan sus librerías dentro de la función.

## Caché

Clave por etapa = hash(config de la etapa, runtime, umbrales si aplica, claves de las
dependencias; para `ingest`, ruta + tamaño + mtime del audio y recorte). Se guarda en
`data/interim/<ep>/.cache/<etapa>.key`. Si el artefacto existe y la clave coincide, se
carga; si no, se recalcula. Tras escribir, se relee el artefacto (round-trip) para validar
que cumple el esquema.

## Dry-run

`--dry-run` aplica `dry_run` del YAML: CPU, `float32`, recorte a 120 s y backend `stub` en
todas las etapas con modelos (`keep_backends` mantiene las deterministas). El resultado es
un `metrics.json` vacío pero válido, con `dry_run: true`.

## Etapas de audio (ingest → align)

| Etapa | Qué hace | Por trozos |
|---|---|---|
| ingest | ffmpeg: mezcla a mono, `loudnorm` EBU R128 (medición + aplicación lineal), 16 kHz PCM16. Hash del original y del normalizado. | no (ffmpeg ya es streaming) |
| vad | Silero (ONNX de faster-whisper, CPU). Segmentos recortados al núcleo del trozo y unidos en las fronteras. | sí |
| diarize | pyannote 3.1; opcionalmente segmentación HiTZ RTVE con los hiperparámetros de 3.1. Cada pista de la `Annotation` es un `Segment`: los solapes se conservan. | no, para que las etiquetas sean coherentes |
| asr | Ventana = núcleo ± 5 s; la no-voz se silencia; palabras fuera de la voz se descartan; fusión por punto medio + deduplicación. | sí |
| align | Regla de solape por palabra (ver docstring de `align.py`), con índice por cubetas de 1 s. | n/a |

Después: `turns` une segmentos de diarización y asigna palabras; `events` detecta C1, el
moderador y marca sus eventos; `metrics` calcula el episodio y `scripts/aggregate.py` agrupa.
Definiciones en `docs/metrics.md`.

Fronteras de trozo: cada ~`chunk_s` se busca el centro del silencio del VAD más cercano dentro
de ±`chunk_margin_s`; si no hay, se corta en el punto nominal y el margen cubre la palabra.

## Evaluación con verdad conocida

`scripts/make_tts_fixture.py` sintetiza un diálogo de dos voces (StyleTTS2 o flite) con
turnos, una interrupción y un asentimiento solapados, y escribe `truth.rttm`/`truth.json`.
`evaluation.py` ofrece métricas por trama (VAD P/R, precisión por hablante en zonas de un solo
hablante, recall de solapes), precisión hablante-por-palabra y WER.
