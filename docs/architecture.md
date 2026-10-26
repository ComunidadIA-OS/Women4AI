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
