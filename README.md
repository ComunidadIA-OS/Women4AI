# Explicómetro

Sistema de código abierto que procesa audio de debates y tertulias (castellano y catalán) y
calcula métricas de **presencia**, **protagonismo** y **conductas asociadas al mansplaining**,
desagregadas por género y por díada emisor→receptor (H→M, M→H, H→H, M→M).

Proyecto del reto *Mujeres por la IA Abierta* (SEDIA). Licencia Apache-2.0.

> Estado: esqueleto. El pipeline corre de punta a punta en CPU con *stubs* (`make dry-run`);
> los backends con modelos están declarados y pendientes de implementar.

## Inicio rápido

```bash
make setup       # Python 3.11, .venv, núcleo + dev + app
make dry-run     # CPU, clip sintético de 120 s → data/interim/dryrun_clip_120s/metrics.json
make test lint
make app         # visor Streamlit
```

En la máquina con GPU: `make setup-gpu`, copia `.env.example` a `.env` y rellénalo, y
`make run AUDIO=data/raw/<episodio>.wav`.

## Pipeline

```
ingest → vad → diarize ─┐
            └→ asr ─────┴→ align → turns ─┬→ llm (C3, C4) ─┐
               diarize → gender ──────────┴────────────────┴→ events (C1, C2) → metrics → report
```

Cada etapa escribe un artefacto en `data/interim/<episodio>/` y se puede re-ejecutar sola
(`python -m explicometro run --episode <id> --only <etapa> --force`). La caché se invalida
si cambia la configuración de la etapa o de cualquiera de sus dependencias.
Detalle en [`docs/architecture.md`](docs/architecture.md).

| Etapa | Artefacto | Backend por defecto | En dry-run |
|---|---|---|---|
| ingest | `audio.json` + `audio.wav` | stdlib / ffmpeg | igual (recorta a 120 s) |
| vad | `vad.json` | Silero VAD | stub |
| diarize | `diarization.rttm` | pyannote 3.1 + segmentación RTVE (HiTZ) | stub |
| asr | `asr_words.json` | faster-whisper large-v3 | stub |
| align | `transcript.json` | solape palabra↔hablante | igual |
| gender | `speakers.json` | ECAPA + LogisticRegression | stub (todo `U`) |
| turns | `turns.parquet` | agrupación de palabras | igual |
| llm | `llm_events.json` | Salamandra 7B instruct (API OpenAI-compatible) | stub |
| events | `events.parquet` | C1 por reglas + C2 con MrBERT-es | C1 reglas, C2 stub |
| metrics | `metrics.json` | funciones puras | igual |
| report | `report.md` | Markdown | igual |

## Métricas

Definiciones cerradas en [`CLAUDE.md`](CLAUDE.md): A presencia, B protagonismo, C1
interrupción competitiva, C2 reformulación sin atribución, C3 explicación no solicitada,
C4 corrección directa, exceso de interrupción y ratio de asimetría (IC 95 % por bootstrap a
nivel de programa).

## Privacidad y datos

- Los hablantes se identifican como `SPK_00`, `SPK_01`…; el esquema rechaza cualquier otro ID.
- No se publica audio: solo scripts de descarga (`scripts/download_episodes.py`), hashes
  (`configs/episodes.sha256`), anotaciones y métricas.

## Estructura

```
src/explicometro/   schemas.py (contrato) · config.py · io.py · pipeline.py · cli.py
                    ingest vad diarize asr align gender turns events metrics llm report
app/                visor Streamlit
configs/            default.yaml · episodes.example.yaml
scripts/            download_episodes · run_pipeline · evaluate · make_fixture
data/               raw/ interim/ processed/ (ignorados) · annotations/ (versionado)
tests/              pytest con casos sintéticos
docs/               arquitectura
```

Ver [`CONTRIBUTING.md`](CONTRIBUTING.md), [`ROADMAP.md`](ROADMAP.md) y
[`THIRD_PARTY.md`](THIRD_PARTY.md).
