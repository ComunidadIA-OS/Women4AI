# Explicómetro

Sistema de código abierto que procesa audio de debates y tertulias (castellano y catalán) y
calcula métricas de **presencia**, **protagonismo** y **conductas asociadas al mansplaining**,
desagregadas por género y por díada emisor→receptor (H→M, M→H, H→H, M→M).

Proyecto del reto *Mujeres por la IA Abierta* (SEDIA). Licencia Apache-2.0.

> Estado: `ingest → vad → diarize → asr → align` implementado (VAD y alineación validados en
> CPU; diarización y ASR pendientes de validar en GPU con `make test-gpu`). `gender` implementado
> (falta entrenar el clasificador: `make train-gender`). `llm` y C2 siguen en *stub*. `make dry-run` recorre todo el pipeline en CPU.

## Inicio rápido

```bash
make setup       # Python 3.11, .venv, núcleo + dev + app
make dry-run     # CPU, clip sintético de 120 s → data/interim/dryrun_clip_120s/metrics.json
make test lint
make app         # visor Streamlit
```

En la máquina con GPU (2× V100S):

```bash
make setup-gpu                      # torch, pyannote.audio 3.1, transformers
cp .env.example .env                # HF_TOKEN (acepta las condiciones de pyannote en HF)
python scripts/run_pipeline.py --episode <id> --until align    # busca data/raw/<id>.*
make test-gpu                       # diálogo de dos voces con verdad conocida
```

Cada ejecución deja `pipeline.log` y `timings.jsonl` (segundos por etapa) en
`data/interim/<id>/`. Los episodios largos se procesan en trozos de 10 min con 5 s de margen
(VAD y ASR); la diarización va sobre el episodio entero para que los hablantes sean coherentes.

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
| ingest | `audio.json` + `audio.wav` | ffmpeg → 16 kHz mono, loudnorm EBU R128 (2 pasadas, lineal) | igual (recorta a 120 s) |
| vad | `vad.json` | Silero VAD (ONNX, CPU) | stub |
| diarize | `diarization.rttm` | pyannote 3.1 (flag: segmentación HiTZ RTVE), con solapes | stub |
| asr | `asr_words.json` | faster-whisper large-v3-turbo · bsc-large-v3-cat · LoS-punctuated (transformers) | stub |
| align | `transcript.json` | mayor solape; en solape gana quien empezó antes salvo palabra entera en el nuevo | igual |
| gender | `speakers.json` | metadata CSV > ECAPA + LogisticRegression > `U` | solo metadata, resto `U` |
| turns | `turns.parquet` | segmentos del mismo hablante con hueco < 0,3 s + palabras; asentimientos marcados | igual |
| llm | `llm_events.json` | Salamandra 7B instruct (API OpenAI-compatible) | stub |
| events | `events.parquet` | C1 por reglas (+ recuperación), moderador, C2 con MrBERT-es | C1 reglas, C2 stub |
| metrics | `metrics.json` | A, B, tasas por díada, ratios con n; agregado con IC bootstrap | igual |
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
src/explicometro/   schemas.py (contrato) · config.py · io.py · pipeline.py · cli.py · audio.py · chunking.py
                    ingest vad diarize asr align gender turns events metrics llm report
app/                visor Streamlit
configs/            default.yaml · episodes.example.yaml
scripts/            download_episodes · run_pipeline · evaluate · make_fixture · make_tts_fixture
data/               raw/ interim/ processed/ (ignorados) · annotations/ (versionado)
tests/              pytest con casos sintéticos
docs/               arquitectura
```

Definiciones operativas de turnos, C1, moderador, tasas, ratios e IC:
[`docs/metrics.md`](docs/metrics.md). Agregado por corpus/cadena/año: `make aggregate`.

Sobre el género de los hablantes (qué infiere el clasificador y qué no, prioridad de la
metadata pública, clase `U`): [`docs/gender.md`](docs/gender.md).

Ver [`CONTRIBUTING.md`](CONTRIBUTING.md), [`ROADMAP.md`](ROADMAP.md) y
[`THIRD_PARTY.md`](THIRD_PARTY.md).
