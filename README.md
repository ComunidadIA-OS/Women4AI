# Explicómetro

Sistema de código abierto que procesa audio de debates y tertulias (castellano y catalán) y
calcula métricas de **presencia**, **protagonismo** y **conductas asociadas al mansplaining**,
desagregadas por género y por díada emisor→receptor (H→M, M→H, H→H, M→M).

Proyecto del reto *Mujeres por la IA Abierta* (SEDIA). Licencia Apache-2.0.

> **Estado (v1.0.0, entrega 30/10/2026).** Pipeline completo `ingest → … → report`
> implementado y probado en CPU (`make test`: 115 tests; `make dry-run`). Ruta de la demo
> (`configs/demo.yaml`): diarización y ASR en GPU, género por metadatos públicos, C1 por reglas
> y C3/C4 con juez LLM vía **OVHcloud AI Endpoints**. Pendiente de validar con datos reales
> anotados: C2 (τ sin calibrar → no emite), clasificador acústico de género (sin entrenar) y
> precisión de C1/C3/C4 frente a un gold set. Ver [Hoja de ruta](#hoja-de-ruta).

## Por qué

En tertulias y debates, quién habla, cuánto, a quién interrumpe y a quién "explica" no se
reparte por igual. El Explicómetro convierte esa intuición en **métricas reproducibles y
auditables**: cada número sale de un artefacto intermedio que se puede revisar, cada etiqueta
del LLM cita el texto que la justifica, y el juez no ve el género de quien habla.

## Inicio rápido

```bash
make setup       # Python 3.11, .venv, núcleo + dev + app
make dry-run     # CPU, clip sintético de 120 s → data/interim/dryrun_clip_120s/metrics.json
make test lint
make app         # visor Streamlit
```

En la infraestructura del reto (notebook OVHcloud con GPU + AI Endpoints), paso a paso en
**[`docs/ovh.md`](docs/ovh.md)**. Resumen:

```bash
make setup-gpu PYTHON=python3                     # en /workspace/equipo13/explicometro
cp .env.example .env                              # HF_TOKEN, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
make check-llm                                    # lista modelos del endpoint y prueba el juez
make run EPISODE=ep01 UNTIL=align CONFIG=configs/demo.yaml       # audio en data/raw/ep01.*
python scripts/speaker_samples.py --episode ep01  # qué dice cada SPK_xx → speakers_ep01.csv
make run EPISODE=ep01 CONFIG=configs/demo.yaml    # → data/interim/ep01/report.md
make app
```

Validación de modelos en una máquina con GPU (2× V100S):

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
| llm | `llm_events.json` (+ `llm_usage.json`) | juez Salamandra 7B cegado (OpenAI-compatible) o `lexicon` | stub |
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

## Componentes de terceros y licencias

El proyecto es **Apache-2.0**. Los pesos de los modelos **no** se redistribuyen: se descargan
de Hugging Face en la primera ejecución. Lista completa (versiones, uso y verificación) en
[`THIRD_PARTY.md`](THIRD_PARTY.md).

| Componente | Licencia | Uso |
|---|---|---|
| Silero VAD (vía faster-whisper) | MIT | detección de voz |
| pyannote.audio 3.1 · `speaker-diarization-3.1` · `segmentation-3.0` | MIT (modelos con condiciones de acceso en HF) | diarización |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | CC-BY-4.0 | embeddings de hablante (diarización) |
| faster-whisper / CTranslate2 · Whisper large-v3-turbo | MIT | ASR por defecto |
| BSC-LT `whisper-large-v3-LoS-punctuated`, `faster-whisper-bsc-large-v3-cat` | Apache-2.0 | ASR castellano/catalán (alternativos) |
| SpeechBrain · `spkrec-ecapa-voxceleb` · scikit-learn | Apache-2.0 · BSD-3 | clasificador de género (opcional) |
| BSC-LT `salamandra-7b-instruct-2606` | Apache-2.0 | juez C3/C4 por defecto (cualquier LLM OpenAI-compatible) |
| BSC-LT `MrBERT-es` | Apache-2.0 | embeddings para C2 |
| HiTZ `pyannote-segmentation-3.0-RTVE` | **CC-BY-NC** | segmentación alternativa, **desactivada por defecto** |
| ffmpeg | GPL (programa externo, no enlazado) | conversión de audio |
| pydantic, pandas, numpy, pyarrow, openai, streamlit | MIT / BSD / Apache-2.0 | núcleo y visor |

## Hoja de ruta

Detalle y decisiones tomadas (con la alternativa apuntada) en [`ROADMAP.md`](ROADMAP.md).

1. **Validar con datos reales:** anotar 2–3 episodios (gold set), `make eval` para C1 y
   moderador, `make calibrate-c2` para fijar τ y activar C2, `make ablation` (LLM vs léxico).
2. **Género sin metadatos:** entrenar el clasificador ECAPA + LR con Common Voice
   (`make train-gender`) y medir cobertura y error por hablante; ante la duda, `U`.
3. **Control del cegado del juez:** re-juzgar una muestra con las marcas de género gramatical
   neutralizadas y comprobar que las tasas por díada no cambian.
4. **Escala:** corpus de varios programas y cadenas, agregado con IC bootstrap
   (`make aggregate`) y visor público de resultados agregados (nunca de personas).
5. **Puesta en producción:** imagen de contenedor, LLM servido con vLLM o AI Endpoints, y
   ejecución por lotes en GPU.

## Estructura

```
src/explicometro/   schemas.py (contrato) · config.py · io.py · pipeline.py · cli.py · audio.py · chunking.py
                    ingest vad diarize asr align gender turns events metrics llm report
app/                visor Streamlit
configs/            default.yaml · demo.yaml (entrega/OVH) · episodes.example.yaml · gender_train.yaml
scripts/            download_episodes · run_pipeline · check_llm · speaker_samples · evaluate · aggregate
                    calibrate_c2 · ablation_llm · train_gender · make_fixture · make_tts_fixture
data/               raw/ interim/ processed/ (ignorados) · annotations/ (versionado)
tests/              pytest con casos sintéticos
docs/               architecture · metrics · llm · gender · ovh (infraestructura del reto)
```

Definiciones operativas de turnos, C1, moderador, tasas, ratios e IC:
[`docs/metrics.md`](docs/metrics.md). Agregado por corpus/cadena/año: `make aggregate`.

Juez LLM (cegado, validación, caché, coste), modo solo-léxico, C2 y calibración de τ:
[`docs/llm.md`](docs/llm.md).

Sobre el género de los hablantes (qué infiere el clasificador y qué no, prioridad de la
metadata pública, clase `U`): [`docs/gender.md`](docs/gender.md).

Ver [`CONTRIBUTING.md`](CONTRIBUTING.md), [`ROADMAP.md`](ROADMAP.md) y
[`THIRD_PARTY.md`](THIRD_PARTY.md).
