# Ejecutar en la infraestructura del reto (OVHcloud)

Guía para procesar un episodio real en el **notebook con GPU de OVHcloud** y usar
**OVHcloud AI Endpoints** como juez LLM de C3/C4. Tiempo estimado: 30–60 min la primera vez
(descarga de modelos), unos minutos por episodio de 10–20 min después.

> Los tokens del notebook y de AI Endpoints son **del equipo y confidenciales**: van solo en
> `.env` (que git ignora). Nunca en el código, en `configs/` ni en commits.

## 0. Antes de empezar (una vez, desde el navegador)

1. Cuenta en Hugging Face → acepta las condiciones de
   [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) y
   [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0), y crea un
   token de lectura (*Settings → Access Tokens*).
2. Elige un fragmento de debate o tertulia **con hombres y mujeres** (5–20 min) cuya
   descarga y uso para investigación estén permitidos. No se sube al repositorio.

## 1. Abrir el notebook y un terminal

Abre el enlace del notebook → *Login with token* → pega el token del equipo. Abre un
terminal (guía en `/workspace/compartido/LEEME.ipynb`). **Trabaja siempre en
`/workspace/equipo13`**: es lo único que se conserva.

```bash
cd /workspace/equipo13
git clone <URL del repo> explicometro && cd explicometro
nvidia-smi                       # debe verse la GPU
python3 --version                # ≥ 3.11
which ffmpeg || sudo apt-get install -y ffmpeg || conda install -y -c conda-forge ffmpeg
```

## 2. Instalar

```bash
make setup-gpu PYTHON=python3    # .venv con torch, pyannote, faster-whisper, transformers
cp .env.example .env             # si no lo ha creado ya `make setup`
nano .env
```

Rellena en `.env`:

```
HF_TOKEN=hf_...                                   # paso 0
LLM_BASE_URL=https://oai.endpoints.kepler.ai.cloud.ovh.net/v1
LLM_API_KEY=<token de AI Endpoints del equipo>
LLM_MODEL=<un modelo de la lista del paso 3>
```

## 3. Comprobar el juez LLM

```bash
make check-llm
```

Lista los modelos que ofrece el endpoint y juzga un par de prueba con el prompt real. Copia en
`LLM_MODEL` el identificador exacto de un modelo **abierto** (pesos públicos) de la lista y
repite hasta ver `ok=True`. Si el endpoint no funciona, el pipeline sigue sin LLM: en
`configs/demo.yaml` cambia `llm.backend: openai_compatible` por `lexicon`.

## 4. Procesar el episodio (hasta la alineación)

```bash
mkdir -p data/raw && cp /ruta/al/audio.mp3 data/raw/ep01.mp3
make run EPISODE=ep01 UNTIL=align CONFIG=configs/demo.yaml
```

Quedan en `data/interim/ep01/`: `vad.json`, `diarization.rttm`, `transcript.json`
(palabras con hablante `SPK_xx`) y `timings.jsonl`. Si la GPU se queda sin memoria, baja
`segmentation_batch_size`/`embedding_batch_size` en `configs/demo.yaml`.

## 5. Asignar el género con metadatos públicos

Mira qué dice cada `SPK_xx` (`make app`, o abre `transcript.json`) y crea
`data/annotations/speakers_ep01.csv`:

```csv
spk_id,nombre_publico,genero_publico,rol
SPK_00,,M,moderador
SPK_01,,H,
SPK_02,,M,
```

Género **solo** si es público (créditos del programa, web de la cadena); si no, `U`. El
nombre es opcional y solo se muestra con `privacy.use_public_names: true`. Detalle en
[`gender.md`](gender.md).

## 6. Terminar el pipeline y ver resultados

```bash
make run EPISODE=ep01 CONFIG=configs/demo.yaml
make app                                   # visor Streamlit
```

Resultados: `data/interim/ep01/report.md` (tablas A, B, C por díada y ratios),
`metrics.json`, `llm_usage.json` (llamadas, tokens, latencia y coste). Para varios episodios:
`make aggregate` (IC 95 % por bootstrap).

## Qué se publica y qué no

| Sí (repositorio) | No |
|---|---|
| código, configs, docs, `speakers_*.csv` con datos públicos | audio (`data/raw/`) |
| `metrics.json` / `report.md` si el equipo lo decide | `.env`, tokens, `data/interim/` |
| anotaciones en `data/annotations/` | modelos descargados (`models/`) |
