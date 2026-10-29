# Componentes de terceros

Registro de cada dependencia y modelo que usa el Explicómetro, con su licencia y para qué
se usa. Requisito del proyecto: **cualquier PR que añada una dependencia o un modelo
debe añadir su fila aquí** (verificando la licencia en la fuente oficial, no de memoria).

Columna *Verificado*: ✅ licencia comprobada en los metadatos del paquete publicado
(PyPI/wheel); ✅ HF = comprobada en la ficha del modelo en Hugging Face (28/10/2026);
⚠️ pendiente.

| Componente | Versión | Licencia | Uso | Verificado |
|---|---|---|---|---|
| ffmpeg (paquete Ubuntu, con libflite) | 6.1 | GPL-2.0+ (build `--enable-gpl`); se usa como programa externo, no se enlaza | ingest: conversión a 16 kHz mono y `loudnorm`; voces flite del fixture de test | ✅ |
| numpy | ≥1.26 | BSD-3-Clause (+ componentes 0BSD, MIT, Zlib, CC0) | lectura de audio por ventanas, métricas de evaluación | ✅ |
| faster-whisper | ≥1.1,<2 | MIT | asr (backend `faster_whisper`) y Silero VAD empaquetado (ONNX) | ✅ |
| CTranslate2 | 4.x (dep. de faster-whisper) | MIT | inferencia Whisper en fp16/int8_float16 | ✅ |
| onnxruntime | 1.x (dep. de faster-whisper) | MIT | inferencia de Silero VAD | ✅ |
| Silero VAD (modelo ONNX v6 incluido en faster-whisper) | v6 | MIT | vad | ✅ (clasificador MIT del paquete `silero-vad`) |
| pyannote.audio | 3.1.x | MIT | diarize | ✅ |
| pyannote.core | 5.x | MIT | estructuras de diarización | ✅ |
| PyTorch | ≥2.3 | BSD-3-Clause y otras permisivas (Apache-2.0, MIT, BSL-1.0) | runtime de pyannote y transformers | ✅ |
| transformers | ≥4.44 | Apache-2.0 | asr (backend `transformers`) | ✅ |
| pyannote/speaker-diarization-3.1 | 3.1 | MIT (gated: aceptar condiciones en HF) | diarize | ✅ HF |
| pyannote/segmentation-3.0 | 3.0 | MIT (gated: aceptar condiciones en HF) | diarize (segmentación por defecto) | ✅ (fuente secundaria; confirmar en HF) |
| pyannote/wespeaker-voxceleb-resnet34-LM | — | CC-BY-4.0 (atribución) | diarize (embeddings de hablante) | ✅ HF |
| HiTZ/pyannote-segmentation-3.0-RTVE | — | **CC-BY-NC** (no comercial; la ficha muestra 4.0 y el README 3.0) | diarize (segmentación alternativa, **desactivada por defecto**) | ✅ HF — ver nota |
| Whisper large-v3-turbo (CTranslate2, `mobiuslabsgmbh/faster-whisper-large-v3-turbo`) | — | MIT | asr (por defecto) | ✅ HF |
| BSC-LT/faster-whisper-bsc-large-v3-cat | — | Apache-2.0 | asr (catalán) | ✅ HF |
| BSC-LT/whisper-large-v3-LoS-punctuated | — | Apache-2.0 | asr (es/ca con puntuación) | ✅ HF |
| openai/whisper-large-v3 (solo `generation_config`) | — | Apache-2.0 | asr: `alignment_heads` para timestamps por palabra | ✅ HF |
| BSC-LT/styletts2-spanish-multispeaker | — | Apache-2.0 | solo tests: voces sintéticas con verdad conocida | ✅ HF |
| SpeechBrain | ≥1.0 | Apache-2.0 | gender: embeddings ECAPA | ✅ |
| scikit-learn | ≥1.5 | BSD-3-Clause | gender: LogisticRegression y métricas | ✅ |
| joblib | ≥1.4 | BSD-3-Clause | gender: guardar/cargar `models/gender_lr.joblib` | ✅ |
| datacollective (SDK de Mozilla Data Collective) | ≥0.7 | MPL-2.0 | scripts/train_gender.py: descarga de Common Voice | ✅ |
| speechbrain/spkrec-ecapa-voxceleb | — | Apache-2.0 | gender: embeddings de hablante | ✅ HF |
| Common Voice Scripted Speech (es, ca) | 25.0 | por comprobar en MDC (históricamente CC0-1.0) | entrenamiento/evaluación del clasificador de género | ⚠️ |
| openai (SDK de Python) | ≥1.40 | Apache-2.0 | llm: cliente OpenAI-compatible (vLLM / OVH AI Endpoints) | ✅ |
| vLLM (servidor, se instala aparte) | la que soporte sm_70 | Apache-2.0 | servir el juez en la V100S | ✅ |
| BSC-LT/salamandra-7b-instruct-2606 | — | Apache-2.0 | llm: juez de C3/C4 | ✅ HF |
| BSC-LT/MrBERT-es | — | Apache-2.0 | events: embeddings de C2 | ✅ HF |

## Notas de compatibilidad de licencias

- Todo lo que se usa **por defecto** es MIT, Apache-2.0, BSD o CC-BY-4.0, compatible con la
  licencia Apache-2.0 del proyecto. CC-BY-4.0 (wespeaker) solo exige atribución: está en esta tabla.
- **HiTZ/pyannote-segmentation-3.0-RTVE es CC-BY-NC (no comercial).** No se distribuye con el
  proyecto y está **desactivada por defecto** (`use_rtve_segmentation: false`). Quien la active
  acepta su licencia y no puede usar el resultado con fines comerciales.
- Los pesos de los modelos **no se redistribuyen** en este repositorio: se descargan de
  Hugging Face en la primera ejecución, cada uno con su licencia.
- Common Voice solo se usa para **entrenar** el clasificador de género; ni el dataset ni el
  clasificador entrenado (`models/`) se publican.
- **LLM por API (OVHcloud AI Endpoints):** el cliente es OpenAI-compatible y agnóstico al
  modelo (`LLM_BASE_URL`, `LLM_MODEL`); el modelo por defecto es abierto (Salamandra, Apache-2.0).
