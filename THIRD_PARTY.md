# Componentes de terceros

Registro de cada dependencia y modelo que usa el Explicómetro, con su licencia y para qué
se usa. Requisito de CLAUDE.md: **cualquier PR que añada una dependencia o un modelo
debe añadir su fila aquí** (verificando la licencia en la fuente oficial, no de memoria).

Columna *Verificado*: ✅ licencia comprobada en los metadatos del paquete publicado
(PyPI/wheel); ⚠️ pendiente de comprobar en la ficha del modelo en Hugging Face antes de la entrega.

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
| pyannote/speaker-diarization-3.1 | 3.1 | MIT (acceso con aceptación de condiciones) | diarize | ⚠️ |
| pyannote/segmentation-3.0 | 3.0 | MIT (acceso con aceptación de condiciones) | diarize (segmentación por defecto) | ⚠️ |
| pyannote/wespeaker-voxceleb-resnet34-LM | — | por comprobar | diarize (embeddings de hablante) | ⚠️ |
| HiTZ/pyannote-segmentation-3.0-RTVE | — | por comprobar | diarize (segmentación alternativa, flag en YAML) | ⚠️ |
| Whisper large-v3-turbo (CTranslate2) | — | por comprobar | asr | ⚠️ |
| BSC-LT/faster-whisper-bsc-large-v3-cat | — | por comprobar | asr (catalán) | ⚠️ |
| BSC-LT/whisper-large-v3-LoS-punctuated | — | por comprobar | asr (es/ca con puntuación) | ⚠️ |
| openai/whisper-large-v3 (solo `generation_config`) | — | por comprobar | asr: `alignment_heads` para timestamps por palabra | ⚠️ |
| BSC-LT/styletts2-spanish-multispeaker | — | por comprobar | solo tests: voces sintéticas con verdad conocida | ⚠️ |
| SpeechBrain | ≥1.0 | Apache-2.0 | gender: embeddings ECAPA | ✅ |
| scikit-learn | ≥1.5 | BSD-3-Clause | gender: LogisticRegression y métricas | ✅ |
| joblib | ≥1.4 | BSD-3-Clause | gender: guardar/cargar `models/gender_lr.joblib` | ✅ |
| datacollective (SDK de Mozilla Data Collective) | ≥0.7 | MPL-2.0 | scripts/train_gender.py: descarga de Common Voice | ✅ |
| speechbrain/spkrec-ecapa-voxceleb | — | por comprobar | gender: embeddings de hablante | ⚠️ |
| Common Voice Scripted Speech (es, ca) | 25.0 | por comprobar en MDC (históricamente CC0-1.0) | entrenamiento/evaluación del clasificador de género | ⚠️ |
