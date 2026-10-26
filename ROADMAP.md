# Roadmap

Entrega: **30/10/2026 08:00**. Prioridad: funciona > validado > bonito.

## Siguiente (por orden)

- [x] `ingest` ffmpeg + loudnorm 2 pasadas, `vad` Silero, `diarize` pyannote 3.1 (+ flag RTVE),
      `asr` (3 modelos), `align` con regla de solape. Validado en CPU; **falta `make test-gpu` en la V100S**.
- [ ] En la V100S: comparar segmentación 3.1 vs RTVE con `test_gpu_until_align_against_truth`
      (y con un episodio anotado) y fijar el valor por defecto de `use_rtve_segmentation`.
- [ ] Generar el diálogo de test con StyleTTS2 (`make tts-fixture TTS=styletts2 ...`) para medir WER.
- [ ] Verificar en Hugging Face las licencias marcadas ⚠️ en `THIRD_PARTY.md`.
- [x] `gender`: metadata CSV > ECAPA + LogisticRegression (≥ 30 s de voz limpia, conf ≥ 0,8) > `U`.
- [ ] Entrenar `models/gender_lr.joblib` en la máquina GPU (`make train-gender`, necesita
      `MDC_API_KEY` y el `dataset_id` de Common Voice catalán en `configs/gender_train.yaml`).
- [ ] Validar el clasificador sobre episodios con `speakers_*.csv` relleno (dominio real).
- [ ] (Opcional) Comparar con un ajuste ligero de `BSC-LT/hubert-base-los-6k`: congelar el
      encoder, pooling medio + cabeza lineal, mismo split por hablante y mismo reporte; usarlo
      solo si mejora la cobertura a umbral 0,8 sin bajar la accuracy por hablante.
- [ ] `llm`: cliente OpenAI-compatible, prompt C3/C4, salida JSON validada con pydantic y
      `evidence` comprobada como cita literal del turno.
- [ ] `events` C2: MrBERT-es + coseno sobre los 5 turnos previos del receptor; filtro de
      atribución explícita (`thresholds.reformulation.attribution_patterns`).
- [ ] Anotar 2–3 episodios y calibrar `cos_threshold` (C2).
- [ ] Agregación por programa: ratio de asimetría con IC 95 % por bootstrap (remuestreo de
      episodios).

## Decisiones simples tomadas hoy (y alternativa apuntada)

| Decisión | Alternativa para más adelante |
|---|---|
| Alineación palabra→hablante por máximo solape | Re-segmentar turnos con los límites de palabra y la diarización (p. ej. WhisperX) |
| Turnos = palabras del mismo hablante con hueco ≤ 0,75 s | Turnos a partir de RTTM fusionando segmentos cortos |
| Moderador/a sin detectar (`protagonism.moderator = None`) | Heurística: más turnos cortos + preguntas + apertura/cierre; o metadato público |
| Género U si confianza < `min_conf` (0,8) o < 30 s de voz limpia | Calibrar umbral con episodios anotados por programa |
| Clasificador de género con embeddings ECAPA congelados + LogisticRegression | Ajuste ligero de HuBERT (BSC-LT/hubert-base-los-6k) |
| Common Voice leído en streaming desde el `.tar.gz` (dos pasadas) | Pedir a MDC un subconjunto o usar las entregas *delta* |
| Fixture de audio sintético generado (no versionado) | Clip real con licencia abierta y hash publicado |
| Caché por clave de config + dependencias | Incluir hash del contenido de los artefactos de entrada |
| `streamlit` como extra `[app]` (no en núcleo) | Meterlo en el núcleo si el visor se vuelve la interfaz principal |
| Diarización sobre el episodio entero (ventanas internas de pyannote, memoria por `*_batch_size`) | Trocear en 10 min y enlazar hablantes entre trozos por embeddings (necesario si un episodio no cabe en RAM) |
| Silero VAD vía el ONNX que trae faster-whisper (CPU, sin torch, sin descargas) | Paquete `silero-vad` con torch (mismo modelo) |
| ASR silenciando la no-voz del VAD (timestamps intactos) | `clip_timestamps` de faster-whisper o recortar y re-alinear segmentos |
| Fronteras de trozo ajustadas al silencio más cercano (±margen) + fusión por punto medio | Fusión por alineación de texto en la zona de margen |
| `loudnorm` en dos pasadas, lineal, mezclando a mono antes de medir | `dynaudnorm` si hay grandes diferencias de micrófono entre tertulianos |
| Diálogo de test con flite (fonética inglesa) cuando no hay StyleTTS2 | Solo StyleTTS2 o grabaciones reales con licencia |
| pyannote en fp32 (no soporta fp16 de forma fiable) | Probar `torch.autocast(fp16)` en embeddings si el tiempo de diarización es un cuello de botella |
