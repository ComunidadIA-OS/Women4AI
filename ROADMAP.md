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
- [x] `llm`: juez cegado C3/C4 (JSON validado, reintento, caché, coste) y modo `lexicon`.
- [x] `events` C2: MrBERT-es + coseno con los 5 turnos previos de otros + regex de atribución.
- [ ] Servir el juez (vLLM con una versión que soporte sm_70, u OVH) y ejecutar `make ablation`.
- [ ] Calibrar τ de C2 con el gold set (`make calibrate-c2`) y ponerlo en el YAML.
- [ ] Control del cegado: re-juzgar una muestra neutralizando marcas de género gramatical y
      comprobar que las tasas por díada no cambian.
- [x] `turns`, `events` (C1, moderador) y `metrics` (episodio + agregado con IC bootstrap).
- [ ] Validar la heurística de moderador y C1 con 2–3 episodios anotados (`make eval`).
- [ ] Anotar 2–3 episodios y calibrar `cos_threshold` (C2).

## Decisiones simples tomadas hoy (y alternativa apuntada)

| Decisión | Alternativa para más adelante |
|---|---|
| Alineación palabra→hablante por máximo solape | Re-segmentar turnos con los límites de palabra y la diarización (p. ej. WhisperX) |
| Turnos = segmentos de diarización del mismo hablante con hueco < 0,3 s | Turnos por palabras del ASR (más sensibles a errores de timestamps) |
| "A recupera" = siguiente turno (no de B) es de A y llega ≤ 5 s tras acabar B | Medir el tiempo hasta que A vuelve a tener la palabra, sin umbral |
| Moderador por cesiones + apertura/cierre, con margen 1,5× | Clasificador de roles con rasgos léxicos (preguntas, nombres de turno) |
| Exceso de interrupción solo con díadas conocidas (U fuera) | Contar también C1 emitidas por U hacia M |
| Agregado = suma de estadísticos suficientes (ratio de totales) | Modelo de efectos mixtos por programa |
| Par (A, B) = turno inmediatamente anterior de otra persona | Ventana de varios turnos previos (B responde a alguien que habló antes) |
| "A es pregunta" por signos de interrogación en sus 2 últimas frases | Clasificador de actos de habla (no depende de la puntuación del ASR) |
| `reformula_idea_de_A` del LLM solo para la ablación | Combinar embeddings + LLM para C2 si la ablación lo justifica |
| Caché del juez en SQLite local | Caché compartida entre máquinas del equipo |
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
