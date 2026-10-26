# Roadmap

Entrega: **30/10/2026 08:00**. Prioridad: funciona > validado > bonito.

## Siguiente (por orden)

- [ ] `vad`: backend Silero.
- [ ] `diarize`: pyannote 3.1 con segmentación `HiTZ/pyannote-segmentation-3.0-RTVE` (fp16).
- [ ] `asr`: faster-whisper large-v3, `word_timestamps=True`, `compute_type=float16`.
- [ ] `gender`: embeddings ECAPA + LogisticRegression; umbral `min_conf` → `U`.
- [ ] `llm`: cliente OpenAI-compatible, prompt C3/C4, salida JSON validada con pydantic y
      `evidence` comprobada como cita literal del turno.
- [ ] `events` C2: MrBERT-es + coseno sobre los 5 turnos previos del receptor; filtro de
      atribución explícita (`thresholds.reformulation.attribution_patterns`).
- [ ] Anotar 2–3 episodios y calibrar `cos_threshold` (C2).
- [ ] Agregación por programa: ratio de asimetría con IC 95 % por bootstrap (remuestreo de
      episodios).
- [ ] Rellenar `THIRD_PARTY.md` con versión y licencia verificadas de cada componente.

## Decisiones simples tomadas hoy (y alternativa apuntada)

| Decisión | Alternativa para más adelante |
|---|---|
| Alineación palabra→hablante por máximo solape | Re-segmentar turnos con los límites de palabra y la diarización (p. ej. WhisperX) |
| Turnos = palabras del mismo hablante con hueco ≤ 0,75 s | Turnos a partir de RTTM fusionando segmentos cortos |
| Moderador/a sin detectar (`protagonism.moderator = None`) | Heurística: más turnos cortos + preguntas + apertura/cierre; o metadato público |
| Género U si confianza < `min_conf` | Calibrar umbral con validación cruzada por programa |
| Fixture de audio sintético generado (no versionado) | Clip real con licencia abierta y hash publicado |
| Caché por clave de config + dependencias | Incluir hash del contenido de los artefactos de entrada |
| `streamlit` como extra `[app]` (no en núcleo) | Meterlo en el núcleo si el visor se vuelve la interfaz principal |
