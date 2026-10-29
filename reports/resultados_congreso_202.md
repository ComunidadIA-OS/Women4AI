# Explicómetro — resultados reales: sesión de control del Congreso (30/09/2026)

Ejecución del 29/10/2026 en el notebook de OVHcloud del reto (2 × V100S). Todas las cifras salen de
`data/processed/metrics_corpus.json`, `data/interim/<episodio>/` y
`reports/congreso_202_resultados.json` (script `scripts/stats_congreso.py`).

## Datos

- **Fuente:** Congreso de los Diputados, Sesión Plenaria nº 202 (XV Legislatura), 30/09/2026, sesión de
  control al Gobierno. Vídeos oficiales de descarga del archivo audiovisual (`static.congreso.es`),
  versión original. Reutilización permitida citando la fuente y sin alterar el contenido.
- **Episodios:** 17 preguntas orales (una por episodio), 1,67 h de audio (6 029 s) y 5 282 s de habla.
  10 preguntas las formulan diputadas y 7 diputados; responden 4 ministras y 4 ministros.
- **Acceso:** `configs/congreso_202.json` (lista oficial de intervenciones) y `configs/episodes.yaml`
  (URL y SHA-256 de cada vídeo). `scripts/congreso_fetch.py` los descarga. El audio no se publica.
- **Género:** el **tratamiento oficial** que publica el Congreso (Diputada Dª / Diputado D., Ministra /
  Ministro). La voz no se usa para inferir el género: solo para enlazar cada SPK_xx con su persona,
  comparando su voz con sus intervenciones oficiales (`scripts/link_speakers.py`). La Presidencia se
  identifica por cómo la llaman («señora presidenta») y porque su voz es la misma en los 17 bloques
  (coseno 0,93–0,97; `scripts/link_chair.py`).

## Configuración (`configs/open.yaml`)

- Diarización pyannote.audio 3.1 con los hiperparámetros publicados de `speaker-diarization-3.1`,
  segmentación **HiTZ/pyannote-segmentation-3.0-RTVE** (`seg_CONF.ckpt`, CC-BY-NC, uso de investigación)
  y embeddings **pyannote/wespeaker-voxceleb-resnet34-LM** (CC-BY-4.0). No requiere token.
- ASR faster-whisper **large-v3-turbo** (castellano, fp16).
- C1 por reglas; C2 desactivada (τ sin calibrar); C3/C4 por **léxico** (sin juez LLM en esta ejecución).

## Calidad de las etapas

| Comprobación | Resultado |
|---|---|
| Hablantes enlazados con su persona | 51 de 51 (17 × 3); 0 s en «no consta» |
| Similitud voz ↔ intervenciones oficiales | coseno mínimo 0,69; margen mínimo frente a la otra persona 0,31 |
| Solapes detectados | 1 solape (0,2 s) en 168 cambios de turno; hueco mediano 1,6 s |
| DER / WER | **sin medir**: falta anotación manual (siguiente paso: Diario de Sesiones para el WER) |

## A. Presencia

| | Mujeres | Hombres |
|---|---|---|
| Tiempo de habla, con la Presidencia | 51,0 % | 49,0 % |
| Palabras, con la Presidencia | 49,6 % | 50,4 % |
| Tiempo de habla, solo participantes (sin la Presidencia) | **46,9 %** | 53,1 % |
| Palabras, solo participantes | 45,8 % | 54,2 % |

## B. Protagonismo

| | Mujeres | Hombres |
|---|---|---|
| Turnos | 137 | 48 |
| Duración media del turno | 19,6 s | 53,9 s |
| Abre el bloque (turno ≥ 10 s) | 8 | 9 |
| Cierra el bloque | 9 | 8 |
| Modera | 17 (Presidencia) | 0 |

Los turnos de las mujeres incluyen los de la Presidencia (cortos: dar y retirar la palabra).

Tiempo medio por persona y bloque (≈ 2,5 min reglamentarios):

| Rol | Mujeres | Hombres |
|---|---|---|
| Pregunta | 145,0 s (n = 10) | 145,3 s (n = 7) |
| Responde | 139,8 s (n = 6) | 142,8 s (n = 11) |

## C. Conductas y ratios

| Conducta | Resultado |
|---|---|
| C1 Interrupción competitiva | 0 eventos |
| C2 Reformulación sin atribución | desactivada (τ sin calibrar) |
| C3 Explicación no solicitada (léxico) | 1 evento M→H (0,023 / min) |
| C4 Corrección directa (léxico) | 0 eventos |
| Exceso de interrupción | sin datos (n = 0 eventos, 17 episodios) |
| Ratio de asimetría H→M / H→H | sin datos (n = 0 eventos, 17 episodios) |

## Lectura

1. En la sesión de control el tiempo lo fija el reglamento: cada persona habla unos 145 s por pregunta,
   sea mujer u hombre. La presencia de las mujeres (46,9 % del tiempo de los participantes) refleja
   **quién tiene asignada la pregunta**, no una diferencia de trato.
2. El audio oficial solo tiene un micrófono abierto cada vez (1 solape en 168 cambios de turno). Por eso
   **las interrupciones no se pueden medir en este formato**: 0 eventos no significa que no las haya en el
   hemiciclo, sino que la emisión no las recoge. Para medir el trato hacen falta tertulias o debates con
   habla cruzada.
3. Con 1 evento de conducta no hay base para ningún ratio: el sistema lo marca «sin datos» en lugar de
   dar un número.

## Coste y tiempo

- Cómputo total de las etapas: 598 s para 1,67 h de audio, unos 6 min por hora de audio. Las etapas
  más lentas son la preparación del audio (234 s), la diarización (203 s) y el ASR (141 s).
- A 1,93 €/h por V100S (facturación por minuto en OVHcloud AI Notebooks), cuesta **≈ 0,19 € por hora de
  audio** con una sola GPU. Con las dos GPU, las 17 preguntas se procesaron en unos 6 minutos.

## Limitaciones

- Un solo formato (parlamento moderado) y una sola sesión: no es representativo de las tertulias.
- La segmentación HiTZ-RTVE es CC-BY-NC: vale para investigación. En producción se usaría pyannote 3.1
  (requiere aceptar sus condiciones en Hugging Face).
- C3/C4 por léxico: es una línea base; el juez LLM no se usó en esta ejecución.
- Sin anotación manual no hay DER, WER ni F1 de eventos.
