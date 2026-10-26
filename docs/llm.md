# C2, C3 y C4: juez LLM y embeddings

## C3 y C4 con un LLM abierto (`llm.py`)

**Qué se envía.** Solo pares (A, B) donde B es el turno inmediatamente posterior a A, de otra
persona, con **≥ 20 palabras**, y A **no es pregunta** (si A preguntó, la explicación de B sí
estaba solicitada). Se manda el final de A y el principio de B (máx. 250 palabras cada uno).
"A es pregunta" = alguna de sus dos últimas frases contiene "?": depende de que el ASR puntúe
(LoS-punctuated y faster-whisper lo hacen).

**Prompt.** `src/explicometro/prompts/judge_es.txt`, sin cambios respecto al acordado. Cambiar el
prompt invalida la caché y la etapa `llm` (su hash entra en la clave).

**Cegado.** El juez recibe **solo los dos textos**: ni IDs de hablante, ni género, ni nombres de
metadatos. Un test comprueba que el mensaje enviado es exactamente plantilla + textos.

> **Límite del cegado.** El castellano y el catalán marcan el género en la gramática ("estoy
> segura", "som convidats"), y los textos pueden mencionar nombres. Cegar los metadatos impide
> que el juez *use* nuestra etiqueta, pero no garantiza que no *infiera* género del texto.
> Control propuesto (ROADMAP): re-juzgar una muestra con las marcas de género neutralizadas
> y comprobar que las tasas H→M / M→H no cambian de forma significativa.

**Validación.** JSON con pydantic (booleanos estrictos, `confianza` en [0, 1]). La `evidencia`
debe tener ≤ 20 palabras y ser **cita literal de B** (se compara sin tildes, mayúsculas ni
puntuación). Si falla, **un reintento** devolviendo el error al modelo; si vuelve a fallar, el
par queda como fallido (sin evento) y se cuenta en `llm_usage.json`. Temperatura 0.

**Eventos.** `explicacion_no_solicitada` → C3, `correccion_directa` → C4, ambos B → A, con
`confianza` como `score`; por debajo de `min_confidence` (0,5) no hay evento. La etiqueta
`reformula_idea_de_A` **no** genera C2 (C2 se define por embeddings); se guarda en
`llm_judgments.jsonl` y se evalúa en la ablación.

**Caché.** SQLite en `data/interim/_llm_cache.sqlite`, clave = hash de (modelo, hash del prompt,
temperatura, texto A, texto B). Re-ejecutar no vuelve a llamar al LLM. Los errores de red no
se cachean.

**Coste y latencia.** `data/interim/<ep>/llm_usage.json`: pares, aciertos de caché, llamadas,
reintentos, fallidos, tokens, latencia total/media/p95 y coste (con `price_*_per_mtok`; 0 para
vLLM propio). Cada juicio, con su latencia, en `llm_judgments.jsonl`.

### Dónde servir el modelo

`.env`: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` (tienen prioridad sobre el YAML).

- **vLLM en la V100S (fp16).** Ejemplo: `vllm serve BSC-LT/salamandra-7b-instruct-2606
  --dtype float16 --max-model-len 8192` y `LLM_BASE_URL=http://localhost:8000/v1`.
  Ojo: según una respuesta en el [foro de vLLM](https://discuss.vllm.ai/t/support-for-v100-sm-70-on-vllm-0-20/2605)
  (de su bot, no de la documentación oficial), desde la 0.20 los wheels no incluyen sm_70
  (V100) y la sugerencia es una 0.18.x o anterior. **Comprobadlo al instalar** y fijad la versión
  que funcione en la máquina. vLLM no está en las dependencias del proyecto: se instala aparte.
- **OVH AI Endpoints** (alternativa): API OpenAI-compatible en
  `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1` según la
  [documentación de Strands Agents sobre OVHcloud](https://strandsagents.com/docs/integrations/model-providers/ovhcloud-ai-endpoints/index.md);
  la clave se crea en el OVHcloud Manager. Comprobad qué modelos ofrece y su precio, y ponedlo
  en `price_input_per_mtok`/`price_output_per_mtok` para que el coste salga en el registro.

Si el servidor no admite `response_format=json_object`, el cliente lo detecta en la primera
llamada y sigue sin él (la validación con pydantic no cambia).

### Modo solo-léxico y ablación

`stages.llm.backend: lexicon` usa los mismos pares y etiqueta C3/C4 con expresiones regulares
(`lexicon_c3`, `lexicon_c4` en `LlmParams`). `python scripts/ablation_llm.py` compara ambos
modos contra el gold (P/R/F1 de C3 y C4) y, además, la etiqueta `reformula` del LLM contra el
C2 del gold. Resultado en `data/processed/ablation_llm.json`.

## C2 — reformulación sin atribución (`events.py`)

- Embeddings de **BSC-LT/MrBERT-es**: media de la última capa con máscara de atención y
  normalización L2, en fp16 en GPU (nunca bf16; atención `sdpa` o `eager`, sin flash-attention).
- Solo turnos con **≥ 15 palabras** (sin asentimientos). Cada turno de B se compara con los
  **5 turnos previos de otros hablantes** (que también cumplan el mínimo).
- Candidato si coseno ≥ τ y B **no atribuye** la idea (regex configurables: `como (decía|ha
  dicho|apuntaba) \w+`, `lo que dice \w+`, `tiene razón`, `té raó`, `com deia`). Un evento por
  turno de B, hacia el turno de A más parecido.
- **τ se calibra en el gold set**, no a ojo: mientras `cos_threshold` sea `null`, C2 no emite
  eventos y lo avisa. `python scripts/calibrate_c2.py` barre τ de 0,50 a 0,99, elige el de mayor
  F1 (en empate, el más alto) y guarda la curva en `data/processed/c2_calibration.json`. Con
  menos de 20 C2 en el gold avisa de que τ es poco fiable.
