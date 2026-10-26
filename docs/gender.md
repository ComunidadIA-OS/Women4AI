# Género de los hablantes: qué medimos y qué no

## La limitación, en una frase

El clasificador **infiere una etiqueta binaria (H/M) a partir de rasgos acústicos de la voz**.
No mide la identidad de género de nadie, no reconoce identidades no binarias y se equivoca más
con voces que se alejan de los promedios de sus datos de entrenamiento (tono, edad, acento,
calidad de micrófono). Por eso:

1. **La metadata pública tiene prioridad.** Si el programa o la propia persona han hecho
   público su género, se usa eso (`gender_source = metadata`) y la voz no se consulta.
2. **Existe la clase `U` (desconocido)**, y se usa sin miedo:
   - metadata explícita "no binario" / "no consta" → `U`, y **no** se clasifica por la voz;
   - menos de 30 s de voz limpia (sin solapes con otras personas) → `U`;
   - confianza del clasificador < 0,8 → `U`.
3. **Las métricas por género excluyen `U`** (ni numerador ni denominador) y cada episodio
   informa de cuánto queda fuera en `metrics.json → unknown` (hablantes, segundos, % del tiempo
   de habla y % de palabras). Un resultado con mucho `U` es un resultado con poca cobertura,
   no un resultado neutro: hay que leerlo junto a ese porcentaje.

Las métricas del Explicómetro describen **patrones agregados de interacción** en programas
públicos. No sirven para etiquetar a personas concretas, y ningún output las nombra salvo que
`privacy.use_public_names: true` y el nombre venga de metadatos públicos.

## Fuentes, por orden de prioridad

| Prioridad | Fuente | `gender_source` | Cuándo |
|---|---|---|---|
| 1 | `data/annotations/speakers_<episodio>.csv` | `metadata` | `genero_publico` relleno |
| 2 | ECAPA + LogisticRegression | `model` | ≥ 30 s de voz limpia y confianza ≥ 0,8 |
| 3 | — | `model` (conf < 0,8) o `none` (< 30 s) | resto → `U` |

### CSV de metadatos

```csv
spk_id,nombre_publico,genero_publico
SPK_00,Nombre de la presentadora,M
SPK_01,,
SPK_02,Nombre,no binario
```

- `spk_id`: el ID anónimo de la diarización (escucha el audio para asignarlo). Si cambias la
  configuración de la diarización, revisa el CSV: los IDs pueden cambiar.
- `genero_publico`: `H`/`M` (también hombre/mujer, home/dona, male/female); `U`/no binario/no
  consta → `U` explícito; **vacío** → se usa el clasificador. Ojo: `M` es **mujer**.
- Solo información pública (créditos del programa, web de la cadena, perfil público de la
  persona). No se infiere género a partir del nombre.
- Editar el CSV invalida la caché de la etapa `gender` y de todo lo que depende de ella.

## Cómo se clasifica la voz

1. Voz limpia por hablante: sus segmentos de diarización menos los de cualquier otra persona,
   con 0,1 s de margen (los solapes mezclan dos voces).
2. Ventanas de 4 s (máx. 120, repartidas a lo largo del episodio) → embedding ECAPA
   (`speechbrain/spkrec-ecapa-voxceleb`) → normalización L2 → media → normalización L2.
3. LogisticRegression (con `StandardScaler`, `class_weight="balanced"`) → probabilidad.

## Entrenamiento y evaluación

`python scripts/train_gender.py` (config en `configs/gender_train.yaml`):

- **Datos:** Common Voice castellano y catalán, descargado de Mozilla Data Collective (desde
  octubre de 2025 Common Voice ya no se distribuye en Hugging Face). Solo contribuciones con
  género declarado masculino/femenino; el resto no se usa para entrenar un modelo binario.
- **Selección:** máx. 2.000 clips por clase repartidos entre idiomas, máx. 5 clips por hablante.
- **Split por hablante** (`client_id`): ninguna voz de test se ha visto en entrenamiento; una
  persona que graba en es y ca va entera al mismo lado.
- **Reporte** (`models/gender_lr.report.json`): accuracy, F1 por clase y matriz de confusión
  (1) por clip, (2) por hablante con embeddings promediados (como en inferencia) y (3) por
  hablante con el umbral 0,8, junto con la **cobertura** (cuántos quedarían en `U`).

Al publicar resultados, acompañadlos de este reporte y del % de tiempo en `U` de cada episodio.

## Sesgos conocidos y cómo mitigarlos

- **Dominio:** Common Voice es lectura de frases con micrófonos domésticos; las tertulias son
  habla espontánea y de estudio. Validad sobre episodios anotados (`speakers_*.csv` sirve de
  verdad) antes de fiaros de la accuracy de Common Voice.
- **Desequilibrio de voces:** pocas personas aportan muchos clips; el tope por hablante lo
  limita.
- **Edad y voces atípicas:** voces infantiles o de personas mayores, y voces que no encajan en
  los promedios, tienden a quedar con confianza baja → `U`. Es el comportamiento deseado.
- **Umbral:** subir `min_conf` reduce errores a costa de más `U`. El reporte de entrenamiento
  muestra el compromiso para 0,8.
