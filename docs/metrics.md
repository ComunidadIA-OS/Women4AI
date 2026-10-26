# Turnos, conductas y métricas: definiciones operativas

Todo lo de aquí es configurable en `configs/default.yaml` (`stages.turns.params` y
`thresholds`). Las métricas de CLAUDE.md son definiciones cerradas: este documento solo
concreta lo que CLAUDE.md deja abierto.

## Turnos (`turns.py`)

- **Turno**: segmentos de diarización del mismo hablante separados por **< 0,3 s**, unidos.
  Turnos de personas distintas pueden solaparse (así se ven las interrupciones).
- Las palabras alineadas se asignan al turno de su hablante que contiene su punto medio.
- **Asentimiento** (`is_backchannel`): dura **< 1 s** o su texto, sin tildes ni puntuación, es
  una entrada del léxico (`sí, ya, claro, vale, ajá, exacto, clar, sí sí`). Un asentimiento
  no toma la palabra: no cuenta como turno en B ni participa en C1 (ni como A ni como B).

## C1 — interrupción competitiva (`events.py`)

B interrumpe a A si: B empieza mientras A habla (`A.inicio < B.inicio < A.fin`), se solapan
**≥ 0,5 s** y A **cede** (su turno termina) **≤ 2 s** después de que entre B. Si B pisa a varias
personas a la vez, se atribuye a la de mayor solape. Cada evento guarda:

- `src_spk` = B (quien interrumpe), `dst_spk` = A, `overlap_s`;
- `recovered`: **A recupera el turno** si el siguiente turno (sin asentimientos) de alguien
  distinto de B es de A y empieza ≤ 5 s después de que acabe el turno de B.

## Moderador

1. **Metadata**: columna opcional `rol` en `speakers_<episodio>.csv` (`moderador/a`,
   `presentador/a`, `invitado/a`). `moderador` lo fija; `invitado` lo excluye de la heurística.
2. **Heurística**: candidatas = quienes hablan en los 3 primeros **y** en los 3 últimos turnos.
   Gana quien más turnos **cede la palabra** (turno ≤ 15 s seguido de otro hablante en ≤ 2 s),
   con un mínimo de 3 y al menos 1,5 × lo que cede cualquier otra persona. Empate o ventaja
   pequeña → no hay moderador (mejor eso que uno equivocado).

Los eventos que **emite** el moderador (C1–C4) se marcan `by_moderator` y se reportan aparte
(`moderator_events`): moderar no es interrumpir. Los que **recibe** sí cuentan.

## Métricas por episodio (`metrics.py`)

Solo cuentan hablantes H y M; lo que es U se reporta en `unknown` (ver `docs/gender.md`).

| Bloque | Qué | Cómo |
|---|---|---|
| A | % hablantes, % tiempo, % palabras por género | sobre H+M |
| B | nº turnos y duración media por género | sin asentimientos (que se cuentan aparte) |
| B | abre / cierra | primer / último turno ≥ 10 s |
| B | modera | ver arriba, con su fuente (`metadata`/`heuristic`) |
| C | tasa por tipo (C1–C4) y díada (H→M, M→H, H→H, M→M) | eventos / minutos de habla del **género receptor**; se publica también el recuento y los minutos |
| — | exceso de interrupción | % C1 recibidas por M / % tiempo de habla de M (C1 con ambos géneros conocidos) |
| — | ratio de asimetría | tasa C1 H→M / tasa C1 H→H |

**Tamaño muestral**: cada ratio lleva `n_events` y `n_episodes`. Si `n_events < 10`, el ratio
se marca **`insuficiente`** y no se da el número. `sin_datos` = no hay eventos o el
denominador es cero (p. ej. ninguna interrupción H→H).

## Agregación por corpus / cadena / año (`scripts/aggregate.py`)

- Cada `metrics.json` guarda estadísticos suficientes (hablantes, segundos y palabras por
  género, recuentos por díada, turnos). Agregar = **sumar y recalcular**, no promediar ratios
  de episodios (un episodio corto con 1 evento no pesa lo mismo que uno de 2 h con 40).
- Los grupos salen de `configs/episodes.yaml` (`channel`, `program`, `year`...); un episodio
  sin entrada va a `desconocido`. Los dry-runs se ignoran.
- **IC 95 %** de exceso y asimetría: bootstrap percentil **remuestreando episodios** con
  reemplazo, 1.000 réplicas, semilla 13 (reproducible). Sin IC si hay < 2 episodios.

```bash
python scripts/aggregate.py                              # corpus, cadena, año, cadena×año
python scripts/aggregate.py --by program --by channel,year
```

Salidas en `data/processed/`: `metrics_<agrupación>.json`, `.csv` (tabla plana) y
`report_<agrupación>.md`.
