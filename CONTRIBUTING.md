# Cómo contribuir

## Puesta en marcha

```bash
make setup        # .venv con núcleo + dev + app (CPU)
make dry-run      # pipeline completo en CPU con stubs sobre un clip sintético de 120 s
make lint test    # lo mismo que pasa la CI
make setup-gpu    # solo en la máquina con GPU (V100S): añade los modelos
```

## Reglas del proyecto

1. **El contrato manda.** Los módulos solo intercambian tipos de `src/explicometro/schemas.py`
   (`AudioMeta`, `Segment`, `Word`, `Turn`, `Speaker`, `Event`, `EpisodeMetrics`). Si necesitas
   un campo nuevo, cámbialo ahí, sube `SCHEMA_VERSION` y explica el motivo en el PR.
2. **Todo configurable por YAML** (`configs/default.yaml`). Nada de rutas ni claves en el
   código; secretos solo en `.env` (`HF_TOKEN`, `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`).
3. **Cada etapa escribe un artefacto** en `data/interim/<episodio>/` y es re-ejecutable sola:
   `python -m explicometro run --episode <id> --only <etapa> --force`.
4. **GPU V100S:** fp16 o int8, nunca bf16 ni flash-attention.
5. **Privacidad:** hablantes como `SPK_00`; nada de nombres salvo metadatos públicos y
   `privacy.use_public_names: true`. No se sube audio al repo.
6. **Definiciones de métricas cerradas** (docs/metrics.md). No se cambian sin avisar al equipo.
7. **Licencias:** cada dependencia o modelo nuevo → fila en `THIRD_PARTY.md`.

## Código

- Python 3.11, tipado, docstrings breves.
- Funciones puras para métricas y reglas; tests con pytest sobre casos sintéticos.
- `ruff check` y `ruff format` limpios (`make format` los arregla).
- Si dudas entre dos opciones, elige la más simple que se pueda validar hoy y apunta la otra
  en `ROADMAP.md`.

## Añadir un backend real a una etapa

1. Implementa la función en el módulo (p. ej. `_silero` en `vad.py`) con imports *dentro* de
   la función, para que el núcleo siga instalándose sin GPU.
2. Regístrala en `BACKENDS` y documenta sus `params` en `configs/default.yaml`.
3. Añade un test con datos sintéticos (marca con `@pytest.mark.gpu` si necesita GPU).
4. Añade la fila de licencia en `THIRD_PARTY.md`.

## Commits y PRs

Commits pequeños con [Conventional Commits](https://www.conventionalcommits.org/es/):
`feat(vad): backend silero`, `fix(metrics): división por cero en asimetría`,
`docs: …`, `test: …`, `chore: …`. Un PR = un cambio revisable; la CI debe pasar.
