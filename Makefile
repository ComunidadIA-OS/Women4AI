# Explicómetro — atajos de desarrollo.
# Por defecto usa .venv/; en CI: `make test PY=python`.

PYTHON  ?= python3.11
VENV    ?= .venv
PY      ?= $(VENV)/bin/python
CONFIG  ?= configs/default.yaml
FIXTURE ?= tests/fixtures/clip_120s.wav
DRY_EP  ?= dryrun_clip_120s
AUDIO   ?=
EPISODE ?=

.PHONY: help setup setup-gpu fixture dry-run run eval app lint format test clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Crea .venv e instala núcleo + dev + app
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,app]"
	@test -f .env || cp .env.example .env

setup-gpu: setup  ## Añade las dependencias de modelos (GPU)
	$(PY) -m pip install -e ".[gpu]"

$(FIXTURE):
	$(PY) scripts/make_fixture.py $(FIXTURE)

fixture: $(FIXTURE)  ## Genera el clip sintético de 120 s

dry-run: $(FIXTURE)  ## Pipeline completo en CPU con stubs sobre el fixture
	$(PY) scripts/run_pipeline.py --audio $(FIXTURE) --episode $(DRY_EP) --config $(CONFIG) --dry-run --force
	$(PY) -m explicometro validate data/interim/$(DRY_EP)/metrics.json

run:  ## Pipeline real: make run AUDIO=data/raw/x.wav [EPISODE=id]
	@test -n "$(AUDIO)" || (echo "Uso: make run AUDIO=ruta [EPISODE=id]" && exit 2)
	$(PY) scripts/run_pipeline.py --audio $(AUDIO) $(if $(EPISODE),--episode $(EPISODE)) --config $(CONFIG)

eval:  ## Evalúa eventos contra data/annotations/
	$(PY) scripts/evaluate.py --config $(CONFIG)

app:  ## Lanza el visor Streamlit
	$(PY) -m streamlit run app/streamlit_app.py

lint:  ## ruff check + formato
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:  ## Aplica ruff format y autofix
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

test:  ## pytest (CPU)
	$(PY) -m pytest

clean:  ## Borra artefactos intermedios y cachés (no toca data/raw ni annotations)
	rm -rf data/interim/* data/processed/* .pytest_cache .ruff_cache
	@touch data/interim/.gitkeep data/processed/.gitkeep
