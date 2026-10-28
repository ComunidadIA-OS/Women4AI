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

.PHONY: ablation aggregate calibrate-c2 check-llm help setup setup-gpu train-gender fixture tts-fixture dry-run run eval app lint format test test-gpu clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## Crea .venv e instala núcleo + dev + app
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,app,audio]"
	@test -f .env || cp .env.example .env

setup-gpu: setup  ## Añade las dependencias de modelos (GPU)
	$(PY) -m pip install -e ".[gpu,train]"

$(FIXTURE):
	$(PY) scripts/make_fixture.py $(FIXTURE)

fixture: $(FIXTURE)  ## Genera el clip sintético de 120 s

tts-fixture:  ## Diálogo TTS de dos voces con verdad (TTS=flite|styletts2, TTS_CMD=..., VOICES=A=..,B=..)
	$(PY) scripts/make_tts_fixture.py --tts $(or $(TTS),flite) $(if $(TTS_CMD),--tts-cmd '$(TTS_CMD)') $(if $(VOICES),--voices $(VOICES))

dry-run: $(FIXTURE)  ## Pipeline completo en CPU con stubs sobre el fixture
	$(PY) scripts/run_pipeline.py --audio $(FIXTURE) --episode $(DRY_EP) --config $(CONFIG) --dry-run --force
	$(PY) -m explicometro validate data/interim/$(DRY_EP)/metrics.json

run:  ## Pipeline real: make run EPISODE=id [UNTIL=align] [AUDIO=ruta] (busca data/raw/<id>.*)
	@test -n "$(EPISODE)$(AUDIO)" || (echo "Uso: make run EPISODE=id [UNTIL=etapa] [AUDIO=ruta]" && exit 2)
	$(PY) scripts/run_pipeline.py $(if $(EPISODE),--episode $(EPISODE)) $(if $(AUDIO),--audio $(AUDIO)) $(if $(UNTIL),--until $(UNTIL)) --config $(CONFIG)

check-llm:  ## Lista modelos del endpoint LLM (.env) y prueba un juicio
	$(PY) scripts/check_llm.py --config $(CONFIG)

aggregate:  ## Métricas por corpus/cadena/año con IC bootstrap → data/processed/
	$(PY) scripts/aggregate.py --config $(CONFIG)

calibrate-c2:  ## Calibra τ de C2 contra el gold set
	$(PY) scripts/calibrate_c2.py --config $(CONFIG)

ablation:  ## C3/C4: LLM vs solo-léxico contra el gold set
	$(PY) scripts/ablation_llm.py --config $(CONFIG)

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

train-gender:  ## Entrena models/gender_lr.joblib con Common Voice (configs/gender_train.yaml)
	$(PY) scripts/train_gender.py --config configs/gender_train.yaml

test:  ## pytest (CPU; los tests GPU se saltan solos)
	$(PY) -m pytest

test-gpu:  ## Tests en la V100S (necesita extras [gpu] y HF_TOKEN)
	$(PY) -m pytest -m gpu -v

clean:  ## Borra artefactos intermedios y cachés (no toca data/raw ni annotations)
	rm -rf data/interim/* data/processed/* .pytest_cache .ruff_cache
	@touch data/interim/.gitkeep data/processed/.gitkeep
