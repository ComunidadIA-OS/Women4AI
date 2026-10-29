#!/bin/bash
# Ejecución real: sesión de control del Congreso (30/09/2026), 17 preguntas, 2 GPU.
cd /workspace/equipo13/explicometro
source /workspace/equipo13/env/bin/activate
export HF_HOME=/workspace/equipo13/hf_cache PYTHONUNBUFFERED=1
mkdir -p logs
python scripts/build_congreso_manifest.py > logs/manifest.log 2>&1 || { echo MANIFEST_FAIL; exit 1; }
python scripts/congreso_fetch.py --manifest configs/congreso_202.json > logs/fetch.log 2>&1 || { echo FETCH_FAIL; exit 1; }
echo FETCH_OK
EPS=$(python -c "import json;print(' '.join(b['episode'] for b in json.load(open('configs/congreso_202.json'))['blocks']))")
run_ep () {
  ep=$1
  python scripts/run_pipeline.py --audio data/raw/$ep.wav --episode $ep --config configs/open.yaml --until align > logs/$ep.log 2>&1 \
  && python scripts/link_speakers.py --episode $ep >> logs/$ep.log 2>&1 \
  && python scripts/run_pipeline.py --episode $ep --config configs/open.yaml >> logs/$ep.log 2>&1 \
  && echo "OK $ep" || echo "FAIL $ep"
}
FIRST=$(echo $EPS | cut -d' ' -f1)
REST=$(echo $EPS | cut -d' ' -f2-)
CUDA_VISIBLE_DEVICES=0 run_ep $FIRST
i=0; A=""; B=""
for ep in $REST; do if [ $((i%2)) -eq 0 ]; then A="$A $ep"; else B="$B $ep"; fi; i=$((i+1)); done
( export CUDA_VISIBLE_DEVICES=0; for ep in $A; do run_ep $ep; done ) &
( export CUDA_VISIBLE_DEVICES=1; for ep in $B; do run_ep $ep; done ) &
wait
python scripts/aggregate.py > logs/aggregate.log 2>&1 && echo AGG_OK || echo AGG_FAIL
echo ALL_DONE
