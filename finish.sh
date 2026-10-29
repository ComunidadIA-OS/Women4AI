#!/bin/bash
# Paso final: reintenta episodios fallidos, resuelve la Presidencia a nivel de sesión,
# recalcula género→informe con los CSV definitivos y agrega con IC bootstrap.
cd /workspace/equipo13/explicometro
source /workspace/equipo13/env/bin/activate
export HF_HOME=/workspace/equipo13/hf_cache PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
while ! grep -q ALL_DONE logs_run.txt; do sleep 5; done
for ep in $(grep FAIL logs_run.txt | awk '{print $2}'); do
  python scripts/run_pipeline.py --episode $ep --config configs/open.yaml --until align >> logs/$ep.log 2>&1 \
  && python scripts/link_speakers.py --episode $ep >> logs/$ep.log 2>&1 && echo "RETRY_OK $ep" || echo "RETRY_FAIL $ep"
done
python scripts/link_chair.py > logs/link_chair.log 2>&1 && echo CHAIR_OK || echo CHAIR_FAIL
EPS=$(python -c "import json;print(' '.join(b['episode'] for b in json.load(open('configs/congreso_202.json'))['blocks']))")
for ep in $EPS; do
  python scripts/run_pipeline.py --episode $ep --config configs/open.yaml --only gender turns llm events metrics report --force >> logs/$ep.log 2>&1 && echo "FINAL_OK $ep" || echo "FINAL_FAIL $ep"
done
python scripts/aggregate.py > logs/aggregate.log 2>&1 && echo AGG_OK || echo AGG_FAIL
echo FINISHED
