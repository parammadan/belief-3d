#!/usr/bin/env bash
# VAL-M4-001: B1 vs B2 on VAL seeds (behaviour check before any test run).
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-cgl} PYTHONPATH=.
PY=${PY:-.venv/bin/python}
mkdir -p reports/m4
for cond in "occlusion:nominal" "dynamic:nominal" "occlusion:harsh" "dynamic:harsh"; do
  sc=${cond%%:*}; nz=${cond##*:}; name="val_m4_${sc}_${nz}"
  rm -rf "datasets/$name" "preds/$name"
  $PY scripts/generate_dataset.py --out "datasets/$name" --split val --offset 400 --episodes 6 --frames 15 \
      --noise "$nz" --scenario "$sc" >/dev/null 2>&1
  $PY scripts/run_m4_inference.py "datasets/$name" "preds/$name" > "reports/m4/${name}_timing.json" 2>/dev/null
  for b in b1 b2; do
    $PY -m benchmarks.perception.eval_m2 "datasets/$name" "preds/$name/$b" --out "reports/m4/${name}_${b}.json" >/dev/null
  done
  echo "done $name"
done
