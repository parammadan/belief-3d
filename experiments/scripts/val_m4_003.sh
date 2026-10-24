#!/usr/bin/env bash
# VAL-M4-003: B2 with consensus relative calibration + honest covariance, on the VAL-M4-001 datasets.
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-cgl} PYTHONPATH=.
PY=${PY:-.venv/bin/python}
for cond in occlusion_harsh dynamic_harsh occlusion_nominal dynamic_nominal; do
  name="val_m4_${cond}"
  rm -rf "preds/${name}_v3"
  $PY scripts/run_m4_inference.py "datasets/$name" "preds/${name}_v3" > "reports/m4/${name}_v3_timing.json" 2>/dev/null
  for b in b1 b2; do
    $PY -m benchmarks.perception.eval_m2 "datasets/$name" "preds/${name}_v3/$b" --out "reports/m4/${name}_v3_${b}.json" >/dev/null
  done
  echo "done $name"
done
