#!/usr/bin/env bash
# EXP-M4P-001: calibration-uncertainty prior (sigma_rot=1 deg, sigma_trans=5 mm, chosen on VAL by
# VAL-M4P-001) evaluated once on the same TEST seeds as EXP-M2-001. sigma=0 must reproduce EXP-M2-001.
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-cgl} PYTHONPATH=.
PY=${PY:-.venv/bin/python}
mkdir -p reports/m4p
for cond in "static:nominal" "occlusion:nominal" "dynamic:nominal" "dynamic:harsh"; do
  sc=${cond%%:*}; nz=${cond##*:}; name="m2_test_${sc}_${nz}"
  rm -rf "datasets/$name" "preds/$name"
  $PY scripts/generate_dataset.py --out "datasets/$name" --split test --episodes 10 --frames 15 \
      --noise "$nz" --scenario "$sc" >/dev/null 2>&1
  $PY scripts/run_m2_inference.py "datasets/$name" "preds/$name/base" >/dev/null 2>&1
  for cams in overhead overhead,mobile; do
    for s in 0 1; do
      tag="${cams//,/+}_rot${s}"
      $PY scripts/retrack.py "preds/$name/base" "preds/$name/$tag" --calib-rot-deg "$s" --calib-trans-mm "$((5*s))" --cameras "$cams"
      $PY -m benchmarks.perception.eval_m2 "datasets/$name" "preds/$name/$tag" --out "reports/m4p/test_${sc}_${nz}_${tag}.json" >/dev/null
    done
  done
  echo "done $name"
done
