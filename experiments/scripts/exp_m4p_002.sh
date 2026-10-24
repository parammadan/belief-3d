#!/usr/bin/env bash
# EXP-M4P-002 (pre-registered in EXPERIMENT_REGISTRY.md): 40 new test seeds, dynamic/harsh.
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-cgl} PYTHONPATH=.
PY=${PY:-.venv/bin/python}
name="m4p2_test_dynamic_harsh"
rm -rf "datasets/$name" "preds/$name"
$PY scripts/generate_dataset.py --out "datasets/$name" --split test --offset 10 --episodes 40 --frames 15 \
    --noise harsh --scenario dynamic >/dev/null 2>&1
$PY scripts/run_m2_inference.py "datasets/$name" "preds/$name/base" >/dev/null 2>&1
for cams in overhead overhead,mobile; do
  for s in 0 1; do
    tag="${cams//,/+}_rot${s}"
    $PY scripts/retrack.py "preds/$name/base" "preds/$name/$tag" --calib-rot-deg "$s" --calib-trans-mm "$((5*s))" --cameras "$cams"
    $PY -m benchmarks.perception.eval_m2 "datasets/$name" "preds/$name/$tag" --out "reports/m4p/${name}_${tag}.json" >/dev/null
  done
  t="${cams//,/+}"
  echo "$t $($PY experiments/analysis/paired.py reports/m4p/${name}_${t}_rot0.json reports/m4p/${name}_${t}_rot1.json)"
done | tee reports/m4p/exp_m4p_002_paired.txt
