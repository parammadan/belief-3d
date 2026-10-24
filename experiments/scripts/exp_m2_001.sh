#!/usr/bin/env bash
# EXP-M2-001: classical perception + B1 tracking on held-out TEST seeds.
# Conditions: scenario x noise; methods: overhead-only (fixed view) vs overhead+mobile.
# Reproduce:  bash experiments/scripts/exp_m2_001.sh   (needs scripts/fetch_assets.py first)
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-cgl} PYTHONPATH=.
PY=${PY:-.venv/bin/python}
EPS=${EPS:-10}; FRAMES=${FRAMES:-15}
mkdir -p reports/m2
for cond in "static:nominal" "occlusion:nominal" "dynamic:nominal" "dynamic:harsh"; do
  sc=${cond%%:*}; nz=${cond##*:}; name="m2_test_${sc}_${nz}"
  rm -rf "datasets/$name"
  $PY scripts/generate_dataset.py --out "datasets/$name" --split test --episodes "$EPS" --frames "$FRAMES" \
      --noise "$nz" --scenario "$sc" > "reports/m2/${name}_generation.json" 2>/dev/null
  for cams in overhead overhead,mobile; do
    tag=${cams//,/+}
    rm -rf "preds/$name/$tag"
    $PY scripts/run_m2_inference.py "datasets/$name" "preds/$name/$tag" --cameras "$cams" >/dev/null 2>&1
    cp "preds/$name/$tag/timing.json" "reports/m2/${name}_${tag}_timing.json"
    $PY -m benchmarks.perception.eval_m2 "datasets/$name" "preds/$name/$tag" --out "reports/m2/${name}_${tag}.json" >/dev/null
    echo "done $name $tag"
  done
done
