#!/usr/bin/env bash
# EXP-M4-001 (pre-registered in EXPERIMENT_REGISTRY.md). B1 vs B2 on new TEST seeds.
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-cgl} PYTHONPATH=.
PY=${PY:-.venv/bin/python}
mkdir -p reports/m4
for cond in "occlusion:nominal:30" "occlusion:harsh:30" "dynamic:nominal:10" "dynamic:harsh:10"; do
  IFS=: read sc nz n <<< "$cond"; name="test_m4_${sc}_${nz}"
  # resumable: deterministic runs, so a finished condition is skipped
  if [ -f "reports/m4/${name}_b1.json" ] && [ -f "reports/m4/${name}_b2.json" ]; then echo "skip $name (done)"; continue; fi
  rm -rf "datasets/$name" "preds/$name"
  $PY scripts/generate_dataset.py --out "datasets/$name" --split test --offset 100 --episodes "$n" --frames 15 \
      --noise "$nz" --scenario "$sc" >/dev/null 2>&1
  $PY scripts/run_m4_inference.py "datasets/$name" "preds/$name" > "reports/m4/${name}_timing.json" 2>/dev/null
  for b in b1 b2; do
    $PY -m benchmarks.perception.eval_m2 "datasets/$name" "preds/$name/$b" --out "reports/m4/${name}_${b}.json" >/dev/null
  done
  rm -rf "datasets/$name"/ep_*/obs_*.npz      # keep gt + preds for analysis; free disk
  echo "done $name"
done
