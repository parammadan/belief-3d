#!/usr/bin/env bash
# VAL-M4P-001: choose the tracker's calibration-uncertainty prior on VAL seeds only.
set -euo pipefail
export MUJOCO_GL=${MUJOCO_GL:-cgl} PYTHONPATH=.
PY=${PY:-.venv/bin/python}
mkdir -p reports/m4p
for nz in nominal harsh; do
  name="val_dynamic_${nz}"
  rm -rf "datasets/$name" "preds/$name"
  $PY scripts/generate_dataset.py --out "datasets/$name" --split val --offset 200 --episodes 8 --frames 15 \
      --noise "$nz" --scenario dynamic >/dev/null 2>&1
  $PY scripts/run_m2_inference.py "datasets/$name" "preds/$name/base" >/dev/null 2>&1
  for cams in overhead overhead,mobile; do
    for s in 0 0.5 1 1.5 2 3; do
      tag="${cams//,/+}_rot${s}"
      tmm=$($PY -c "print(5*$s)")
      $PY scripts/retrack.py "preds/$name/base" "preds/$name/$tag" --calib-rot-deg "$s" --calib-trans-mm "$tmm" --cameras "$cams"
      $PY -m benchmarks.perception.eval_m2 "datasets/$name" "preds/$name/$tag" --out "reports/m4p/${name}_${tag}.json" >/dev/null
    done
  done
  echo "done $name"
done
