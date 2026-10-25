#!/bin/bash
# CPU array (8 tasks): EXP-M8-002 coverage-calibration data. Task i = (split, scenario, noise):
# generate N episodes, run B1+B2 inference, dump per track-frame NEES rows. Deterministic; resumable.
#SBATCH --job-name=b3d-h2
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/madan.pa/belief3d/logs/h2_%A_%a.log
set -euo pipefail
module load python/3.13.5
cd /scratch/madan.pa/belief3d/belief-3d
export PYTHONPATH=. MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/scratch/madan.pa/belief3d/mesa24/lib:${LD_LIBRARY_PATH:-}
CONDS=(calib:occlusion:nominal calib:occlusion:harsh calib:dynamic:nominal calib:dynamic:harsh
       test:occlusion:nominal test:occlusion:harsh test:dynamic:nominal test:dynamic:harsh)
IFS=: read split sc nz <<< "${CONDS[$SLURM_ARRAY_TASK_ID]}"
off=$([ "$split" = test ] && echo "${TEST_OFFSET:?}" || echo "${CALIB_OFFSET:?}")
name=${split}_${sc}_${nz}
D=/scratch/madan.pa/belief3d/m8h2; R=../reports/m8/h2
mkdir -p $D $R
if [ -f $R/${name}_b2_rows.jsonl ]; then echo "done $name"; exit 0; fi
rm -rf $D/data_$name $D/preds_$name
../env/bin/python scripts/generate_dataset.py --out $D/data_$name --split $split --offset $off --episodes ${N:?} \
    --frames 15 --noise $nz --scenario $sc
../env/bin/python scripts/run_m4_inference.py $D/data_$name $D/preds_$name > $R/${name}_timing.json
for b in b1 b2; do
  ../env/bin/python -m benchmarks.perception.eval_m2 $D/data_$name $D/preds_$name/$b --out $R/${name}_$b.json \
      --rows $R/${name}_${b}_rows.jsonl.tmp > /dev/null
done
mv $R/${name}_b1_rows.jsonl.tmp $R/${name}_b1_rows.jsonl
mv $R/${name}_b2_rows.jsonl.tmp $R/${name}_b2_rows.jsonl
rm -rf $D/data_$name/ep_*/obs_*.npz
