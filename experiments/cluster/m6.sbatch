#!/bin/bash
# CPU array: M6 closed-loop pick-and-place episodes. Task i runs EPS episodes starting at seed offset
# BASE + i*EPS, for all policies (paired), writing ../reports/m6/<TAG>/shard_<i>.jsonl. Resumable.
#SBATCH --job-name=b3d-m6
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/madan.pa/belief3d/logs/m6_%A_%a.log
set -euo pipefail
module load python/3.13.5
cd /scratch/madan.pa/belief3d/belief-3d
export PYTHONPATH=. MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/scratch/madan.pa/belief3d/mesa24/lib:${LD_LIBRARY_PATH:-}
i=$SLURM_ARRAY_TASK_ID
# sbatch --export splits on commas, so list-valued settings are passed colon-separated.
CAMS_=$(echo "${CAMS:-overhead:mobile}" | tr ':' ',')
POL_=$(echo "${POLICIES:-fixed:random:entropy:covariance:task_risk}" | tr ':' ',')
mkdir -p ../reports/m6/${TAG}
../env/bin/python scripts/run_m6.py --split ${SPLIT} --offset $((BASE + i*EPS)) --episodes ${EPS} \
    --scenario ${SCEN:-occlusion} --noise ${NOISE:-nominal} --budget ${BUDGET:-2} --cameras $CAMS_ \
    --policies $POL_ --max-attempts ${ATTEMPTS:-2} --out ../reports/m6/${TAG}/shard_$(printf %03d $i).jsonl
