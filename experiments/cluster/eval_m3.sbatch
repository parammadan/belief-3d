#!/bin/bash
# CPU job: M3 pose comparison on the held-out TEST shards (no GPU needed for inference).
#SBATCH --job-name=b3d-eval-m3
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/madan.pa/belief3d/logs/eval_m3_%j.log
set -euo pipefail
module load python/3.13.5
cd /scratch/madan.pa/belief3d/belief-3d
export PYTHONPATH=. OMP_NUM_THREADS=4
../env/bin/python -m benchmarks.perception.eval_m3 --shards '../shards/test_*.npz' \
    --ckpt ../runs/${RUN:-pose_v1}/best.pt --out ../reports/m3/eval_m3_${RUN:-pose_v1}.json --limit ${LIMIT:-4000}
