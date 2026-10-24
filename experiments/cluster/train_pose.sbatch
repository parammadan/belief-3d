#!/bin/bash
# GPU job (approved by Param 2026-10-24). Time-capped; exits when training ends (no idle GPU).
# Smoke test first:  sbatch --time=00:10:00 --export=ALL,MAXSTEPS=60 train_pose.sbatch
#SBATCH --job-name=b3d-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/madan.pa/belief3d/logs/train_%j.log
set -euo pipefail
module load python/3.13.5   # the venv links against this module's libpython
cd /scratch/madan.pa/belief3d/belief-3d
export PYTHONPATH=.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
EXTRA=""
[ -n "${MAXSTEPS:-}" ] && EXTRA="--max-steps $MAXSTEPS --log-every 10"
../env/bin/python -m models.training.train_pose --train '../shards/train_*.npz' --val '../shards/val_*.npz' \
    --out ../runs/${RUN:-pose_v1} --epochs ${EPOCHS:-30} --batch 512 $EXTRA
