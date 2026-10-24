#!/bin/bash
# CPU array job: neural-pose training/val point-cloud shards (no GPU).
#SBATCH --job-name=b3d-gen
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --array=0-27
#SBATCH --output=/scratch/madan.pa/belief3d/logs/gen_%A_%a.log
set -euo pipefail
cd /scratch/madan.pa/belief3d/belief-3d
export PYTHONPATH=. MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${MUJOCO_GL:-egl}
i=$SLURM_ARRAY_TASK_ID
# 20 train shards (100 eps each), 4 val shards (50 eps), 4 test shards (50 eps); disjoint seed ranges.
SC=(dynamic occlusion varied dynamic); NZ=(nominal nominal harsh harsh)
if [ "$i" -lt 20 ]; then
  ../env/bin/python -m data.generators.pointcloud_samples --split train --offset $((i*100)) --episodes 100 \
      --frames 10 --scenario ${SC[$((i%4))]} --noise ${NZ[$((i%4))]} --out ../shards/train_$(printf %04d $i).npz
elif [ "$i" -lt 24 ]; then
  j=$((i-20))
  ../env/bin/python -m data.generators.pointcloud_samples --split val --offset $((1000+j*50)) --episodes 50 \
      --frames 10 --scenario ${SC[$j]} --noise ${NZ[$j]} --out ../shards/val_$(printf %04d $j).npz
else
  j=$((i-24))
  ../env/bin/python -m data.generators.pointcloud_samples --split test --offset $((3000+j*50)) --episodes 50 \
      --frames 10 --scenario ${SC[$j]} --noise ${NZ[$j]} --out ../shards/test_$(printf %04d $j).npz
fi
