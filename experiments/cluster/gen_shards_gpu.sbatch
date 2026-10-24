#!/bin/bash
# Fallback when CPU nodes cannot render headlessly: ONE short GPU job renders all shards
# via NVIDIA EGL, running 8 generator processes concurrently on the same GPU. It exits
# when all shards are written (no idle GPU). GPU use approved by Param 2026-10-24.
#SBATCH --job-name=b3d-gen-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/madan.pa/belief3d/logs/gen_gpu_%j.log
set -uo pipefail
module load python/3.13.5
cd /scratch/madan.pa/belief3d/belief-3d
export PYTHONPATH=. MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
SC=(dynamic occlusion varied dynamic); NZ=(nominal nominal harsh harsh)
run() {  # $1 split  $2 offset  $3 episodes  $4 idx  $5 name
  ../env/bin/python -m data.generators.pointcloud_samples --split $1 --offset $2 --episodes $3 --frames 10 \
      --scenario ${SC[$(($4%4))]} --noise ${NZ[$(($4%4))]} --out ../shards/$5.npz > ../logs/$5.log 2>&1
}
jobs_list=()
for i in $(seq 0 19); do jobs_list+=("train $((i*100)) 100 $i train_$(printf %04d $i)"); done
for j in 0 1 2 3; do jobs_list+=("val $((1000+j*50)) 50 $j val_$(printf %04d $j)"); done
for j in 0 1 2 3; do jobs_list+=("test $((3000+j*50)) 50 $j test_$(printf %04d $j)"); done
for spec in "${jobs_list[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge 8 ]; do sleep 2; done
  run $spec &
done
wait
ls ../shards/*.npz | wc -l
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
