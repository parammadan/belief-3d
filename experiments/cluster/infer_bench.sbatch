#!/bin/bash
# GPU job (approved 2026-10-24): M7 inference benchmark. Short and time-capped; exits when done.
#SBATCH --job-name=b3d-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/madan.pa/belief3d/logs/bench_%j.log
set -euo pipefail
module load python/3.13.5
cd /scratch/madan.pa/belief3d/belief-3d
export PYTHONPATH=.
TRT_LIBS=$(../env/bin/python -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null || true)
# onnxruntime-gpu needs CUDA 12 + cuDNN 9 shared libs. They ship inside torch's nvidia/* wheels but
# are not on the loader path: run 11351691 silently fell back to CPUExecutionProvider without them.
NV_LIBS=$(ls -d ../env/lib/python3.13/site-packages/nvidia/*/lib 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH=${TRT_LIBS}:${NV_LIBS}${LD_LIBRARY_PATH:-}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p ../reports/m7
../env/bin/python -m benchmarks.performance.infer_bench --ckpt ../runs/pose_v1/best.pt \
    --val '../shards/val_*.npz' --out ../reports/m7/infer_bench.json
