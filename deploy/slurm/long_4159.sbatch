#!/usr/bin/env bash
#SBATCH --job-name=hsrl-4159-long
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=1-00:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1 HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1
hsrl_gpu_job python -u workflows/launch_nus_refinement.py --task long-4159 \
  --out "${HSRL_LONG_OUT:-artifacts/runs/nus-4159-economy-long-002}"
