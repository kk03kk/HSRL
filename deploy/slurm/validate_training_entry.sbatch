#!/usr/bin/env bash
#SBATCH --job-name=hsai-entry-check
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source deploy/slurm/common.sh
hsrl_require_compute
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/validate_training_entry.py \
  --run artifacts/runs/nus-arena-v1.4-r1-20261117/train \
  --model artifacts/models/arena-v1.4-r1 --out "${HSAI_VALIDATION_OUT:?}"
