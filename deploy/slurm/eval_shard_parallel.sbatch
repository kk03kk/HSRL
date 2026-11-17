#!/usr/bin/env bash
#SBATCH --job-name=hsai-gap-shard
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi15,xgpi17,xgpi11
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
# Compatibility entry; preferred entry is evaluate_suite.sbatch with a committed plan.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"
source deploy/slurm/common.sh
hsrl_require_compute
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/evaluate_legacy.py

