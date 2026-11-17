#!/usr/bin/env bash
#SBATCH --job-name=hsai-eval-suite
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/evaluate_suite.py \
  --plan "${HSAI_EVAL_PLAN:?}" --out "${HSAI_EVAL_OUT:?}" --device cuda:0
