#!/usr/bin/env bash
#SBATCH --job-name=hsai-credit-probe
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/probe_training_credit.py \
  --checkpoint artifacts/models/arena-v1.4-r1/model.pt \
  --library artifacts/libraries/arena-cal-v14-snap \
  --out "${HSAI_CREDIT_OUT:?}" --device cuda:0 --games 128 --gradient-samples 3
