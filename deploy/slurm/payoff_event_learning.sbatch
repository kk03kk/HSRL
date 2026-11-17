#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-event-bc
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/learn_payoff_process.py \
  --checkpoint artifacts/models/arena-v1.4-r1/model.pt \
  --dataset artifacts/runs/nus-payoff-dataset-20261118-1 \
  --out "${HSAI_EVENT_LEARN_OUT:?}" --updates 2000 --lr 0.000003 \
  --per-kind-batch 32 --anchor-kl 10 --seed 1203801 --device cuda:0 --sampling mixed_events
