#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-event-labels
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/analyze_payoff_labels.py \
  --checkpoint artifacts/runs/nus-payoff-event-learning-20261118-1/model.pt \
  --dataset artifacts/runs/nus-payoff-dataset-20261118-1 \
  --plan config/research/payoff-event-suffix-20261118.json --policy event_student \
  --out "${HSAI_EVENT_LABELS_OUT:?}" --device cuda:0
