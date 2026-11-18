#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-fit-diagnostics
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/bin/python tools/analyze_payoff_labels.py \
  --checkpoint artifacts/runs/nus-payoff-event-learning-20261118-1/model.pt \
  --plan config/research/payoff-event-suffix-20261118.json --policy event_student \
  --dataset artifacts/runs/nus-payoff-queries-20261118-1 --split all \
  --out "${HSAI_PAYOFF_FIT_OUT:?}/event_student" --device cuda:0
.venv/bin/python tools/analyze_payoff_labels.py \
  --checkpoint artifacts/runs/nus-payoff-query-replay-20261118-1/model.pt \
  --plan config/research/payoff-query-replay-suffix-20261118.json --policy query_replay \
  --dataset artifacts/runs/nus-payoff-queries-20261118-1 --split all \
  --out "${HSAI_PAYOFF_FIT_OUT:?}/query_replay" --device cuda:0
exec .venv/bin/python tools/analyze_payoff_labels.py \
  --checkpoint artifacts/runs/nus-payoff-query-aggregate-20261118-1/model.pt \
  --plan config/research/payoff-query-aggregate-suffix-20261118.json --policy query_aggregate \
  --dataset artifacts/runs/nus-payoff-queries-20261118-1 --split all \
  --out "${HSAI_PAYOFF_FIT_OUT:?}/query_aggregate" --device cuda:0
