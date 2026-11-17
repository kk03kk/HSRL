#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-label-analysis
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
  --checkpoint artifacts/runs/nus-payoff-learning-20261118-2/model.pt \
  --dataset artifacts/runs/nus-payoff-dataset-20261118-1 \
  --out "${HSAI_LABEL_ANALYSIS_OUT:?}" --device cuda:0
