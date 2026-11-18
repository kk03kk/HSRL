#!/usr/bin/env bash
#SBATCH --job-name=hsai-fresh-history-analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/analyze_economy_suffix.py \
  --runs artifacts/runs/nus-fresh-history-parent-20261118-2 \
    artifacts/runs/nus-fresh-history-control-20261118-2 \
    artifacts/runs/nus-fresh-history-candidate-20261118-2 \
  --policies parent residual_no_history residual_history \
  --variants complete_gold --extra-reference residual_no_history \
  --out "${HSAI_FRESH_HISTORY_ANALYSIS_OUT:?}/complete.json"
