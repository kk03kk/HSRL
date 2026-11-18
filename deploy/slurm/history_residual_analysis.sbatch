#!/usr/bin/env bash
#SBATCH --job-name=hsai-history-residual-analysis
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
  --runs artifacts/runs/nus-economy-suffix-20261118-1 \
    artifacts/runs/nus-economic-payoff-20261118-1 \
    artifacts/runs/nus-payoff-query-replay-suffix-20261118-1 \
    "${HSAI_NO_HISTORY_SUFFIX_ROOT:?}" "${HSAI_HISTORY_SUFFIX_ROOT:?}" \
  --policies parent growth query_replay residual_no_history residual_history \
  --extra-reference query_replay residual_no_history \
  --out "${HSAI_HISTORY_ANALYSIS_OUT:?}/complete.json"
