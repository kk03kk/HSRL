#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-memory-analysis
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
    artifacts/runs/nus-economic-payoff-20261118-1 "${HSAI_PAYOFF_MEMORY_OUT:?}" \
  --policies parent growth growth_no_memory --variants complete_gold --extra-reference growth \
  --out "${HSAI_PAYOFF_MEMORY_ANALYSIS_OUT:?}/complete.json"
