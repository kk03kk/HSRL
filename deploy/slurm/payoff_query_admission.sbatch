#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-query-admission
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/learn_payoff_refinement.py \
  --plan config/research/payoff-query-refinement-20261118.json \
  --queries "${HSAI_QUERY_ROOT:?}" --arm aggregate \
  --out "${HSAI_QUERY_ADMISSION_OUT:?}" --device cpu --validate-only
