#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-retention
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/audit_payoff_retention.py \
  --run artifacts/runs/nus-payoff-deployment-20261118-1 \
  --out "${HSAI_PAYOFF_RETENTION_OUT:?}" --workers 4
