#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-shadow-audit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/audit_payoff_closed_loop.py \
  --runs artifacts/runs/nus-economic-payoff-20261118-1 artifacts/runs/nus-payoff-student-suffix-20261118-2 \
  --names growth payoff_student --out "${HSAI_SHADOW_AUDIT_OUT:?}" --workers 8
