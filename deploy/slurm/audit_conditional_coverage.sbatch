#!/usr/bin/env bash
#SBATCH --job-name=hsai-conditional-coverage
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/audit_conditional_coverage.py \
  artifacts/runs/nus-eval-v14-r1-20261117-1/panel.json \
  --model v1.4_r1 --workers 4 --out "${HSAI_CONDITIONAL_COVERAGE_OUT:?}"
