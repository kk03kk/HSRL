#!/usr/bin/env bash
#SBATCH --job-name=hsai-process-audit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/audit_economic_decisions.py \
  artifacts/runs/nus-input-development-20261118-1/panel.json \
  artifacts/runs/nus-process-development-20261118-1/panel.json \
  --out artifacts/state/research-20261118/natural-opportunities-v1.json --workers 4
