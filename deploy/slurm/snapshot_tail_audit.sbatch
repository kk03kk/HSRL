#!/usr/bin/env bash
#SBATCH --job-name=hsai-snapshot-tail-audit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/audit_snapshot_tail.py --workers 8 \
  --panels artifacts/runs/nus-eval-v14-r1-20261117-1/panel.json \
    artifacts/runs/nus-input-development-20261118-1/panel.json \
    artifacts/runs/nus-process-development-20261118-1/panel.json \
    artifacts/runs/nus-research-development-20261118-1/panel.json \
    artifacts/runs/nus-cycle-development-20261118-1/panel.json \
    artifacts/runs/nus-conditional-development-20261118-1/panel.json \
  --out "${HSAI_TAIL_AUDIT_OUT:?}"
