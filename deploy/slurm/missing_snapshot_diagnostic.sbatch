#!/usr/bin/env bash
#SBATCH --job-name=hsai-snapshot-diagnostic
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/diagnose_missing_snapshot.py \
 --source artifacts/runs/nus-economic-payoff-20261118-1/trials/target_safe/case-019-complete_gold-r01 \
 --out "${HSAI_SNAPSHOT_DIAGNOSTIC_OUT:?}"
