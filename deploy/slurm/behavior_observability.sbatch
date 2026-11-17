#!/usr/bin/env bash
#SBATCH --job-name=hsai-behavior-observe
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/audit_behavior_observability.py \
  --dataset "${HSAI_WHOLE_DATASET:?}" --out "${HSAI_OBSERVABILITY_OUT:?}"
