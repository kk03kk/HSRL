#!/usr/bin/env bash
#SBATCH --job-name=hsai-research-suffix
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
exec .venv/bin/python workflows/probe_economy_suffix.py \
  --plan "${HSAI_SUFFIX_PLAN:?}" --out "${HSAI_SUFFIX_OUT:?}" --device cpu --workers 8
