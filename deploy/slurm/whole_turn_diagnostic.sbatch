#!/usr/bin/env bash
#SBATCH --job-name=hsai-turn-diagnose
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/diagnose_whole_turn_failure.py \
  --bank artifacts/runs/nus-recruit-bank-20261118-2 \
  --partial artifacts/runs/nus-whole-turn-dataset-20261118-2 \
  --out "${HSAI_WHOLE_DIAG_OUT:?}"
