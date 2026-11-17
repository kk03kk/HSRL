#!/usr/bin/env bash
#SBATCH --job-name=hsai-economic-starts
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/probe_economic_starts.py \
  --bank "${HSAI_START_DIAG_BANK:?}" --out "${HSAI_START_DIAG_OUT:?}" --device cpu \
  --checkpoint control=artifacts/runs/nus-input-base-20261118-1/train/model.pt \
  --checkpoint process=artifacts/runs/nus-process-base-20261118-2/model.pt
