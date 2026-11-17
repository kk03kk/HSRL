#!/usr/bin/env bash
#SBATCH --job-name=hsai-fullboard-probe
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/probe_full_board_process.py \
  --dataset artifacts/runs/nus-economic-dataset-20261118-1 \
  --checkpoint control=artifacts/runs/nus-input-base-20261118-1/train/model.pt \
  --checkpoint process=artifacts/runs/nus-process-base-20261118-2/model.pt \
  --out artifacts/runs/nus-fullboard-process-20261118-1 --device cpu
