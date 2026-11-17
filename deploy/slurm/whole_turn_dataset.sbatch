#!/usr/bin/env bash
#SBATCH --job-name=hsai-turn-data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/build_whole_turn_dataset.py \
  --bank artifacts/runs/nus-recruit-bank-20261118-2 \
  --anchors artifacts/runs/nus-cycle-dataset-20261118-1 \
  --baseline artifacts/runs/nus-whole-turn-economy-20261118-1 \
  --out "${HSAI_WHOLE_DATA_OUT:?}" --workers 4
