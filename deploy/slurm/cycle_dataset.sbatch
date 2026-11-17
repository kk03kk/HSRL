#!/usr/bin/env bash
#SBATCH --job-name=hsai-cycle-data
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/build_cycle_dataset.py \
  --bank artifacts/runs/nus-recruit-bank-20261118-2 \
  --checkpoint artifacts/models/arena-v1.4-r1/model.pt \
  --out "${HSAI_CYCLE_DATASET:?}" --max-sources 192 --workers 4 --anchor-games 32
