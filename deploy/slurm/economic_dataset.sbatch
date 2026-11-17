#!/usr/bin/env bash
#SBATCH --job-name=hsai-economic-data
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/build_economic_dataset.py \
  --bank artifacts/runs/nus-recruit-bank-20261118-1 \
  --out "${HSAI_DATASET_OUT:?}" --max-states 96 --workers 4
