#!/usr/bin/env bash
#SBATCH --job-name=hsai-core-probe
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/probe_core_acquisition.py \
  --bank artifacts/runs/nus-recruit-bank-20261118-2 --out "${HSAI_ACQUISITION_OUT:?}" \
  --checkpoint parent=artifacts/models/arena-v1.4-r1/model.pt \
  --max-per-core 1 --replicas 1 --workers 1 --device cpu
