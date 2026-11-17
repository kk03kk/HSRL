#!/usr/bin/env bash
#SBATCH --job-name=hsai-core-acquisition
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/probe_core_acquisition.py \
  --bank artifacts/runs/nus-recruit-bank-20261118-2 --out "${HSAI_ACQUISITION_OUT:?}" \
  --checkpoint parent=artifacts/models/arena-v1.4-r1/model.pt \
  --checkpoint cycle=artifacts/runs/nus-cycle-all-20261118-1/learn/model.pt \
  --max-per-core 12 --replicas 8 --workers 4 --device cuda:0
