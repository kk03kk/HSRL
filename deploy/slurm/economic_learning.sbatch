#!/usr/bin/env bash
#SBATCH --job-name=hsai-economic-learn
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/learn_economic_process.py \
  --checkpoint "${HSAI_PROCESS_PARENT:?}" \
  --dataset artifacts/runs/nus-economic-dataset-20261118-1 \
  --out "${HSAI_PROCESS_OUT:?}" --updates "${HSAI_PROCESS_UPDATES:-0}" \
  --lr 3e-6 --anchor-kl 3 --batch 64 --device cuda:0
