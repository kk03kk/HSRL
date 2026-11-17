#!/usr/bin/env bash
#SBATCH --job-name=hsai-economy-suffix
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
exec .venv/bin/python workflows/probe_economy_suffix.py \
  --plan config/research/economy-suffix-20261118.json \
  --out "${HSAI_ECONOMY_SUFFIX_OUT:?}" --device cuda:0 --workers 4
