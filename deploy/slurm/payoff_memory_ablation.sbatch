#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-memory-ablation
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/probe_economy_suffix.py \
  --plan config/research/payoff-memory-ablation-20261118.json \
  --out "${HSAI_PAYOFF_MEMORY_OUT:?}" --device cpu --workers 8
