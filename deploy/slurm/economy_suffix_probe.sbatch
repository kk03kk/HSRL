#!/usr/bin/env bash
#SBATCH --job-name=hsai-economy-suffix-probe
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
exec .venv/bin/python workflows/probe_economy_suffix.py \
  --plan config/research/economy-suffix-20261118.json \
  --out "${HSAI_ECONOMY_SUFFIX_PROBE_OUT:?}" --probe --device cpu --workers 2
