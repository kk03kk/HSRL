#!/usr/bin/env bash
#SBATCH --job-name=hsai-economic-payoff
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
args=()
if [[ "${HSAI_PAYOFF_PROBE:-0}" == "1" ]]; then args+=(--probe); fi
exec .venv/bin/python workflows/probe_economy_suffix.py \
  --plan config/research/economic-payoff-20261118.json \
  --out "${HSAI_PAYOFF_OUT:?}" --device cpu --workers "${HSAI_PAYOFF_WORKERS:-8}" "${args[@]}"
