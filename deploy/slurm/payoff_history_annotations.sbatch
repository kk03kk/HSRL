#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-history-annotations
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:35:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
extra=()
if [[ "${HSAI_HISTORY_PROBE:-0}" == 1 ]]; then extra+=(--probe); fi
exec .venv/bin/python workflows/annotate_payoff_history.py \
  --dataset artifacts/runs/nus-payoff-dataset-20261118-1 \
  --expected-manifest-sha256 006175b0310401ff3d2f3e1d6b0aad025d8c0a9fe2c1a27faddda5ab63e769ea \
  --out "${HSAI_HISTORY_OUT:?}" --workers 8 "${extra[@]}"
