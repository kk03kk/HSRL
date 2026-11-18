#!/usr/bin/env bash
#SBATCH --job-name=hsai-capped-turn-analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/analyze_capped_turns.py \
  --analysis artifacts/runs/nus-history-residual-analysis-20261118-1/complete.json \
  --out "${HSAI_CAPPED_TURN_OUT:?}/complete.json" --workers 8
