#!/usr/bin/env bash
#SBATCH --job-name=hsai-tail-replay
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/replay_snapshot_tails.py \
  --inventory artifacts/runs/nus-snapshot-tail-audit-20261118-1/complete.json \
  --out "${HSAI_TAIL_REPLAY_OUT:?}"
