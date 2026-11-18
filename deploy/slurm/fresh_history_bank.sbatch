#!/usr/bin/env bash
#SBATCH --job-name=hsai-fresh-history-bank
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/collect_recruit_bank_parallel.py \
  --out "${HSAI_FRESH_HISTORY_BANK_OUT:?}" --games 128 --shards 8 --seed 12602000 --assessment-only
