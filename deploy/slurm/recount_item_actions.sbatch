#!/usr/bin/env bash
#SBATCH --job-name=hsai-card-types
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -eu
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir "${HSAI_RECOUNT_OUT:?}"
exec .venv/bin/python tools/recount_item_actions.py \
    artifacts/runs/nus-input-development-20261118-1/panel.json v1.4_r1 "$HSAI_RECOUNT_OUT/counts.json"
