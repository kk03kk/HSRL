#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-admission
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python workflows/learn_payoff_process.py \
 --checkpoint artifacts/models/arena-v1.4-r1/model.pt \
 --dataset artifacts/runs/nus-payoff-dataset-20261118-1 \
 --out "${HSAI_PAYOFF_ADMISSION_OUT:?}" --validate-only
