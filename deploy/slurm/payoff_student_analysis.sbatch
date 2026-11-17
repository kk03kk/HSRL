#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-student-analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
exec .venv/bin/python tools/analyze_economy_suffix.py \
  --runs artifacts/runs/nus-economy-suffix-20261118-1 \
    artifacts/runs/nus-economic-payoff-20261118-1 \
    artifacts/runs/nus-payoff-student-suffix-20261118-1 \
  --policies parent growth payoff_student \
  --out "${HSAI_STUDENT_ANALYSIS_OUT:?}/complete.json"
