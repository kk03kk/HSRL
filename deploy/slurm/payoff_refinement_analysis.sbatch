#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-refinement-analysis
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
exec .venv/bin/python tools/analyze_economy_suffix.py \
  --runs artifacts/runs/nus-economy-suffix-20261118-1 \
    artifacts/runs/nus-economic-payoff-20261118-1 \
    artifacts/runs/nus-payoff-student-suffix-20261118-2 \
    artifacts/runs/nus-payoff-event-suffix-20261118-1 \
    "${HSAI_REPLAY_SUFFIX_ROOT:?}" "${HSAI_AGGREGATE_SUFFIX_ROOT:?}" \
  --policies parent growth payoff_student event_student query_replay query_aggregate \
  --extra-reference query_replay event_student \
  --out "${HSAI_REFINEMENT_ANALYSIS_OUT:?}/complete.json"
