#!/usr/bin/env bash
#SBATCH --job-name=hsai-payoff-data
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
args=()
if [[ "${HSAI_PAYOFF_DATA_PROBE:-0}" == "1" ]]; then args+=(--probe); fi
exec .venv/bin/python workflows/build_payoff_dataset.py \
  --bank artifacts/runs/nus-recruit-bank-20261118-2 \
  --anchors artifacts/runs/nus-cycle-dataset-20261118-1 \
  --development artifacts/runs/nus-economic-payoff-20261118-1 \
  --checkpoint artifacts/models/arena-v1.4-r1/model.pt \
  --library artifacts/libraries/arena-cal-v14-snap \
  --out "${HSAI_PAYOFF_DATA_OUT:?}" --workers "${HSAI_PAYOFF_DATA_WORKERS:-8}" "${args[@]}"
