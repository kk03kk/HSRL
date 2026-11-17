#!/usr/bin/env bash
#SBATCH --job-name=hsai-conditional-probe
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
args=()
if [[ "${HSAI_CONDITIONAL_MECHANISMS:-0}" == "1" ]]; then args+=(--mechanisms); fi
exec .venv/bin/python workflows/probe_conditional_economy.py \
  --checkpoint artifacts/models/arena-v1.4-r1/model.pt \
  --dataset artifacts/runs/nus-whole-turn-dataset-20261118-3 \
  --out "${HSAI_CONDITIONAL_PROBE_OUT:?}" --ppo-probe "${args[@]}"
