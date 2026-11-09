#!/usr/bin/env bash
#SBATCH --job-name=hsai-elo-8-smoke
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17,xgpi20
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=00:45:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
OUT="${HSAI_ELO_OUT:-artifacts/runs/elo-8-smoke-${SLURM_JOB_ID}}"
python -u workflows/eval_round_robin_elo.py --config config/eval-elo-8.json --out "$OUT" \
  --pair-limit 1 --games-per-pair 2 --device cuda
