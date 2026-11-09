#!/usr/bin/env bash
#SBATCH --job-name=hsai-lb-final-a
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17,xgpi20
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

MODELS="m4159,m1080772-v5506,m1080772-v6039,m1080772-v7357"
OUT="artifacts/models/leaderboard-v1/eval/${SLURM_JOB_ID}-final-a"
mkdir -p "$OUT"
python -u workflows/eval_leaderboard.py --models "$MODELS" \
  --library artifacts/libraries/bot-firestone-3702-economy-v2-live-v1/dev \
  --games 64 --seed 21000000 --out "$OUT/live-v1"
