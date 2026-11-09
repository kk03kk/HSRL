#!/usr/bin/env bash
#SBATCH --job-name=hsai-ab-1109
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi11,xgpi15,xgpi17,xgpi20
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
SRC="${AB_SOURCE:?set AB_SOURCE to a full source directory containing model.pt/trainer.pt/checkpoint.json}"
OUT="${AB_OUT:?set AB_OUT}"
REAL=artifacts/libraries/bot-firestone-3702-economy-v2-live-v1/train
P50=artifacts/libraries/bot-p50-3702-v2-weaker/train-start
DEV=artifacts/libraries/bot-firestone-3702-economy-v2-live-v1/dev
for arm in base entropy shaping realbot; do
  lib="$P50"
  [ "$arm" = realbot ] && lib="$REAL"
  python -u workflows/short_ab.py --source-dir "$SRC" --source-name "$(basename "$SRC")" \
    --out "$OUT/$arm" --arm "$arm" --library "$lib" --eval-library "$DEV" \
    --minutes 30 --new-decisions 15000000 --games 512 --eval-games 64 \
    --seed 200000000 --eval-seed 21000000 --device cuda:0
done
