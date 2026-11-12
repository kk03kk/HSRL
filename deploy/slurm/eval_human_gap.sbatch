#!/usr/bin/env bash
#SBATCH --job-name=hsai-human-gap
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --propagate=NONE
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# Human-gap panel: put our models and the Firestone human baseline on one axis.
#
# Overridable environment:
#   HSAI_GAP_GAMES      games per checkpoint        (default 96)
#   HSAI_GAP_OUT        output dir                  (default artifacts/runs/human-gap-<ts>)
#   HSAI_GAP_TIER_PLAN  forced-tier diagnostic      (default "" = model decides)
#   HSAI_GAP_MODELS     "name=path,name=path,..."   (default: the four E1 models)
#
# The tier plan "2:2,5:3,6:4,8:5" is the human tempo curve measured from the
# replays (T2@turn2, T3@5, T4@6, T5@8). Forcing it separates "bad tempo" from
# "everything else"; NEVER use a forced plan for anything that gets ranked.
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

games="${HSAI_GAP_GAMES:-96}"
out="${HSAI_GAP_OUT:-artifacts/runs/human-gap-$(date +%Y%m%d-%H%M%S)}"
tier_plan="${HSAI_GAP_TIER_PLAN:-}"
baseline="${HSAI_GAP_BASELINE:-artifacts/runs/human-baseline-20261112/baseline.json}"

if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi

if [[ -n "${HSAI_GAP_MODELS:-}" ]]; then
  specs=()
  IFS=',' read -ra pairs <<< "$HSAI_GAP_MODELS"
  for pair in "${pairs[@]}"; do specs+=(--checkpoint "$pair"); done
else
  specs=(
    --checkpoint "v1.1=artifacts/models/v1.1/model.pt"
    --checkpoint "s12311=artifacts/models/league-latest-20261110-step012311/model.pt"
    --checkpoint "s13806=artifacts/models/league-latest-20261111-step013806/model.pt"
    --checkpoint "final14858=artifacts/models/league-final-20261111-step014858/model.pt"
  )
fi

mkdir -p "$out"
{
  echo "start=$(date -Is)"
  echo "games=$games"
  echo "forced_tier_plan=${tier_plan:-<none>}"
  echo "baseline=$baseline"
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
} | tee "$out/run.env"

PYTHONPATH=engine/python:. python workflows/eval_human_gap.py \
  "${specs[@]}" \
  --games "$games" \
  --device cuda:0 \
  --forbid-unfreeze 1 \
  --forced-tier-plan "$tier_plan" \
  --human-baseline "$baseline" \
  --out "$out/panel.json" 2>&1 | tee "$out/panel.log"

echo "done=$(date -Is)" | tee -a "$out/run.env"
