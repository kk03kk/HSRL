#!/usr/bin/env bash
#SBATCH --job-name=hsai-gap-snap
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
# Panel eval whose opponents come from a SNAPSHOT library (the exact injection the
# 36-hour champion was trained with: whole-seat snapshot -> ended_turn=true -> the seat
# fights with exactly that board, and the engine resets ended_turn at each turn start).
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

games="${HSAI_GAP_GAMES:-512}"
out="${HSAI_GAP_OUT:-artifacts/runs/human-gap-snap-$(date +%Y%m%d-%H%M%S)}"
baseline="${HSAI_GAP_BASELINE:-artifacts/runs/human-baseline-20261112/baseline.json}"
snaps="${HSAI_GAP_MIRROR_SNAPS:?需要 HSAI_GAP_MIRROR_SNAPS}"
[[ -d "$snaps" ]] || { echo "no snapshot dir: $snaps" >&2; exit 2; }
if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi

specs=()
IFS=';' read -ra pairs <<< "${HSAI_GAP_MODELS:?需要 HSAI_GAP_MODELS}"
for pair in "${pairs[@]}"; do specs+=(--checkpoint "$pair"); done

mkdir -p "$out"
{
  echo "start=$(date -Is)"
  echo "games=$games"
  echo "mirror_snaps=$snaps"
  echo "mirror_sample=${HSAI_GAP_MIRROR_SAMPLE:-p50band}"
  echo "mirror_strict=${HSAI_GAP_MIRROR_STRICT:-0}"
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
} | tee "$out/run.env"

PYTHONPATH=engine/python:. python workflows/eval_human_gap.py \
  "${specs[@]}" \
  --games "$games" \
  --device cuda:0 \
  --forbid-unfreeze 1 \
  --human-baseline "$baseline" \
  --mirror-snaps "$snaps" \
  --mirror-sample "${HSAI_GAP_MIRROR_SAMPLE:-p50band}" \
  --mirror-strict "${HSAI_GAP_MIRROR_STRICT:-0}" \
  --out "$out/panel.json" 2>&1 | tee "$out/panel.log"

echo "done=$(date -Is)" | tee -a "$out/run.env"
