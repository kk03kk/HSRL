#!/usr/bin/env bash
#SBATCH --job-name=hsai-real-9691-long
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=08:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1
out="${HSAI_LONG_OUT:-artifacts/runs/real-bot-9691-long-20261109}"
lib="${HSRL_REAL_BANK:-artifacts/libraries/bot-firestone-3702-real-v3-20261109}"
source_state="artifacts/runs/p50-forced-tier-long-20261108-135823/last-committed/step_009691"
if [[ -e "$out" ]]; then echo "输出目录已存在: $out" >&2; exit 2; fi
mkdir -p "$out"
export HSRL_POOL_EVIDENCE_DIR="$(realpath -m "$out")/pool-evidence"
python -S tools/nus_submit.py --phase train --check-only || true
python workflows/audit_full_trainer.py --source "$source_state" --version 9691 \
  --out "$out/source-audit.json" --device cpu
python -m hsbg.az.train.personal train \
  --config config/real-9691-long-8h-1gpu.json \
  --checkpoint "$source_state/model.pt" \
  --initial-trainer-state "$source_state/trainer.pt" \
  --library "$lib/train" \
  --out "$out/train" \
  --device cuda \
  --run-minutes 420 \
  --stop-at-decisions 1050000000
mkdir -p "$out/final-eval"
python -m hsbg.az.train.personal eval \
  --checkpoint "$out/train/model.pt" --library "$lib/dev" --device cuda \
  --games 16 --seed 15000000 --out "$out/final-eval/new.json" \
  --mirror-strict --exposure 1 --economy \
  --max-actions-per-turn 80 --move-guard 1 --move-cap 5 --move-repeat-limit 2
python -m hsbg.az.train.personal eval \
  --checkpoint "$source_state/model.pt" --library "$lib/dev" --device cuda \
  --games 16 --seed 15000000 --out "$out/final-eval/9691.json" \
  --mirror-strict --exposure 1 --economy \
  --max-actions-per-turn 80 --move-guard 1 --move-cap 5 --move-repeat-limit 2
python -m hsbg.az.train.personal board "$out/final-eval/new.json" "$out/final-eval/9691.json" \
  --out "$out/final-eval/leaderboard.json"
