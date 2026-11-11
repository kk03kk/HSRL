#!/usr/bin/env bash
#SBATCH --job-name=hsai-league-stage2
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=37:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# Stage-2 long-run template (NOT submitted by the audit).
# Requires a complete stage fork made by workflows/fork_stage.py; that fork
# resets the LR horizon and anchor because the trainer state otherwise wins.
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_LEAGUE_OUT:-artifacts/runs/league-stage2-$(date +%Y%m%d)}"
lib="${HSRL_LEAGUE_BANK:-artifacts/libraries/league-eot-v1.1-20261109}"
source_state="${HSAI_LEAGUE_SOURCE:-artifacts/models/stage2-proposal-20261111}"
validate_every="${HSRL_VALIDATE_EVERY:-4}"
validate_games="${HSRL_VALIDATE_GAMES:-32}"
validate_test_at="${HSRL_VALIDATE_TEST_AT:-36 72}"

if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
if [[ ! -f "$source_state/model.pt" || ! -f "$source_state/trainer.pt" ]]; then
  echo "missing stage fork: $source_state" >&2; exit 2
fi
mkdir -p "$out/eval"
export HSRL_POOL_EVIDENCE_DIR="$(realpath -m "$out")/pool-evidence"

validate_split() {
  local split="$1" seed="$2" tag="$3"
  local out_json="$out/eval/${tag}-${split}.json"
  echo "validate ${split} ${tag} start $(date -Is)"
  if ! python -u workflows/validate_league_checkpoint.py \
      --checkpoint "$out/train/model.pt" \
      --library "$lib" --split "$split" \
      --games "$validate_games" --seed "$seed" \
      --device cuda --out "$out_json" ; then
    echo "validate ${split} ${tag} FAILED (training continues)" >&2
  else
    echo "validate ${split} ${tag} done $(date -Is)"
  fi
}

for i in $(seq 1 72); do
  echo "segment $i/72 start $(date -Is)"
  if [[ "$i" == "1" ]]; then
    python -m hsbg.az.train.personal train \
      --config config/league-stage2.json \
      --checkpoint "$source_state/model.pt" \
      --initial-trainer-state "$source_state/trainer.pt" \
      --library "$lib/train" --out "$out/train" --device cuda \
      --run-minutes 30
  else
    python -m hsbg.az.train.personal train \
      --config config/league-stage2.json --resume \
      --library "$lib/train" --out "$out/train" --device cuda \
      --run-minutes 30
  fi
  echo "segment $i/72 done $(date -Is)"
  if (( i % validate_every == 0 )); then
    validate_split dev 930000 "segment-$(printf '%02d' "$i")"
    if [[ " $validate_test_at " == *" $i "* ]]; then
      validate_split test 940000 "segment-$(printf '%02d' "$i")"
    fi
  fi
done
