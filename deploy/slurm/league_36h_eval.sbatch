#!/usr/bin/env bash
#SBATCH --job-name=hsai-league-36h-eval
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=38:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# Long league run WITH held-out validation.
#
# Validation policy:
#   - dev every HSRL_VALIDATE_EVERY segments (default 4), 32 games, fixed seed;
#   - test only at HSRL_VALIDATE_TEST_AT (default "36 72"), 32 games, fixed seed;
#   - validation runs after the segment process exits, on the same allocation,
#     so it never competes with the learner/actors;
#   - a validation failure is reported but does not abort training.
#
# The old 36h run (1102669) used league_36h.sbatch and had no periodic
# validation; this template is the replacement for the next long run.
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_LEAGUE_OUT:-artifacts/runs/league-v1.1-36h-eval-$(date +%Y%m%d)}"
lib="${HSRL_LEAGUE_BANK:-artifacts/libraries/league-eot-v1.1-20261109}"
source_state="${HSAI_LEAGUE_SOURCE:-artifacts/models/league-final-20261111-step014858}"
validate_every="${HSRL_VALIDATE_EVERY:-4}"
validate_games="${HSRL_VALIDATE_GAMES:-32}"
validate_dev_seed="${HSRL_VALIDATE_DEV_SEED:-930000}"
validate_test_seed="${HSRL_VALIDATE_TEST_SEED:-940000}"
validate_test_at="${HSRL_VALIDATE_TEST_AT:-36 72}"

if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
if [[ ! -f "$lib/train/manifest.json" ]]; then echo "missing league train manifest" >&2; exit 2; fi
if [[ ! -f "$source_state/model.pt" || ! -f "$source_state/trainer.pt" ]]; then
  echo "missing source state: $source_state" >&2; exit 2
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
      --config config/league-36h.json \
      --checkpoint "$source_state/model.pt" \
      --initial-trainer-state "$source_state/trainer.pt" \
      --library "$lib/train" --out "$out/train" --device cuda \
      --run-minutes 30
  else
    python -m hsbg.az.train.personal train \
      --config config/league-36h.json --resume \
      --library "$lib/train" --out "$out/train" --device cuda \
      --run-minutes 30
  fi
  echo "segment $i/72 done $(date -Is)"

  if (( i % validate_every == 0 )); then
    validate_split dev "$validate_dev_seed" "segment-$(printf '%02d' "$i")"
    if [[ " $validate_test_at " == *" $i "* ]]; then
      validate_split test "$validate_test_seed" "segment-$(printf '%02d' "$i")"
    fi
  fi
done
