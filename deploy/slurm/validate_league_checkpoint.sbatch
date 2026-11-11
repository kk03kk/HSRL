#!/usr/bin/env bash
#SBATCH --job-name=hsai-validate
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# One-shot held-out dev/test validation. The long-run template calls the same
# Python entry point inline after each validation segment.
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

ckpt="${HSAI_VAL_CKPT:-artifacts/models/league-final-20261111-step014858/model.pt}"
out="${HSAI_VAL_OUT:-artifacts/runs/validate-final-20261111}"
lib="${HSRL_LEAGUE_BANK:-artifacts/libraries/league-eot-v1.1-20261109}"
games="${HSAI_VAL_GAMES:-32}"
dev_seed="${HSAI_VAL_DEV_SEED:-930000}"
test_seed="${HSAI_VAL_TEST_SEED:-940000}"

if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
mkdir -p "$out"

python -u workflows/validate_league_checkpoint.py \
  --checkpoint "$ckpt" --library "$lib" --split dev \
  --games "$games" --seed "$dev_seed" --device cuda \
  --out "$out/dev.json"

python -u workflows/validate_league_checkpoint.py \
  --checkpoint "$ckpt" --library "$lib" --split test \
  --games "$games" --seed "$test_seed" --device cuda \
  --out "$out/test.json"
