#!/usr/bin/env bash
#SBATCH --job-name=hsai-league-36h
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
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1
out="${HSAI_LEAGUE_OUT:-artifacts/runs/league-v1.1-36h-20261109}"
lib="${HSRL_LEAGUE_BANK:-artifacts/libraries/league-eot-v1.1-20261109}"
source_state="artifacts/models/v1.1"
if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
if [[ ! -f "$lib/train/manifest.json" ]]; then echo "missing league train manifest" >&2; exit 2; fi
mkdir -p "$out"
export HSRL_POOL_EVIDENCE_DIR="$(realpath -m "$out")/pool-evidence"
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
done
