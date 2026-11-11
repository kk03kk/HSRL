#!/usr/bin/env bash
#SBATCH --job-name=hsai-e1-four
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# E1 four-model round-robin: V1.1, 12311, 13806, final.
# ONE round only: 6 pairs x 448 games = 2688 games, ordered56 seats,
# shared seed block bootstrap. After valid/eligible, stop.
# Do NOT rerun for confirmation and do NOT run a champion challenge (user rule).
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSRL_E1_OUT:-artifacts/runs/e1-four-20261111}"
if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
mkdir -p "$out"
nvidia-smi > "$out/nvidia-smi.txt"
python -m pip freeze > "$out/requirements.txt"
python -c "from hsbg.az.train.personal import training_source_digest; print(training_source_digest())" > "$out/revision.txt"
cp config/eval-e1-four.json "$out/config.json"
python -u workflows/eval_round_robin_elo.py \
  --config config/eval-e1-four.json \
  --out "$out/report" --device cuda
