#!/usr/bin/env bash
#SBATCH --job-name=hsai-league-src
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
[[ -n "${SRC_NAME:-}" && -n "${SRC_MODEL:-}" && -n "${SRC_SEED:-}" && -n "${SRC_OUT:-}" && -n "${SRC_LIB:-}" ]]
python workflows/build_v1_1_panel.py \
  --model "$SRC_MODEL" --games 768 --seed "$SRC_SEED" \
  --out "$SRC_OUT" --library "$SRC_LIB" \
  --device cuda --temperature 0 --max-actions-per-turn 80 \
  --move-cap 5 --move-repeat-limit 2 --forbid-cancel 1 --cancel-cap 2
