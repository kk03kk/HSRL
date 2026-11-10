#!/usr/bin/env bash
#SBATCH --job-name=hsai-elo3-league
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
python -m py_compile workflows/eval_three_models_elo.py
python workflows/eval_three_models_elo.py \
  --config config/eval-league-3.json --out artifacts/runs/elo3-audit-v2-20261110 --device cuda --audit-only
python workflows/eval_three_models_elo.py \
  --config config/eval-league-3.json --out artifacts/runs/elo3-smoke-v2-20261110 --device cuda --games-per-pair 16
python workflows/eval_three_models_elo.py \
  --config config/eval-league-3.json --out artifacts/runs/elo3-v1.1-league-v2-20261110 --device cuda
