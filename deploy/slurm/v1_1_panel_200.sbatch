#!/usr/bin/env bash
#SBATCH --job-name=hsai-v1.1-panel-200
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
python -m py_compile workflows/build_v1_1_panel.py
python - <<'PY'
import hsbg_engine
print('engine', hsbg_engine.__file__, hsbg_engine.game_version())
PY
python workflows/build_v1_1_panel.py \
  --model artifacts/models/v1.1/model.pt \
  --games 12 --seed 241000000 \
  --out artifacts/runs/v1.1-panel-smoke-pending-guarded-20261109 \
  --library artifacts/libraries/v1.1-eot-smoke-pending-guarded-20261109 \
  --device cuda --temperature 0 --max-actions-per-turn 80 --move-cap 5 --move-repeat-limit 2 --cancel-cap 2
python workflows/build_v1_1_panel.py \
  --model artifacts/models/v1.1/model.pt \
  --games 200 --seed 240000000 \
  --out artifacts/runs/v1.1-panel-200-pending-guarded-20261109 \
  --library artifacts/libraries/v1.1-eot-200-pending-guarded-20261109 \
  --device cuda --temperature 0 --max-actions-per-turn 80 --move-cap 5 --move-repeat-limit 2 --cancel-cap 2
