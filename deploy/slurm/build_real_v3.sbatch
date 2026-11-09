#!/usr/bin/env bash
#SBATCH --job-name=hsrl-bank-real-v3
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=04:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
out="${HSRL_BANK_OUT:-artifacts/libraries/bot-firestone-3702-real-v3-20261109}"
smoke="${out}-smoke-${SLURM_JOB_ID}"
python -m py_compile workflows/build_3702_bot_bank.py
python - <<'PY'
import hsbg_engine
print('engine', hsbg_engine.__file__)
PY
python workflows/build_3702_bot_bank.py --out "$smoke" --workers 16 --smoke \
  --profile legacy-v1 --consistent-profile --seed-base 12000000 \
  --train-games 16 --dev-games 16 --test-games 16 --rounds 1
python workflows/build_3702_bot_bank.py --out "$out" --workers 16 \
  --profile legacy-v1 --consistent-profile --seed-base 12000000 \
  --train-games 1024 --dev-games 256 --test-games 256 --rounds 1
