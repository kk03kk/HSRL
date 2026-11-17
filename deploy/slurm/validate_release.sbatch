#!/usr/bin/env bash
#SBATCH --job-name=hsai-release-check
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi11,xgpi15,xgpi17
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
test -n "${SLURM_JOB_ID:-}"
cd "${SLURM_SUBMIT_DIR:?}"
out="${HSAI_VALIDATION_OUT:?}"
test ! -e "$out"
mkdir -p "$out"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
export PYTEST_ADDOPTS="--junitxml=$out/pytest.xml -ra"
.venv/bin/python tools/verify_workspace.py --skip-rust --json "$out/workspace.json"
.venv/bin/python -m pip check > "$out/pip-check.txt"
if [[ ! -d artifacts/models/arena-v1.4-r1 ]]; then
  .venv/bin/python workflows/register_completed_model.py \
    --run artifacts/runs/nus-arena-v1.4-r1-20261117/train \
    --destination artifacts/models/arena-v1.4-r1 --model-id arena-v1.4-r1 \
    --recovery-source artifacts/releases/recovery-v14-r1-20261117 > "$out/model-registration.json"
fi
.venv/bin/python tools/validate_model_restore.py \
  --source artifacts/models/arena-v1.4-r1 --out "$out/restore-probe" --device cuda:0
.venv/bin/python workflows/evaluate_suite.py \
  --plan config/evaluation/v14-smoke-20261117.json --out "$out/evaluation-smoke" --device cuda:0
echo COMPLETE
