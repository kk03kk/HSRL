#!/usr/bin/env bash
#SBATCH --job-name=hsai-cpu-check
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source deploy/slurm/common.sh
hsrl_require_compute
out="${HSAI_VALIDATION_OUT:?}"
test ! -e "$out"
mkdir -p "$out"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 CARGO_BUILD_JOBS=8
export PYTEST_ADDOPTS="--junitxml=$out/pytest.xml -ra"
.venv/bin/python -m pip check > "$out/pip-check.txt"
.venv/bin/python -m pip freeze --all > "$out/pip-freeze.txt"
.venv/bin/python tools/verify_workspace.py --json "$out/workspace.json"
.venv/bin/python tools/validate_eval_evidence.py \
  artifacts/runs/nus-modernize-validation-20261117-2/evaluation-smoke/panel.json \
  --out "$out/smoke-evidence.json" --replay 10000
