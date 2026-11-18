#!/usr/bin/env bash
#SBATCH --job-name=hsai-research-cpu-validation
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
source deploy/slurm/common.sh
hsrl_require_compute
out="${HSAI_RESEARCH_CPU_VALIDATION_OUT:?}"
test ! -e "$out"
mkdir -p "$out"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
export PYTEST_ADDOPTS="--junitxml=$out/pytest.xml -ra"
git rev-parse HEAD > "$out/source.txt"
.venv/bin/python -m pip check > "$out/pip-check.txt"
.venv/bin/python tools/verify_workspace.py --skip-rust --python-timeout 720 --json "$out/verify.json"
