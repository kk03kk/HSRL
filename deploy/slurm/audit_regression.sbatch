#!/usr/bin/env bash
#SBATCH --job-name=hsai-audit-reg
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --propagate=NONE
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
if [[ -f "$HOME/.cargo/env" ]]; then source "$HOME/.cargo/env"; fi
source .venv/bin/activate
export CARGO_BUILD_JOBS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
out="${HSRL_AUDIT_REG_OUT:-artifacts/runs/audit-regression-20261111-01}"
python -u tools/verify_workspace.py \
  --rust-timeout 2400 --python-timeout 1200 \
  --json "$out/verify.json" 2>&1 | tee "$out/verify.log"
