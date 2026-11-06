#!/usr/bin/env bash
#SBATCH --job-name=hsrl-bank-3702
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=02:50:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
if [[ -f "$HOME/.cargo/env" ]]; then source "$HOME/.cargo/env"; fi
export CARGO_BUILD_JOBS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
.venv/bin/python -m pip install -e ./engine
.venv/bin/python tools/verify_workspace.py --json "artifacts/local/verify-3702-$SLURM_JOB_ID.json"
.venv/bin/python workflows/build_3702_bot_bank.py --out "${HSRL_BOT_3702:?需要新版库输出目录}" --workers 8
