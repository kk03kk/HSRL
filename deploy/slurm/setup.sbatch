#!/usr/bin/env bash
#SBATCH --job-name=hsrl-setup
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
export CARGO_BUILD_JOBS=8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1
hostname
ulimit -a
if [[ ! -x .venv/bin/python || "${HSRL_SETUP_INSTALL:-0}" == 1 ]]; then
  bash deploy/bootstrap_nus.sh
elif [[ "${HSRL_SETUP_REBUILD_ENGINE:-0}" == 1 ]]; then
  .venv/bin/python -m pip install -e ./engine
fi
source .venv/bin/activate
python -u tools/verify_workspace.py --rust-timeout "${HSRL_RUST_TIMEOUT:-3600}" \
  --python-timeout "${HSRL_PYTHON_TIMEOUT:-1800}" --json "artifacts/local/verify-nus-$SLURM_JOB_ID.json"
python workflows/migrate_checkpoint_3702.py \
  --model artifacts/models/best/s14_best_main_model.pt --out "artifacts/local/setup-3702-$SLURM_JOB_ID"
python -m hsbg.az.train.personal preflight \
  --checkpoint "artifacts/local/setup-3702-$SLURM_JOB_ID/model.pt" --device cpu \
  --out artifacts/local/preflight-nus-cpu.json
