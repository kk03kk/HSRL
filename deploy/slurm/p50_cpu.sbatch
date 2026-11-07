#!/usr/bin/env bash
#SBATCH --job-name=hsai-p50-cpu
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
export CARGO_BUILD_JOBS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAYON_NUM_THREADS=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
if [[ -f "$HOME/.cargo/env" ]]; then source "$HOME/.cargo/env"; fi
if [[ ! -x .venv/bin/python ]]; then bash deploy/bootstrap_nus.sh; fi
source .venv/bin/activate
python -m pip install -e . --no-deps
python -m pip install -e ./engine
python tools/verify_workspace.py --rust-timeout 7200 --python-timeout 1800 --json artifacts/local/p50-cpu-validation.json
python workflows/audit_full_trainer.py --source "${HSAI_SOURCE_7357:-../HSRL/personal/runs/nus-4159-economy-long-002/milestones/decisions-558819917}" --version 7357 --out artifacts/local/7357-cpu-restore.json
bank="${HSAI_P50_BANK:-artifacts/libraries/bot-p50-3702-v1}"
# 本轮校准曲线必须显式给出：默认指向"稍弱于模型"的分级曲线，参数缺失直接失败。
curve="${HSAI_P50_CURVE:?必须显式指定本轮对手曲线（例如 config/opponent-p50-slightly-weaker.json）}"
python workflows/build_p50_bank.py --source "${HSAI_OLD_BANK:-../HSRL/personal/artifacts/libraries/bot-firestone-3702-economy-v2-live-v1}" --out "$bank" --curve "$curve" --workers 8
