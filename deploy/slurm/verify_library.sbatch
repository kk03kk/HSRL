#!/usr/bin/env bash
#SBATCH --job-name=hsai-bankverify
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=00:30:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 校验真人曲线对手库的逐回合战力。对手库的 snap 文件展开后有数 GB，
# json.load 需要几十 GB 内存，所以必须上计算节点，不能放登录节点。
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

bank="${HSAI_BV_BANK:?set HSAI_BV_BANK to the library dir}"
baseline="${HSAI_BV_BASELINE:-artifacts/runs/human-baseline-20261112/baseline.json}"
source="${HSAI_BV_SOURCE:-artifacts/libraries/league-eot-v1.1-20261109/train}"
out="${HSAI_BV_OUT:-artifacts/runs/human-baseline-20261112/bank-verify.json}"

echo "bank=$bank"
echo "source=$source"
echo "baseline=$baseline"
echo "out=$out"
echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"

PYTHONPATH=engine/python:. python tools/verify_human_curve_bank.py \
  --bank "$bank" --source "$source" --baseline "$baseline" --out "$out" \
  ${HSAI_BV_EXTRA:-}
echo "done=$(date -Is)"
