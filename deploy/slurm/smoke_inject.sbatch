#!/usr/bin/env bash
#SBATCH --job-name=hsai-smoke-inject
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --propagate=NONE
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 门禁：真人库的帧能不能真的注入到引擎座位、逐字段是否一致、成本多少。
# 换 --entry-boards / --human-boards 的库之后**必须**先跑它，再谈训练。
#
#   HSAI_SI_BANK   注入库（默认 data/real_library/firestone-bank-1000/train-m8k.jsonl）
#   HSAI_SI_TURNS  回合列表（默认 8,9,10,11,12,13）
#   HSAI_SI_OUT    报告路径
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

bank="${HSAI_SI_BANK:-data/real_library/firestone-bank-1000/train-m8k.jsonl}"
turns="${HSAI_SI_TURNS:-8,9,10,11,12,13}"
per="${HSAI_SI_PER:-6}"
out="${HSAI_SI_OUT:-artifacts/runs/smoke-inject-$(date +%Y%m%d-%H%M%S).json}"
if [[ ! -f "$bank" ]]; then echo "missing bank: $bank" >&2; exit 2; fi

PYTHONPATH=engine/python:. python tools/smoke_human_inject.py \
  --library "$bank" --turns "$turns" --per-turn "$per" --sample p50 --fidelity full \
  --out "$out" 2>&1 | tee "${out%.json}.log"
