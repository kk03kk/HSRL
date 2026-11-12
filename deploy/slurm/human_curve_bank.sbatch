#!/usr/bin/env bash
#SBATCH --job-name=hsai-humancurve
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --propagate=NONE
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 真人曲线对手库：把模型自对弈快照的攻击/生命按比例重分配，使每回合总攻血落在
# 真人 Train 分布的目标分位上。用途见 docs/analysis/HUMAN_GAP_20261112.md
# ——模型用真人 1/4.5 的战力就能稳赢现有对手，后期再强也没有回报。
#
# Overridable environment:
#   HSAI_HC_SPLIT   train|dev|test           (default train)
#   HSAI_HC_QUOTA   derived frames per bucket (default 32)
#   HSAI_HC_OUT     output library dir        (default artifacts/libraries/humancurve-<split>-<ts>)
#   HSAI_HC_SOURCES JSON map                  (default {"league": "…league-eot-v1.1-20261109/train"})
#   HSAI_HC_HUMAN   human library.jsonl       (default data/real_library/firestone-20261112-1000/library.jsonl)
#   HSAI_HC_AUDIT   1 = also run the held-out pressure audit
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

split="${HSAI_HC_SPLIT:-train}"
quota="${HSAI_HC_QUOTA:-32}"
ts="$(date +%Y%m%d-%H%M%S)"
out="${HSAI_HC_OUT:-artifacts/libraries/humancurve-${split}-${ts}}"
human="${HSAI_HC_HUMAN:-data/real_library/firestone-20261112-1000/library.jsonl}"
if [[ -n "${HSAI_HC_SOURCES:-}" ]]; then
  sources="$HSAI_HC_SOURCES"
else
  sources="{\"league\":\"artifacts/libraries/league-eot-v1.1-20261109/${split}\"}"
  # 额外的板块来源，格式 name=path。用 '=' 而不是 JSON，因为 sbatch --export 以逗号分隔。
  if [[ -n "${HSAI_HC_EXTRA_SOURCE:-}" ]]; then
    extra_name="${HSAI_HC_EXTRA_SOURCE%%=*}"
    extra_path="${HSAI_HC_EXTRA_SOURCE#*=}"
    sources="${sources%\}},\"${extra_name}\":\"${extra_path}\"}"
  fi
fi
work="artifacts/runs/humancurve-work-${ts}"

if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
mkdir -p "$work"

echo "split=$split quota=$quota"
echo "out=$out"
echo "sources=$sources"
echo "human=$human"
echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"

audit_args=()
if [[ "${HSAI_HC_AUDIT:-0}" == "1" ]]; then
  audit_args=(--audit-out "$work/pressure-audit.json")
fi

PYTHONPATH=engine/python:. python workflows/build_human_curve_bank.py \
  --sources "$sources" \
  --human "$human" \
  --out "$out" \
  --split "$split" \
  --quota "$quota" \
  --work-dir "$work" \
  ${audit_args[@]+"${audit_args[@]}"} 2>&1 | tee "$work/build.log"

echo "done=$(date -Is)"
