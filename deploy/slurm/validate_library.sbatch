#!/usr/bin/env bash
#SBATCH --job-name=hsai-libvalidate
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --propagate=NONE
#SBATCH --time=01:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 逐帧过 Rust 引擎验证一个快照库，产出带 validation 的新库。
#
# 为什么必须有这一步：``require_calibrated_library`` 要求
# ``validation.accepted > 0`` 且 ``quarantined == 0``，而
# ``build_human_curve_bank`` / ``scale_library`` 产出的 manifest **没有** validation 字段。
# 缺了它，库会被 ``--mirror-strict 1`` 直接拒绝——这正是这条管线一直没跑通的原因之一。
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

src="${HSAI_LV_SOURCE:?set HSAI_LV_SOURCE to the library dir}"
out="${HSAI_LV_OUT:?set HSAI_LV_OUT to a new directory}"

if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi

echo "source=$src"
echo "out=$out"
echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"

PYTHONPATH=engine/python:. python workflows/prepare_validated_library.py \
  --source "$src" --out "$out" 2>&1 | tail -40

echo "--- resulting manifest ---"
head -c 600 "$out/manifest.json" || true
echo
echo "done=$(date -Is)"
