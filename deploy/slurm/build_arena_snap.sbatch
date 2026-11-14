#!/usr/bin/env bash
#SBATCH --job-name=hsai-arenasnap
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --propagate=NONE
#SBATCH --time=00:40:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 建"校准战团库 + 快照库"（纯 CPU，不占 GPU）。
# 两步同一作业内完成，保证 boards.jsonl 与 snap_*.json.gz 一定同源；
# boards.jsonl 已存在时只补建快照库（HSAI_LIB_FORCE=1 可强制重建）。
#
#   HSAI_LIB_PANEL   基线面板（默认 gap-calsnap2-champion-512：与 v2 同一个基线，
#                    这样 turn 1-16 的选择与 v2 逐条一致，只多出 17/18）
#   HSAI_LIB_OUT     输出目录前缀（默认 artifacts/libraries/arena-cal-v3）
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

panel="${HSAI_LIB_PANEL:-artifacts/runs/gap-calsnap2-champion-512/panel.json}"
out="${HSAI_LIB_OUT:-artifacts/libraries/arena-cal-v3}"
pool="${HSAI_LIB_POOL:-data/real_library/firestone-bank-6k/train-m6k.jsonl}"
per_turn="${HSAI_LIB_PER_TURN:-96}"

if [[ -e "$out-snap" ]]; then echo "output exists: $out-snap" >&2; exit 2; fi
mkdir -p "$out"
echo "start=$(date -Is) git=$(git rev-parse HEAD) node=$(hostname)"

if [[ -f "$out/boards.jsonl" && "${HSAI_LIB_FORCE:-0}" != "1" ]]; then
  echo "boards.jsonl 已存在（$(wc -l < "$out/boards.jsonl") 行），跳过建库"
else
  python tools/build_arena_library.py \
    --pool "$pool" --panel "$panel" --panel-model champion14858 \
    --out-dir "$out" --per-turn "$per_turn"
fi

python tools/build_snapshot_library.py \
  --rows "$out/boards.jsonl" --out-dir "$out-snap" --fidelity full --seat 0

echo "done=$(date -Is)"
