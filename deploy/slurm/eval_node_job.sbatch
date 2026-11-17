#!/usr/bin/env bash
#SBATCH --job-name=hsai-evalnode
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi15,xgpi17,xgpi11
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 一个作业 = 一块 GPU + P 个并发进程（默认 8）= 覆盖一段连续种子。
#
# 为什么是「1 卡多进程」而不是「1 卡 1 进程」（2026-11-17 实测）：
#   1 卡 1 进程 : 1.29 s/局（聚合 135 决策/s）—— 单进程喂不满 GPU
#   1 卡 8 进程 : 0.58 s/局（聚合 458 决策/s）—— 3.4 倍
#   纯 CPU 8 进程: 1.88 s/局（聚合 135 决策/s）—— 慢 3.3 倍，已否决
# 所以提速只能**加卡**，每卡仍配 8 进程。
#
# 环境变量：
#   HSAI_GAP_MODELS / HSAI_GAP_MIRROR_SNAPS / HSAI_GAP_OUT(本作业目录)
#   HSAI_GAP_SEED(本作业起始种子) / HSAI_GAP_GAMES(本作业总局数)
#   HSAI_NODE_PROCS(默认 8) / HSAI_GAP_FORCED_TIER(可空=自由升本)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

games="${HSAI_GAP_GAMES:?需要本作业局数}"
procs="${HSAI_NODE_PROCS:-8}"
seed0="${HSAI_GAP_SEED:?需要本作业起始种子}"
out="${HSAI_GAP_OUT:?需要本作业输出目录}"
snaps="${HSAI_GAP_MIRROR_SNAPS:?需要 HSAI_GAP_MIRROR_SNAPS}"
baseline="${HSAI_GAP_BASELINE:-artifacts/runs/human-baseline-20261112/baseline.json}"
[[ -d "$snaps" ]] || { echo "no snapshot dir: $snaps" >&2; exit 2; }
[[ -e "$out" ]] && { echo "output exists: $out" >&2; exit 2; }
mkdir -p "$out"

specs=()
IFS=';' read -ra pairs <<< "${HSAI_GAP_MODELS:?需要 HSAI_GAP_MODELS}"
for pair in "${pairs[@]}"; do specs+=(--checkpoint "$pair"); done
ft_args=()
[[ -n "${HSAI_GAP_FORCED_TIER:-}" ]] && ft_args+=(--forced-tier-plan "${HSAI_GAP_FORCED_TIER}")

per=$(( (games + procs - 1) / procs ))
{
  echo "start=$(date -Is)"
  echo "games=$games procs=$procs per_proc=$per seed0=$seed0 node=$(hostname) job=${SLURM_JOB_ID:-local}"
  echo "mirror_snaps=$snaps"
} > "$out/run.env"
echo "games=$games procs=$procs per_proc=$per seed0=$seed0"

t0=$(date +%s)
for i in $(seq 0 $((procs-1))); do
  s=$(( seed0 + i * per ))
  d="$out/p$i"
  mkdir -p "$d"
  (
    PYTHONPATH=engine/python:. python workflows/eval_human_gap.py \
      "${specs[@]}" --games "$per" --seed "$s" --device cuda:0 \
      --max-actions-per-turn 100 --move-cap 2 --move-repeat-limit 2 \
      --forbid-unfreeze 1 --freeze-cap "${HSAI_GAP_FREEZE_CAP:-1}" \
      --lobby-exclude "${HSAI_GAP_LOBBY_EXCLUDE:-}" \
      --human-baseline "$baseline" \
      --mirror-snaps "$snaps" \
      --mirror-sample "${HSAI_GAP_MIRROR_SAMPLE:-p50band}" \
      --mirror-strict "${HSAI_GAP_MIRROR_STRICT:-0}" \
      "${ft_args[@]}" \
      --out "$d/panel.json" > "$d/panel.log" 2>&1
    echo "$i rc=$?"
  ) &
done
wait
echo "procs_done elapsed=$(( $(date +%s) - t0 ))s" | tee -a "$out/run.env"
echo "done=$(date -Is)" | tee -a "$out/run.env"
