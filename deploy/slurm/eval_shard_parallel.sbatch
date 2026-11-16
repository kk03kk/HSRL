#!/usr/bin/env bash
#SBATCH --job-name=hsai-gap-shard
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
# 分片并行评测：一个作业内起 N 个进程，各自跑一段**互不重叠的连续种子块**，
# 全部结束后用 tools/merge_panels.py 合并。
#
# 为什么能提速：eval_human_gap.py 本身是串行的（一局一局跑），单进程只用 1 核，
# 但引擎是 Rust、模型前向极小 ⇒ CPU 是瓶颈。并行进程近乎线性加速，
# 且每个进程仍用同一块 GPU（模型只有 68 MB，多进程共享没问题）。
#
# 关键约束：所有模型必须用**同一个 base seed + 同一套分片**，否则跨模型不再是
# 同一批种子，配对 bootstrap 失效。本脚本固定由 HSAI_GAP_SEED 派生，勿手改单片。
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

total="${HSAI_GAP_GAMES:-1024}"
nshard="${HSAI_SHARDS:-8}"
base="${HSAI_GAP_SEED:-930000}"
out="${HSAI_GAP_OUT:?需要 HSAI_GAP_OUT（新目录）}"
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

per=$(( (total + nshard - 1) / nshard ))
{
  echo "start=$(date -Is)"
  echo "total_games=$total shards=$nshard per_shard=$per base_seed=$base"
  echo "mirror_snaps=$snaps"
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
} | tee "$out/run.env"

t0=$(date +%s)
pids=()
for i in $(seq 0 $((nshard-1))); do
  s=$(( base + i * per ))
  d="$out/shard$i"
  mkdir -p "$d"
  (
    PYTHONPATH=engine/python:. python workflows/eval_human_gap.py \
      "${specs[@]}" \
      --games "$per" --seed "$s" --device cuda:0 \
      --forbid-unfreeze 1 --freeze-cap "${HSAI_GAP_FREEZE_CAP:-2}" \
      --human-baseline "$baseline" \
      --mirror-snaps "$snaps" \
      --mirror-sample "${HSAI_GAP_MIRROR_SAMPLE:-p50band}" \
      --mirror-strict "${HSAI_GAP_MIRROR_STRICT:-0}" \
      "${ft_args[@]}" \
      --out "$d/panel.json" > "$d/panel.log" 2>&1
    echo "$i exit=$?"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
t1=$(date +%s)
echo "shards_done elapsed=$((t1-t0))s" | tee -a "$out/run.env"

shards=()
for i in $(seq 0 $((nshard-1))); do
  [ -s "$out/shard$i/panel.json" ] && shards+=("$out/shard$i/panel.json")
done
echo "collected ${#shards[@]}/${nshard} shard panels"
if [ "${#shards[@]}" -gt 0 ]; then
  python tools/merge_panels.py "${shards[@]}" --out "$out/panel.json" | tee -a "$out/run.env"
fi
echo "done=$(date -Is) total_elapsed=$(( $(date +%s) - t0 ))s" | tee -a "$out/run.env"

