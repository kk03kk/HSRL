#!/usr/bin/env bash
#SBATCH --job-name=hsai-gap-1shard
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi15,xgpi17,xgpi11
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --propagate=NONE
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# 单个分片 = 一个作业 = 一块 GPU = 一个进程。
# 为什么这么拆（2026-11-17 实测）：把 N 个进程塞进同一块 MIG 切片时，GPU 前向被串行化，
# 每局从 1.29 s 拖到 4.4 s，8 进程只剩 2.2× 加速。拆成独立作业、各自占卡后
# 每局回到 ~1.3 s，8 分片并行 ⇒ 512 局约 85 秒（理论）。
#
# 环境变量：
#   HSAI_GAP_MODELS / HSAI_GAP_MIRROR_SNAPS / HSAI_GAP_GAMES(本分片的局数)
#   HSAI_GAP_SEED(本分片起始种子) / HSAI_GAP_OUT(本分片目录) / HSAI_GAP_FORCED_TIER(可空)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUTF8=1 PYTHONUNBUFFERED=1

games="${HSAI_GAP_GAMES:?需要本分片局数}"
out="${HSAI_GAP_OUT:?需要本分片输出目录}"
seed="${HSAI_GAP_SEED:?需要本分片起始种子}"
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

{
  echo "start=$(date -Is)"
  echo "games=$games seed=$seed node=$(hostname) job=${SLURM_JOB_ID:-local}"
  echo "mirror_snaps=$snaps"
} > "$out/run.env"

PYTHONPATH=engine/python:. python workflows/eval_human_gap.py \
  "${specs[@]}" --games "$games" --seed "$seed" --device cuda:0 \
  --max-actions-per-turn 100 --move-cap 2 --move-repeat-limit 2 \
  --forbid-unfreeze 1 --freeze-cap "${HSAI_GAP_FREEZE_CAP:-1}" \
  --lobby-exclude "${HSAI_GAP_LOBBY_EXCLUDE:-}" \
  --human-baseline "$baseline" \
  --mirror-snaps "$snaps" \
  --mirror-sample "${HSAI_GAP_MIRROR_SAMPLE:-p50band}" \
  --mirror-strict "${HSAI_GAP_MIRROR_STRICT:-0}" \
  "${ft_args[@]}" \
  --out "$out/panel.json" > "$out/panel.log" 2>&1
rc=$?
echo "done=$(date -Is) rc=$rc" >> "$out/run.env"
exit $rc
