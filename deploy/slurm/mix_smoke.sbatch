#!/usr/bin/env bash
#SBATCH --job-name=hsai-mix
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# L2 冒烟：behavior-mix（ε=0.2，carrier_or_spell 决策点用 π_search 采样）。
#
# 为什么不走 late_entry.sbatch / personal.py：`personal.py` 的 `validate_config` 只认
# `OPTIONS` 表里的键，`behavior-mix-*` 不在表里（它是 train_entity_sync 的 argparse 项），
# 于是配置里带这些键会立刻 `ValueError: 未知配置项`。把 7 个键加进 `personal.OPTIONS`
# 需要改 digest 文件，而 L1 链正在跑（改 digest 会让链**无法再 resume**）。
# 所以这里**直接**用与 L1 完全相同的 argv 调 `train_entity_sync.py`，只加 behavior-mix 参数，
# 保证与 L1 是单变量对比。argv 来源：`artifacts/runs/slurm/hsai-late-entry-1134647.out` 的 `[run]` 行。
#
#   HSAI_MIX_OUT  新的输出目录（必须不存在）
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_MIX_OUT:?必须指定 HSAI_MIX_OUT（新目录）}"
if [[ -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
mkdir -p "$out"
export HSRL_POOL_EVIDENCE_DIR="$(realpath -m "$out")/pool-evidence"
{
  echo "start=$(date -Is)"
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "git=$(git rev-parse HEAD)"
  echo "train_sha=$(python -c 'from hsbg.az.train.personal import training_source_digest as d;print(d())')"
  echo "mix_eps=${HSAI_MIX_EPS:-0.2}"
} | tee "$out/run.env"

python workflows/train_entity_sync.py \
  --checkpoint artifacts/models/league-final-20261111-step014858/model.pt \
  --device cuda --out "$out/train" \
  --opponent bot --behavior net --pipeline 0 --rust-collector 0 --lockstep 1 \
  --mirror-snaps artifacts/libraries/league-eot-v1.1-20261109/train \
  --mirror-seats 7 --mirror-p 1 --human-board-p 0 \
  --trainer-state artifacts/models/league-final-20261111-step014858/trainer.pt \
  --games 256 --iterations 0 --minutes 120 \
  --epochs 2 --minibatch 1024 --grad-accum 1 --lr 2e-06 --lr-min-ratio 0.8333333333 \
  --max-updates 25000 --clip 0.2 --target-kl 0.05 --gamma 1.0 --lam 0.995 \
  --entropy-coef 0.002 --pi-coef 0.0 --seq-coef 0.0 --max-grad-norm 1.0 \
  --anchor-beta 0.1 --anchor-iterations 250 --anchor-beta-tail 0.02 \
  --seed 398000401 --sims 0 --processes 12 --workers 1 \
  --mirror-sample p50 --mirror-seats 7 --mirror-p 1 --mirror-strict 0 --mirror-max-turn 4 \
  --human-boards data/real_library/firestone-bank-6k/train-m6k.jsonl \
  --human-board-p 1.0 --human-board-seats 7 --human-board-sample p50 --human-board-set-tier 1 \
  --human-board-fidelity full --human-board-fast-inject 1 --human-board-min-turn 9 \
  --entry-boards data/real_library/firestone-bank-6k/train-m6k.jsonl \
  --entry-p 0.5 --entry-turn-min 8 --entry-turn-max 8 --entry-sample p75 --entry-min-tier 5 \
  --entry-fidelity full --entry-fast-inject 1 --entry-set-health 0 \
  --max-actions-per-turn 80 --snapshot-every 25 --keep-snapshots 2 \
  --local-batch-inference 0 --records-dtype fp32 --leaf-amp bf16 --anchor-amp 1 \
  --pipeline-guard abort --leaf-wait-ms 0 --threads 1 --pipeline 1 --amp bf16 \
  --gpus 0 --gpu-index-space visible --ipc-mode shm --leaf-fused 1 \
  --move-guard 1 --move-cap 5 --move-repeat-limit 2 --forbid-cancel 1 \
  --pipeline-zero-step-fallback 1 --reset-anchor 1 --forbid-unfreeze 1 \
  --comp-target-coef 0.0 --comp-target-min-turn 10 --comp-target-cards config/strategy_cards.json \
  --comp-target-tier-w S:1.0,A:0.6,B:0.3 --comp-target-cap 1.0 --comp-target-min-hits 2 \
  --tier-target-coef 0.0 --tier-target-table 2:2,5:3,8:5 --tier-gate-k 0.75 \
  --shaping-coef 0.0 --shaping-mode log_strength --shaping-gamma 1.0 \
  --shaping-w-str 1.0 --shaping-w-tier 0.0 \
  --behavior-mix-eps "${HSAI_MIX_EPS:-0.2}" --behavior-mix-when carrier_or_spell \
  --behavior-mix-max-per-turn 6 --behavior-mix-topk 8 --behavior-mix-turns 1 \
  --behavior-mix-max-ms 500 --behavior-mix-exact 1 \
  --behavior-mix-report "$out/behavior_mix.jsonl" \
  --eval-panel bot --eval-every-decisions 0 --minutes 120.0 \
  2>&1 | tee "$out/train.log"

echo "done=$(date -Is)" | tee -a "$out/run.env"
