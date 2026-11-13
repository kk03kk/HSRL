#!/usr/bin/env bash
#SBATCH --job-name=hsai-spell
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
# L4：真·学习率解冻臂。
#
# 为什么要这个作业：`late_entry.sbatch` 走 `personal.py` → 总会带 `--trainer-state`，
# 而 `PPOTrainer.load()` 会用 trainer state 里的 `scheduler.base_lrs` 覆盖配置的 lr
# （`workflows/train_entity_sync.py:1932-1947` 已显式警告）。所以今天所有臂
# ——包括我误以为"lr 已翻倍"的 L3——实际生效 lr 都是 trainer 带来的 **1.667e-6**。
# 真正的解冻必须**不带 trainer-state**（新建 Adam，从权重分叉），这正是本作业做的：
#   `--checkpoint <champion model.pt>` + 不给 `--trainer-state` + `--lr 3e-6`
# 代价：丢掉 champion 的 Adam 动量（这是"新实验分叉"，不是续训；provenance 里写明）。
#
#   HSAI_LR_OUT  新输出目录（必须不存在）
#   HSAI_LR      基础学习率（默认 3e-06）
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_LR_OUT:?必须指定 HSAI_LR_OUT（新目录）}"
lr="${HSAI_LR:-3e-06}"
resume="${HSAI_LR_RESUME:-0}"
ckpt="${HSAI_LR_CKPT:-artifacts/models/league-final-20261111-step014858/model.pt}"
spell_coef="${HSAI_SPELL_COEF:-0.05}"
spell_k="${HSAI_SPELL_PER_GAME:-3}"
shaping_coef="${HSAI_SHAPING_COEF:-0.0}"
term_coef="${HSAI_TERM_COEF:-0.0}"
# anchor-beta 是"允许偏离 champion 多远"的旋钮：0.02 实测把策略钉在 KL≈0.25 平衡点
# （RUNS §5.73），继续同配方训练 100 轮名次不动（§5.78）。做成 env 以便做单参数扫描；
# 默认值不变 => 旧配方逐位一致。
anchor_beta="${HSAI_ANCHOR_BETA:-0.02}"
# 阵容覆盖奖励（comp-target）：每局上界 = coef·1.0，与名次项（0..1）同量级，
# 形状是"历史最高完成度的增量"（望远镜求和、有界、不鼓励短视贪心）。
# L4 把 S 级核心卡命中从 0.139 练到 0.089（RUNS §5.80），这杆枪正对它。
comp_coef="${HSAI_COMP_TARGET_COEF:-0.0}"
# 强制买法术（探索课程）：零频动作只能靠强制注入打开（动作条件奖励对它无效，§5.79）。
# per_game=0（默认）⇒ 不传这两个 flag，旧路径逐位一致。
fs_per_game="${HSAI_FORCED_SPELL_PER_GAME:-0}"
fs_turn_min="${HSAI_FORCED_SPELL_TURN_MIN:-6}"
fs_prob="${HSAI_FORCED_SPELL_PROB:-1.0}"
fs_teacher="${HSAI_FORCED_SPELL_TEACHER:-0}"
[[ -f "$ckpt" ]] || { echo "missing checkpoint: $ckpt" >&2; exit 2; }
if [[ "$resume" != "1" && -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
if [[ "$resume" == "1" && ! -f "$out/train/trainer.pt" ]]; then
  echo "resume 需要已有的 $out/train/trainer.pt" >&2; exit 2
fi
mkdir -p "$out"
export HSRL_POOL_EVIDENCE_DIR="$(realpath -m "$out")/pool-evidence"
{
  echo "start=$(date -Is)"
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "git=$(git rev-parse HEAD)"
  echo "train_sha=$(python -c 'from hsbg.az.train.personal import training_source_digest as d;print(d())')"
  echo "resume=$resume"
  echo "lr=$lr"
  echo "ckpt=$ckpt"
  echo "spell_bootstrap_coef=$spell_coef per_game=$spell_k"
  echo "shaping_coef=$shaping_coef (log_strength, w_tier=0)"
  echo "terminal_strength_coef=$term_coef"
  echo "anchor_beta=$anchor_beta"
  echo "comp_target_coef=$comp_coef"
  echo "forced_spell per_game=$fs_per_game turn_min=$fs_turn_min prob=$fs_prob teacher=$fs_teacher"
  # 说明：shaping_coef>0 时是势函数塑形（gamma 与训练一致 => 策略不变）；spell_coef>0 时是
  # 动作条件补贴（非策略不变，每局前 K 次买法术）。两者可同时开，但一次实验只开一个更干净。
  if [[ "$resume" == "1" ]]; then
    echo "mode=resume-from-fork (--resume, trainer state 里已是 $lr)"
  else
    echo "mode=fresh-adam-fork (NO --trainer-state)"
  fi
} | tee "$out/run.env"

START_ARGS=(--checkpoint "$ckpt")
if [[ "$resume" == "1" ]]; then
  START_ARGS=(--resume)
fi
# 法术自举补贴参数**只在真的开了才传**：main 分支上还没有这两个 argparse 项，
# 传了会 "unrecognized arguments" 直接失败；塑形臂（spell=0）必须在未打补丁的代码上也能跑。
SPELL_ARGS=()
if [[ "${spell_coef}" != "0" && -n "${spell_coef}" ]]; then
  SPELL_ARGS=(--spell-bootstrap-coef "$spell_coef" --spell-bootstrap-per-game "$spell_k")
fi
# 强制买法术同理：per_game=0 时**不传**，未打补丁的代码也能跑。
FORCED_SPELL_ARGS=()
if [[ "${fs_per_game}" != "0" && -n "${fs_per_game}" ]]; then
  FORCED_SPELL_ARGS=(--forced-spell-per-game "$fs_per_game" \
    --forced-spell-turn-min "$fs_turn_min" \
    --forced-spell-prob "$fs_prob" \
    --forced-spell-teacher "$fs_teacher")
fi
if [[ "${HSAI_DRY:-0}" == "1" ]]; then
  echo "[dry-run] ckpt=$ckpt lr=$lr shaping=$shaping_coef spell=$spell_coef/$spell_k resume=$resume"
  echo "[dry-run] spell_args=${SPELL_ARGS[*]:-<none>}"
  exit 0
fi

python workflows/train_entity_sync.py \
  "${START_ARGS[@]}" \
  --device cuda --out "$out/train" \
  --opponent bot --behavior net --pipeline 0 --rust-collector 0 --lockstep 1 \
  --mirror-snaps artifacts/libraries/league-eot-v1.1-20261109/train \
  --mirror-seats 7 --mirror-p 1 --human-board-p 0 \
  --games "${HSAI_LR_GAMES:-256}" --iterations 0 --minutes "${HSAI_LR_MINUTES:-150}" \
  --epochs 2 --minibatch 1024 --grad-accum 1 --lr "$lr" --lr-min-ratio 0.8333333333 \
  --max-updates 25000 --clip 0.2 --target-kl 0.05 --gamma 1.0 --lam 0.995 \
  --entropy-coef 0.002 --pi-coef 0.0 --seq-coef 0.0 --max-grad-norm 1.0 \
  --anchor-beta "$anchor_beta" --anchor-iterations 0 --anchor-beta-tail "$anchor_beta" \
  --seed "${HSAI_LR_SEED:-398000601}" --sims 0 --processes "${HSAI_LR_PROCS:-12}" --workers 1 \
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
  --comp-target-coef "$comp_coef" --comp-target-min-turn 10 --comp-target-cards config/strategy_cards.json \
  --comp-target-tier-w S:1.0,A:0.6,B:0.3 --comp-target-cap 1.0 --comp-target-min-hits 2 \
  --tier-target-coef 0.0 --tier-target-table 2:2,5:3,8:5 --tier-gate-k 0.75 \
  --shaping-coef "$shaping_coef" --shaping-mode log_strength --shaping-gamma 1.0 \
  --terminal-strength-coef "$term_coef" \
  --shaping-w-str 1.0 --shaping-w-tier 0.0 \
  --eval-panel bot --eval-every-decisions 0 \
  ${SPELL_ARGS[@]+"${SPELL_ARGS[@]}"} \
  ${FORCED_SPELL_ARGS[@]+"${FORCED_SPELL_ARGS[@]}"} \
  2>&1 | tee "$out/train.log"

echo "done=$(date -Is)" | tee -a "$out/run.env"
