#!/usr/bin/env bash
#SBATCH --job-name=hsai-arena
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# Formal training on the CALIBRATED snapshot field.
# Recipe = the 36h champion's own numbers (512 games / 8 procs / 1 epoch /
# anchor beta 0.1 for the first 250 iters then 0.02), plus exactly one reward addition
# (tier tempo) and the calibrated opponent library converted to a snapshot library.
#
#   HSAI_ARENA_OUT      new run dir (must not exist)
#   HSAI_ARENA_MINUTES  training minutes for THIS segment (default 150)
#   HSAI_ARENA_RESUME   0=fresh fork from --checkpoint, 1=continue this run dir
#
# 双卡布局 = 冠军那条作业的原样：learner 独占 cuda(0 号卡)，8 个采集进程都用 1 号卡
# （``--gpus 1`` 是**索引列表**，不是"几张卡"；见 config/league-36h.json 的 "gpus": "1"）。
set -euo pipefail
cd "${HSRL_ROOT:-$SLURM_SUBMIT_DIR}"
source deploy/slurm/common.sh
hsrl_require_compute
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RAYON_NUM_THREADS=1
export HSRL_TORCH_THREADS=2 PYTHONUTF8=1 PYTHONUNBUFFERED=1

out="${HSAI_ARENA_OUT:?必须指定 HSAI_ARENA_OUT（新目录）}"
minutes="${HSAI_ARENA_MINUTES:-150}"
segments="${HSAI_ARENA_SEGMENTS:-1}"
resume="${HSAI_ARENA_RESUME:-0}"
ckpt="${HSAI_ARENA_CKPT:-artifacts/models/league-final-20261111-step014858/model.pt}"
snaps="${HSAI_ARENA_SNAPS:-artifacts/libraries/arena-cal-v2-snap}"
seed="${HSAI_ARENA_SEED:-611071401}"
lr="${HSAI_ARENA_LR:-3e-06}"
games="${HSAI_ARENA_GAMES:-512}"
procs="${HSAI_ARENA_PROCS:-8}"
epochs="${HSAI_ARENA_EPOCHS:-1}"
anchor_beta="${HSAI_ARENA_ANCHOR:-0.1}"
anchor_iters="${HSAI_ARENA_ANCHOR_ITERS:-250}"
anchor_tail="${HSAI_ARENA_ANCHOR_TAIL:-0.02}"
# 升本节奏奖励（唯一新增奖励）。量级: 4 档 x 0.15 x 0.25 = 0.15 = 名次全距 2.0 的 7.5%
tier_coef="${HSAI_ARENA_TIER_COEF:-0.25}"
tier_table="${HSAI_ARENA_TIER_TABLE:-2:2,5:3,7:4,8:5}"
tier_gate_k="${HSAI_ARENA_TIER_GATE_K:-0.75}"
tier_gate_ref="${HSAI_ARENA_TIER_GATE_REF:-2:1,5:30,7:100,8:150}"
# 惩罚从 0.02 提到 0.10：错一档 = 0.25x0.10 = 2.5% 名次全距，四档全错 = 10%
tier_penalty="${HSAI_ARENA_TIER_PENALTY:-0.10}"

[[ -d "$snaps" ]] || { echo "missing snapshot library: $snaps" >&2; exit 2; }
if [[ "$resume" != "1" && -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
if [[ "$resume" == "1" && ! -f "$out/train/trainer.pt" ]]; then
  echo "resume 需要已有的 $out/train/trainer.pt" >&2; exit 2
fi
mkdir -p "$out"
{
  echo "start=$(date -Is)"
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "git=$(git rev-parse HEAD)"
  echo "resume=$resume ckpt=$ckpt seed=$seed lr=$lr"
  echo "field=SNAPSHOT $snaps (mirror-p 1, seats 7, strict 0, sample p50band, max-turn 0)"
  echo "games=$games procs=$procs epochs=$epochs minutes=$minutes segments=$segments"
  echo "anchor beta=$anchor_beta iters=$anchor_iters tail=$anchor_tail reset=1"
  echo "reward: terminal placement utility ONLY + tier_target coef=$tier_coef penalty=$tier_penalty bonus=0.15 table=$tier_table gate_k=$tier_gate_k gate_ref=$tier_gate_ref"
  echo "all other shaping = 0 (spell/comp/retained/shaping/terminal_strength)"
} | tee "$out/run.env"

# 续训**必须**用 --trainer-state：本训练器没有 --resume。完整 Trainer 含 Adam/调度器/
# RNG/anchor，只有 model.pt 不能精确续训（AGENTS.md §4）。段 1 = 从 --checkpoint fork；
# 段 2+ = 接着本目录的 model.pt + trainer.pt 续训。
first_args=(--checkpoint "$ckpt")
cont_args=(--checkpoint "$out/train/model.pt" --trainer-state "$out/train/trainer.pt")
if [[ "$resume" == "1" ]]; then first_args=("${cont_args[@]}"); fi

train_segment() {
python workflows/train_entity_sync.py \
  "$@" \
  --device cuda --out "$out/train" \
  --opponent bot --behavior net --pipeline 1 --rust-collector 0 --lockstep 1 \
  --mirror-snaps "$snaps" --mirror-seats 7 --mirror-p 1 --mirror-sample p50band \
  --mirror-strict 0 --mirror-max-turn 0 \
  --human-board-p 0 --human-boards "" \
  --entry-p 0 \
  --games "$games" --iterations 0 --minutes "$minutes" \
  --epochs "$epochs" --minibatch 1024 --grad-accum 1 --lr "$lr" --lr-min-ratio 0.8333333333 \
  --max-updates 25000 --clip 0.2 --target-kl 0.05 --gamma 1.0 --lam 0.995 \
  --entropy-coef 0.002 --pi-coef 0.0 --seq-coef 0.0 --max-grad-norm 1.0 \
  --anchor-beta "$anchor_beta" --anchor-iterations "$anchor_iters" \
  --anchor-beta-tail "$anchor_tail" --reset-anchor 1 \
  --seed "$seed" --sims 0 --processes "$procs" --workers 1 \
  --snapshot-every 25 --keep-snapshots 2 \
  --local-batch-inference 0 --records-dtype fp32 --leaf-amp bf16 --anchor-amp 1 \
  --pipeline-guard abort --leaf-wait-ms 0 --threads 1 --amp bf16 \
  --gpus 1 --gpu-index-space visible --ipc-mode shm --leaf-fused 1 \
  --move-guard 1 --move-cap 5 --move-repeat-limit 2 --forbid-cancel 1 \
  --pipeline-zero-step-fallback 1 --forbid-unfreeze 1 \
  --tier-target-coef "$tier_coef" --tier-penalty "$tier_penalty" \
  --tier-target-table "$tier_table" \
  --tier-gate-k "$tier_gate_k" --tier-gate-ref "$tier_gate_ref" \
  --comp-target-coef 0.0 --shaping-coef 0.0 --terminal-strength-coef 0.0 \
  2>&1 | tee -a "$out/train.log"
}

for i in $(seq 1 "$segments"); do
  if [[ -e "$out/STOP_REQUESTED" ]]; then
    echo "STOP_REQUESTED 于段边界 $i/$segments，停止 $(date -Is)" | tee -a "$out/run.env"
    break
  fi
  echo "segment $i/$segments start $(date -Is)" | tee -a "$out/run.env"
  if [[ "$i" == "1" ]]; then
    train_segment "${first_args[@]}"
  else
    train_segment "${cont_args[@]}"
  fi
  echo "segment $i/$segments done $(date -Is)" | tee -a "$out/run.env"
done

echo "done=$(date -Is)" | tee -a "$out/run.env"
