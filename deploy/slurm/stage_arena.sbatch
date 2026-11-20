#!/usr/bin/env bash
#SBATCH --job-name=hsai-arena
#SBATCH --account=allusers
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:2
#SBATCH --exclude=xgpi15,xgpi17,xgpi11
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --propagate=NONE
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/runs/slurm/%x-%j.out
#SBATCH --error=artifacts/runs/slurm/%x-%j.err
#
# Formal training on the CALIBRATED snapshot field.
# 超参 = AutoTavern 论文 Table 7（arXiv:2610.33845 §6），**2026-11-16 起全部回到论文原值**：
# lr 6e-06→5e-06、有效 minibatch 8192（minibatch 1024 x grad-accum 8）、clip 0.2、target-KL 0.05、
# lambda 0.995、entropy 0.002、grad-clip 1.0、KL anchor beta=0.1 仅前 100 轮、sims 0。
# （v8 加探索试过 target-KL 0.10/0.15 + entropy 0.008 + anchor 0.02 全程 + lr 1.2e-05，
#   三轮实测证伪：ent 一路升到 1.49 但棋力无进步，见 D-065 ⇒ 已全部回退。）
# max-updates 300 = 论文一个 stage 的**轮数**。注意 --max-updates 的单位是「训练轮」
# （PPOTrainer.update() 每轮调用一次才 scheduler.step()），不是优化步数 —— 2026-11-14
# 曾按 15.5 步/轮错算成 4650，导致 302 轮只退火 6.5%、lr 全程 5.94e-06（见 TRAINING_RECIPE §8 K4）。
#
# 与论文的两处**已知差异**（用户 2026-11-14 已确认）：
#   1) 起点是冠军权重（--checkpoint，新 Adam），不是从零；论文每 stage 从上一 stage 继续。
#      （不能带 --trainer-state：PPOTrainer.load 会用 state 的 base_lrs 覆盖 --lr，
#       那样学习率会被强制回 1.667e-06，论文的 6e-06 就失效。）
#   2) 奖励除终局名次效用外，多挂一个**升本节奏项**（论文没有），参数见下面 tier_*。
# 规模口径：512 局/轮 ≈ 12.7 万决策 = 论文"一轮约 100 万决策"的 1/8；
# 所以一次 3 小时作业约等于论文一个 stage 的 1/8 训练量。
#
#   HSAI_ARENA_OUT      new run dir (must not exist)   [required]
#   HSAI_ARENA_CKPT     starting weights               [required, 2026-11-17]
#   HSAI_ARENA_MINUTES  training minutes for THIS segment (default 165)
#   HSAI_ARENA_SEGMENTS segments in this job (default 1)
#   HSAI_ARENA_RESUME   0=fresh fork from --checkpoint, 1=continue this run dir
#
# 分段续训**安全**（2026-11-14 核实并更正）：anchor 比的是 ``i = start_i + 本段轮数``，
# 而 ``start_i = trainer.updates`` 在 ppo_trainer.py:381 里是**每轮 update() 加 1**，
# 即**累计训练轮数**（证据：正式作业末行 `[iter 302] ... updates=302`）。所以续训时
# 第二段的 i 从上一段结束的轮数继续，β=0.1 的保护期照常按累计轮数走，不会被跳过。
# 之前这里写的"i 是优化步数、续训会跳几万"是**错的**（把 ppo 内部的 micro-step 计数
# 当成了 trainer.updates）。
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
minutes="${HSAI_ARENA_MINUTES:-165}"
iterations="${HSAI_ARENA_ITERATIONS:-0}"
stop_decisions="${HSAI_ARENA_STOP_DECISIONS:-0}"
segments="${HSAI_ARENA_SEGMENTS:-1}"
resume="${HSAI_ARENA_RESUME:-0}"
# 2026-11-17 用户定：**必须显式给**。以前默认是 `arena-v1.2/model.pt`，忘记覆盖就会
# 静默从旧模型起训（审核发现的隐患 #1）；现在缺参数直接报错，不允许默认值替人做决定。
ckpt="${HSAI_ARENA_CKPT:?must set HSAI_ARENA_CKPT (starting weights, e.g. artifacts/models/arena-v1.3-r3/model.pt)}"
snaps="${HSAI_ARENA_SNAPS:-artifacts/libraries/arena-cal-v14-snap}"
seed="${HSAI_ARENA_SEED:-611071401}"
lr="${HSAI_ARENA_LR:-6e-06}"
games="${HSAI_ARENA_GAMES:-512}"
procs="${HSAI_ARENA_PROCS:-8}"
epochs="${HSAI_ARENA_EPOCHS:-1}"
anchor_beta="${HSAI_ARENA_ANCHOR:-0.1}"
anchor_iters="${HSAI_ARENA_ANCHOR_ITERS:-100}"
anchor_tail="${HSAI_ARENA_ANCHOR_TAIL:-0.02}"
# 升本节奏奖励（唯一新增奖励）。
# v4 调整（2026-11-14 第二轮，用户要求"在不妨碍名次奖励的前提下尽量加"）：
#   旧：2:2,5:3,7:4,8:5，coef 0.25 / bonus 0.15 / penalty 0.10
#       → 实测 302 轮后候选 tier 反而比冠军低 0.5~0.7，且第 9 回合后**没有任何检查点**
#       → 早期三档在 90% 的局里都命中（等于白送分），唯一有约束的 T8 也只有 30% 命中
#   新：只留**有约束力**的三档（命中率按候选实测直方图算：T8 30% / T11 36% / T14 12%）
#       coef 1.0、命中 +0.20、落空 −0.15
#       → 每局期望 −0.177（名次全距 2.0 的 8.9%），旧版是 −0.048（2.4%）⇒ 压力约 3.7 倍
#       → 最坏情形（三档全错）−0.45 = 22.5%，仍明显小于名次信号 ±1.0
# -- 升本课程：纯教师 BC（DAgger），完全不强制（2026-11-17 用户定）------------
# 计划表是**四个**里程碑（不再是五个），语义是「教师时间表」而不是「强制时间表」：
#   2 回合 -> 2 本 / 5 -> 3 / 7 -> 4 / 8 -> 5        （**没有 6 本这一档**）
# 2026-11-17 用户决定：去掉原来的「10 -> 6」。教师不再把模型推向 6 本，
#   什么时候上 6 由策略自己决定。同时修掉脚本注释与实际值不一致（注释曾写 12、值是 10）。
# 语义（hsbg/az/train/forced_tier.py）：
#   prob=0   -> out.force 永不置位，动作 100% 由策略自己采样（不再有 logprob=0 的决策）
#   teacher  -> 到点且 Upgrade 合法（掩码 mask[3] 已编码「金币够+未满级+不在待选」）时，
#               每回合最多写一条 record["teacher_action"] = 3；
#               金币不够 / 待选中 / 已满级 -> 只记 blocked，不写标签
#   until=1B -> 教师标签全程开启（三轮长训全程不放手）
#   coef=0.1 -> PPO 里 0.1 * CE(pi_theta, a_teacher) 的辅助 BC 权重
# 重要特性：若模型自己提前升到目标等级，该计划点记为 achieved_by=self，
#   之后不再写标签 —— 教师只补「到点还没升」的缺口，不惩罚提前升。
# 注意：教师由 --forced-teacher-until / --forced-teacher-coef 这两个参数开启，
#   不存在 --forced-tier-teacher 这个 flag。
forced_tier_plan="${HSAI_ARENA_FORCED_TIER:-2:2,5:3,7:4,8:5}"
forced_tier_carry="${HSAI_ARENA_FORCED_TIER_CARRY:-1}"
forced_tier_prob="${HSAI_ARENA_FORCED_TIER_PROB:-0}"
forced_teacher_until="${HSAI_ARENA_FORCED_TEACHER_UNTIL:-1B}"
forced_teacher_coef="${HSAI_ARENA_FORCED_TEACHER_COEF:-0.1}"
# 升本奖励本轮**关闭**：强制到 6 本后，任何"到 5 本/6 本"的软目标都必然白送分，等于常数额赏。
# v7 起把**陈旧参数一并清空**（2026-11-15 审核发现）：以前虽 coef=0 不起作用，但 table/gate_ref
# 还挂着 v4 的值，万一有人把 coef 改回非 0，档位和门控 ref 对不上会导致整档被无条件挡掉。
# **要重新开启就三个一起给**：--tier-target-coef >0、--tier-target-table、--tier-gate-ref（turn 必须对齐）。
tier_coef="${HSAI_ARENA_TIER_COEF:-0.0}"
tier_bonus="${HSAI_ARENA_TIER_BONUS:-0.20}"
tier_table="${HSAI_ARENA_TIER_TABLE:-}"
tier_gate_k="${HSAI_ARENA_TIER_GATE_K:-0.0}"
tier_gate_ref="${HSAI_ARENA_TIER_GATE_REF:-}"
tier_penalty="${HSAI_ARENA_TIER_PENALTY:-0.15}"
# 冠军奖金（2026-11-16 用户要求「提高吃鸡奖励」，激进但不过火）。
# u(1) 由 1.0 → 1.0+b，其余名次**完全不变** ⇒ 冠军/亚军 = 1.40× → 2.38×。
# 同一张效用表同时用于终局奖励与 V(s)=Σp·u，两边恒等。0 = 论文原值。
win_bonus="${HSAI_ARENA_WIN_BONUS:-0}"
# 2026-11-17（用户定）：**只奖励前二**。整张表直接给，win_bonus 置 0。
#   2 名→3 名断崖 1.35（原设计 0.286 的 4.7 倍）；3–8 名之间只留 0.15 梯度。
rank_utility_table="${HSAI_ARENA_RANK_UTILITY:-1.00,0.35,-1.00,-1.15,-1.30,-1.45,-1.60,-1.75}"

move_cap=2
move_guard=1
move_repeat=2
max_actions=100
unfreeze_ban=1
freeze_cap="${HSAI_ARENA_FREEZE_CAP:-1}"
ban_trinkets="${HSAI_ARENA_BAN_TRINKETS:-1}"
mirror_strict=0
mirror_sample=p50band
action_profile="${HSAI_ACTION_PROFILE:-v14}"
if [[ "$action_profile" =~ ^human-turn-3702(-open|-standard|-economy|-economy-move8)?$ || "$action_profile" == human-turn-3703-player150 ]]; then
  move_cap=4
  unfreeze_ban=0
  freeze_cap=2
  ban_trinkets=0
  mirror_strict=1
  mirror_sample=random
  forced_tier_plan=""
  forced_tier_carry=0
  forced_tier_prob=0
  forced_teacher_coef=0
  forced_teacher_until=0
fi
if [[ "$action_profile" == human-turn-3703-player150 ]]; then
  max_actions=150
  move_cap=3
  move_repeat=0
fi
if [[ "$action_profile" == human-turn-3702-open ]]; then
  move_guard=0
  move_cap=0
  freeze_cap=0
  max_actions=200
fi
if [[ "$action_profile" =~ ^human-turn-3702-(standard|economy|economy-move8)$ ]]; then
  move_cap=10
  move_repeat=0
  freeze_cap=3
  max_actions=200
  if [[ "$action_profile" != human-turn-3702-standard ]]; then max_actions=300; fi
  if [[ "$action_profile" == human-turn-3702-economy-move8 ]]; then move_cap=8; fi
fi
[[ -d "$snaps" ]] || { echo "missing snapshot library: $snaps" >&2; exit 2; }
# --- GPU 落位：MIG UUID 重定向（2026-11-15 用户授权）---------------------------
# 背景：本站 h100-47 是 MIG 3g.47gb 切片（46.4 GiB/片），每节点 4 片，映射固定为
#   gres 0,1 → 物理卡0 ；gres 2,3 → 物理卡1 。
# Slurm 申请 2 片时总挑编号最小的连续两片 ⇒ 大概率落在同一张物理卡上；而 Slurm 只设置
# **数字** CUDA_VISIBLE_DEVICES=0,1，CUDA 会把它解释成"物理卡0 和 物理卡1"，第 2 张不存在
# ⇒ torch 只看到 1 张（这就是以前要靠"抽签"的全部原因）。
#
# 解法（mig_probe 1143745 实测有效）：从 nvidia-smi -L 里读出**本作业已被分配**的 MIG UUID，
# 用它重写 CUDA_VISIBLE_DEVICES，CUDA 就按 MIG 实例枚举 ⇒ 两张切片都可用，与落在哪张
# 物理卡无关。MIG 切片的 SM 是独立分区的，所以 learner 与采集仍能真并行（同卡时共享
# 显存带宽/L2，吞吐可能略低于跨卡，但不再有 40% 的串行惩罚）。
#
# 安全边界：只使用 nvidia-smi 在**当前分配**下报出来的 UUID（不凭空指定、不扩大资源）；
# 原始值、UUID、改写后的 device_count 全部写进 run.env 可审计。
# 关掉：HSAI_ARENA_MIG_REWRITE=0；强制只跑跨物理卡：HSAI_ARENA_REQUIRE_DUAL=1。
cvd_orig="${CUDA_VISIBLE_DEVICES:-}"
mig_used=""
nvis="$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
if [[ "${HSAI_ARENA_MIG_REWRITE:-1}" == "1" && "$nvis" -lt 2 ]]; then
  mig_ids="$(nvidia-smi -L 2>/dev/null | grep -o 'MIG-[0-9a-f-]*' | paste -sd, -)"
  if [[ -n "$mig_ids" && "$mig_ids" != *,*,* ]]; then
    export CUDA_VISIBLE_DEVICES="$mig_ids"
    mig_used="$mig_ids"
    nvis="$(python -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 0)"
  fi
fi
echo "gpu-gate: node=$(hostname) SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} cvd_orig=${cvd_orig} mig_rewrite=${mig_used:-none} visible_gpus=$nvis"
actor_gpus="${HSAI_ARENA_ACTOR_GPUS:-}"
if [[ -z "$actor_gpus" ]]; then
  if [[ "$nvis" -ge 2 ]]; then
    actor_gpus=1
    echo "gpu-mode: split (learner=cuda:0, actors=cuda:1)"
  else
    actor_gpus=0
    echo "gpu-mode: SHARED fallback (only $nvis visible GPU) -> learner 与采集同卡，吞吐约 -40%"
  fi
fi
if [[ "${HSAI_ARENA_REQUIRE_DUAL:-0}" == "1" && "$nvis" -lt 2 ]]; then
  echo "NEED_DUAL_GPU: visible=$nvis (<2) on $(hostname); abort for resubmit" >&2
  exit 3
fi

if [[ "$resume" != "1" && -e "$out" ]]; then echo "output exists: $out" >&2; exit 2; fi
if [[ "$resume" == "1" && ! -f "$out/train/trainer.pt" ]]; then
  echo "resume 需要已有的 $out/train/trainer.pt" >&2; exit 2
fi
if [[ "$resume" == "1" ]]; then
  mkdir -p "$out"
else
  mkdir -p "$(dirname "$out")"
  mkdir "$out"
fi
{
  echo "start=$(date -Is)"
  echo "job=${SLURM_JOB_ID:-local} node=$(hostname)"
  echo "gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "gpu_placement=slurm_job_gpus=${SLURM_JOB_GPUS:-} cvd_orig=${cvd_orig} mig_rewrite=${mig_used:-none} visible=${nvis} actor_gpus=${actor_gpus}"
  echo "git=$(git rev-parse HEAD)"
  echo "resume=$resume ckpt=$ckpt seed=$seed lr=$lr win_bonus=$win_bonus rank_utility=$rank_utility_table"
  echo "field=SNAPSHOT $snaps (mirror-p 1, seats 7, strict $mirror_strict, sample $mirror_sample, max-turn 0)"
  echo "games=$games procs=$procs epochs=$epochs minutes=$minutes segments=$segments"
  echo "budget iterations=$iterations stop_decisions=$stop_decisions"
  echo "anchor beta=$anchor_beta iters=$anchor_iters tail=$anchor_tail reset=fresh_only"
  echo "forced-tier plan=$forced_tier_plan carry=$forced_tier_carry prob=$forced_tier_prob"
  echo "teacher until=$forced_teacher_until coef=$forced_teacher_coef"
  # 动作契约 / 规则限制：让 run.env 自证"这一轮用的是哪套限制"（2026-11-17 审计）
  echo "contract profile=$action_profile max_actions=$max_actions move_guard=$move_guard move_cap=$move_cap move_repeat=$move_repeat freeze_cap=$freeze_cap forbid_cancel=1 forbid_unfreeze=$unfreeze_ban ban_trinkets=$ban_trinkets lobby_exclude=${HSAI_ARENA_LOBBY_EXCLUDE:-<none>} sims=0"
  echo "KPI = T12 median board strength (M1). tier soft-reward coef=$tier_coef (0=off, forced tier replaces it)"
  echo "reward: terminal placement utility ONLY + tier_target coef=$tier_coef penalty=$tier_penalty bonus=0.15 table=$tier_table gate_k=$tier_gate_k gate_ref=$tier_gate_ref"
  echo "all other shaping = 0 (spell/comp/retained/shaping/terminal_strength)"
} | tee "$out/run.env"

# 续训**必须**用 --trainer-state：本训练器没有 --resume。完整 Trainer 含 Adam/调度器/
# RNG/anchor，只有 model.pt 不能精确续训（AGENTS.md §4）。段 1 = 从 --checkpoint fork；
# 段 2+ = 接着本目录的 model.pt + trainer.pt 续训。
# A fresh weight fork creates an anchor; every Trainer continuation preserves it.
first_args=(--checkpoint "$ckpt" --reset-anchor 1)
if [[ -n "${HSAI_ARENA_PARENT_TRAINER:-}" ]]; then
  first_args=(--checkpoint "$ckpt" --trainer-state "$HSAI_ARENA_PARENT_TRAINER" --reset-anchor 0)
fi
cont_args=(--checkpoint "$out/train/model.pt" --trainer-state "$out/train/trainer.pt" --reset-anchor 0)
if [[ "$resume" == "1" ]]; then first_args=("${cont_args[@]}"); fi

train_segment() {
python workflows/train_entity_sync.py \
  "$@" \
  --device cuda --out "$out/train" \
  --opponent bot --behavior net --pipeline 1 --rust-collector 0 --lockstep 1 \
  --mirror-snaps "$snaps" --mirror-seats 7 --mirror-p 1 --mirror-sample "$mirror_sample" \
  --mirror-strict "$mirror_strict" --mirror-max-turn 0 \
  --human-board-p 0 --human-boards "" \
  --entry-p 0 \
  --recruit-start-bank "${HSAI_RECRUIT_START_BANK:-}" --recruit-start-p "${HSAI_RECRUIT_START_P:-0}" \
  --games "$games" --iterations "$iterations" --minutes "$minutes" --stop-at-decisions "$stop_decisions" \
  --epochs "$epochs" --minibatch 1024 --grad-accum 8 --lr "$lr" --lr-min-ratio 0.8333333333 \
  --max-updates 300 --clip 0.2 --target-kl "${HSAI_ARENA_TARGET_KL:-0.05}" --gamma 1.0 --lam 0.995 \
  --entropy-coef "${HSAI_ARENA_ENTROPY:-0.002}" --pi-coef 0.0 --seq-coef 0.0 --max-grad-norm 1.0 \
  --win-bonus "$win_bonus" --rank-utility-table "$rank_utility_table" \
  --anchor-beta "$anchor_beta" --anchor-iterations "$anchor_iters" \
  --anchor-beta-tail "$anchor_tail" \
  --forced-tier-plan "$forced_tier_plan" --forced-tier-carry "$forced_tier_carry" \
  --forced-tier-prob "$forced_tier_prob" \
  --forced-teacher-until "$forced_teacher_until" \
  --forced-teacher-coef "$forced_teacher_coef" \
  --seed "$seed" --sims 0 --processes "$procs" --workers 1 \
  --snapshot-every 5 --keep-snapshots 3 \
  --local-batch-inference 0 --records-dtype fp32 --leaf-amp bf16 --anchor-amp 1 \
  --pipeline-guard abort --leaf-wait-ms 0 --threads 1 --amp bf16 \
  --gpus "$actor_gpus" --gpu-index-space visible --ipc-mode shm --leaf-fused 1 \
  --move-guard "$move_guard" --move-cap "$move_cap" --move-repeat-limit "$move_repeat" --forbid-cancel 1 \
  --pipeline-zero-step-fallback 1 --forbid-unfreeze "$unfreeze_ban" \
  --max-actions-per-turn "$max_actions" \
  --lobby-exclude "${HSAI_ARENA_LOBBY_EXCLUDE:-}" \
  --lobby-filter-probability "${HSAI_ARENA_LOBBY_FILTER_PROBABILITY:-1}" \
  --freeze-cap "$freeze_cap" --ban-trinkets "$ban_trinkets" \
  --tier-target-coef "$tier_coef" --tier-penalty "$tier_penalty" \
  --tier-bonus "$tier_bonus" \
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

