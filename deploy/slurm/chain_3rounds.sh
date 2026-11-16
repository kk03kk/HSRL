#!/usr/bin/env bash
# v1.3 三轮链（2026-11-17 用户定：**没有中间评测**，三轮练完再统一评测）
#
#   轮1  v1.2  → 165min → r1
#   轮2  r1    → 165min → r2
#   轮3  r2    → 165min → r3
#   ↓ 统一评测：v1.2 + r1 + r2 + r3（各 1024 局，v14hold 场，无强制升本）
#
# 依赖全部 afterany（前一步无论成败都继续下一步，不做任何门槛判断）。
# 用法：bash deploy/slurm/chain_3rounds.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"

TRAIN_SNAP="artifacts/libraries/arena-cal-v14-snap"
EVAL_SNAP="artifacts/libraries/arena-cal-v14hold-snap"
START="artifacts/models/arena-v1.2/model.pt"
NODES="${HSAI_NODES:-3}"
PROCS="${HSAI_NODE_PROCS:-8}"
GAMES="${HSAI_GAP_GAMES:-1024}"

[[ -f "$START" ]]      || { echo "MISSING start $START" >&2; exit 2; }
[[ -d "$TRAIN_SNAP" ]] || { echo "MISSING train field $TRAIN_SNAP" >&2; exit 2; }
[[ -d "$EVAL_SNAP" ]]  || { echo "MISSING eval field $EVAL_SNAP" >&2; exit 2; }

R1="artifacts/runs/nus-arena-v1.3-r1"
R2="artifacts/runs/nus-arena-v1.3-r2"
R3="artifacts/runs/nus-arena-v1.3-r3"
EF="artifacts/runs/gap-v1.2-plus3-1024-20261117"
for d in "$R1" "$R2" "$R3" "$EF"; do
  [[ -e "$d" ]] && { echo "EXISTS $d —— 请换目录或先移走" >&2; exit 2; }
done

submit_train() {  # $1=round $2=ckpt $3=out  $4=dep
  local dep_args=()
  [[ -n "${4:-}" ]] && dep_args=(--dependency="afterany:$4")
  sbatch --parsable "${dep_args[@]}" --job-name="hsai-v13-r$1" \
    --export=ALL,HSAI_ARENA_OUT="$3",HSAI_ARENA_CKPT="$2",HSAI_ARENA_SNAPS="$TRAIN_SNAP" \
    deploy/slurm/stage_arena.sbatch | tail -1
}

echo "train_field=$TRAIN_SNAP"
echo "eval_field =$EVAL_SNAP  （只有最后统一评测用）"
echo "start=$START  nodes=$NODES procs/node=$PROCS games(每模型)=$GAMES"
echo

j1=$(submit_train 1 "$START"           "$R1" "")
echo "轮1 训练 job=$j1   ckpt=$START"
j2=$(submit_train 2 "$R1/train/model.pt" "$R2" "$j1")
echo "轮2 训练 job=$j2   ckpt=$R1/train/model.pt   (afterany:$j1)"
j3=$(submit_train 3 "$R2/train/model.pt" "$R3" "$j2")
echo "轮3 训练 job=$j3   ckpt=$R2/train/model.pt   (afterany:$j2)"
echo

# ---- 统一评测（三轮全部结束后；4 个模型 × 1024 局）----
MODELS="v1.2=$START;v1.3_r1=$R1/train/model.pt;v1.3_r2=$R2/train/model.pt;v1.3_r3=$R3/train/model.pt"
per=$(( (GAMES + NODES - 1) / NODES ))
jids=()
for i in $(seq 0 $((NODES-1))); do
  s=$(( 930000 + i * per ))
  jid=$(sbatch --parsable --dependency="afterany:$j3" --job-name="hsai-evfinal$i" \
        --export=ALL,HSAI_GAP_MODELS="$MODELS",HSAI_GAP_MIRROR_SNAPS="$EVAL_SNAP",\
HSAI_GAP_GAMES="$per",HSAI_GAP_SEED="$s",HSAI_GAP_OUT="$EF/node$i",HSAI_NODE_PROCS="$PROCS" \
        deploy/slurm/eval_node_job.sbatch | tail -1)
  jids+=("$jid")
  echo "统一评测 node$i job=$jid seed=$s games(每模型)=$per (afterany:$j3)"
done
deps=$(IFS=:; echo "${jids[*]}")
jm=$(sbatch --parsable --dependency="afterany:$deps" --job-name="hsai-evfinalm" \
     --export=ALL,HSAI_GAP_OUT="$EF",HSAI_NODES="$NODES",HSAI_NODE_PROCS="$PROCS" \
     deploy/slurm/eval_nodes_merge.sbatch | tail -1)
echo "统一评测 合并 job=$jm  ->  $EF/panel.json"
echo
echo "=== 提交完成：3 轮训练（链式，afterany）+ 1 次四模型统一评测 ==="
