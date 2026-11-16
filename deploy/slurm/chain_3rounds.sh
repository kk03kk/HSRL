#!/usr/bin/env bash
# v1.3 三轮链：训练 ↔ 评测交替，**评测不设门槛**（afterany），最后一轮跑四模型统一评测。
#
#   轮1  v1.2        → 165min → r1        ↓ 评测 r1（1024 局，v14hold，无强制升本）
#   轮2  r1          → 165min → r2        ↓ 评测 r2
#   轮3  r2          → 165min → r3        ↓ 统一评测 v1.2 + r1 + r2 + r3（各 1024 局）
#
# 用法：bash deploy/slurm/chain_3rounds.sh
# 依赖：arena-cal-v14-snap（训练场）已建；arena-cal-v14hold-snap（考试场）已建。
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"

TRAIN_SNAP="artifacts/libraries/arena-cal-v14-snap"
EVAL_SNAP="artifacts/libraries/arena-cal-v14hold-snap"
START="artifacts/models/arena-v1.2/model.pt"
NODES="${HSAI_NODES:-3}"
PROCS="${HSAI_NODE_PROCS:-8}"
GAMES="${HSAI_GAP_GAMES:-1024}"

[[ -f "$START" ]]              || { echo "MISSING start $START" >&2; exit 2; }
[[ -d "$TRAIN_SNAP" ]]         || { echo "MISSING train field $TRAIN_SNAP" >&2; exit 2; }
[[ -d "$EVAL_SNAP" ]]          || { echo "MISSING eval field $EVAL_SNAP" >&2; exit 2; }

submit_train() {   # $1=round $2=ckpt $3=outdir  $4=dep(s)
  local r="$1" ckpt="$2" out="$3" dep="${4:-}"
  local dep_args=()
  [[ -n "$dep" ]] && dep_args=(--dependency="afterany:$dep")
  local jid
  jid=$(sbatch --parsable "${dep_args[@]}" --job-name="hsai-v13-r$r" \
        --export=ALL,HSAI_ARENA_OUT="$out",HSAI_ARENA_CKPT="$ckpt",HSAI_ARENA_SNAPS="$TRAIN_SNAP" \
        deploy/slurm/stage_arena.sbatch | tail -1)
  echo "$jid"
}

submit_eval() {    # $1=tag $2=models-spec $3=outdir $4=dep(s)
  local tag="$1" models="$2" out="$3" dep="${4:-}"
  local dep_args=()
  [[ -n "$dep" ]] && dep_args=(--dependency="afterany:$dep")
  local jids=() jid i s
  local per=$(( (GAMES + NODES - 1) / NODES ))
  for i in $(seq 0 $((NODES-1))); do
    s=$(( 930000 + i * per ))
    jid=$(sbatch --parsable "${dep_args[@]}" --job-name="hsai-ev$tag$i" \
          --export=ALL,HSAI_GAP_MODELS="$models",HSAI_GAP_MIRROR_SNAPS="$EVAL_SNAP",\
HSAI_GAP_GAMES="$per",HSAI_GAP_SEED="$s",HSAI_GAP_OUT="$out/node$i",HSAI_NODE_PROCS="$PROCS" \
          deploy/slurm/eval_node_job.sbatch | tail -1)
    jids+=("$jid")
  done
  local deps m
  deps=$(IFS=:; echo "${jids[*]}")
  m=$(sbatch --parsable --dependency="afterany:$deps" --job-name="hsai-ev${tag}m" \
      --export=ALL,HSAI_GAP_OUT="$out",HSAI_NODES="$NODES",HSAI_NODE_PROCS="$PROCS" \
      deploy/slurm/eval_nodes_merge.sbatch | tail -1)
  echo "$m"
}

R1="artifacts/runs/nus-arena-v1.3-r1"
R2="artifacts/runs/nus-arena-v1.3-r2"
R3="artifacts/runs/nus-arena-v1.3-r3"
E1="artifacts/runs/gap-v1.3-r1-1024-20261117"
E2="artifacts/runs/gap-v1.3-r2-1024-20261117"
E3="artifacts/runs/gap-v1.3-r3-1024-20261117"
EF="artifacts/runs/gap-v1.2-plus3-1024-20261117"

for d in "$R1" "$R2" "$R3" "$E1" "$E2" "$E3" "$EF"; do
  [[ -e "$d" ]] && { echo "EXISTS $d —— 请换目录或先移走" >&2; exit 2; }
done

echo "train_field=$TRAIN_SNAP  eval_field=$EVAL_SNAP  nodes=$NODES procs/node=$PROCS games=$GAMES"
echo

j_r1=$(submit_train 1 "$START" "$R1" "")
echo "轮1 训练  job=$j_r1  ckpt=$START  out=$R1"
j_e1=$(submit_eval r1 "v1.3_r1=$R1/train/model.pt" "$E1" "$j_r1")
echo "  评测 r1 job=$j_e1 (afterany:$j_r1)"
j_r2=$(submit_train 2 "$R1/train/model.pt" "$R2" "$j_e1")
echo "轮2 训练  job=$j_r2  ckpt=$R1/train/model.pt"
j_e2=$(submit_eval r2 "v1.3_r2=$R2/train/model.pt" "$E2" "$j_r2")
echo "  评测 r2 job=$j_e2"
j_r3=$(submit_train 3 "$R2/train/model.pt" "$R3" "$j_e2")
echo "轮3 训练  job=$j_r3  ckpt=$R2/train/model.pt"
j_e3=$(submit_eval r3 "v1.3_r3=$R3/train/model.pt" "$E3" "$j_r3")
echo "  评测 r3 job=$j_e3"

MODELS="v1.2=$START;v1.3_r1=$R1/train/model.pt;v1.3_r2=$R2/train/model.pt;v1.3_r3=$R3/train/model.pt"
if [[ -e "$EF" ]]; then
  echo "  统一评测已存在，跳过：$EF"
else
  j_ef=$(submit_eval final "$MODELS" "$EF" "$j_e3")
  echo "四模型统一评测 job=$j_ef  models=4  games=$GAMES  out=$EF"
fi
echo
echo "=== 提交完成（全部 afterany，不看评测结果，一律续训）==="
