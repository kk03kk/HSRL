#!/usr/bin/env bash
# 一次评测 = N 个「1 卡 8 进程」作业（各覆盖一段连续种子）+ 1 个自动合并作业。
# 实测：1 卡 8 进程 = 0.58 s/局；3 卡并行 ⇒ 512 局约 100 s、1024 局约 200 s。
#
# 用法：
#   HSAI_GAP_MODELS="arena_v13=path.pt" \
#   HSAI_GAP_MIRROR_SNAPS=artifacts/libraries/arena-cal-v13hold-snap \
#   HSAI_GAP_OUT=artifacts/runs/gap-xxx HSAI_GAP_GAMES=512 HSAI_NODES=3 \
#   HSAI_GAP_SEED=930000 [HSAI_GAP_FORCED_TIER="..."]  bash 本脚本
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/projects/HSAI}"

total="${HSAI_GAP_GAMES:-512}"
nodes="${HSAI_NODES:-3}"
procs="${HSAI_NODE_PROCS:-8}"
base="${HSAI_GAP_SEED:-930000}"
out="${HSAI_GAP_OUT:?需要 HSAI_GAP_OUT（新目录）}"
[[ -e "$out" ]] && { echo "output exists: $out" >&2; exit 2; }
mkdir -p "$out"

per=$(( (total + nodes - 1) / nodes ))
echo "total=$total nodes=$nodes procs_per_node=$procs per_node=$per base_seed=$base"
echo "out=$out  forced_tier=${HSAI_GAP_FORCED_TIER:-<无，自由升本>}"

jids=()
for i in $(seq 0 $((nodes-1))); do
  s=$(( base + i * per ))
  jid=$(HSAI_GAP_GAMES="$per" HSAI_GAP_SEED="$s" HSAI_GAP_OUT="$out/node$i" \
        HSAI_NODE_PROCS="$procs" \
        sbatch --parsable --export=ALL --job-name="hsai-e$i" \
        deploy/slurm/eval_node_job.sbatch | tail -1)
  jids+=("$jid")
  echo "  node$i job=$jid seed=$s games=$per"
done

deps=$(IFS=:; echo "${jids[*]}")
m_jid=$(sbatch --parsable --dependency=afterany:"$deps" \
        --export=ALL,HSAI_GAP_OUT="$out",HSAI_NODES="$nodes",HSAI_NODE_PROCS="$procs" \
        deploy/slurm/eval_nodes_merge.sbatch | tail -1)
echo "MERGE job=$m_jid (afterany:$deps)"
echo "=== done ==="
